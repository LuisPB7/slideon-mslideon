"""Complete SlideOn model: Transformer, token projections, and sparse latent head."""

# Tensor-facing model code. models.py:SlideOn is the separate text/tokenization
# adapter inside Sentence Transformers; this file supplies the Transformers
# config and network loaded by that adapter through AutoConfig/AutoModel.
#
# Shape vocabulary used below: B = text sequences, L = padded token length,
# H = backbone hidden width (768 for English), N = valid tokens in the batch,
# V = latent vocabulary size (32,768), and K = selected features per token (32).
# Sparse representations here are dense tensors with many zero entries.
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel


# A saved checkpoint's architecture/inference contract. This is different from
# the editable training JSON: no optimizer, sampling or loss schedule lives here.
# Defaults retain optional shared paths, but English uses no lift/projections.
class SlideOnConfig(PretrainedConfig):
    # Transformers uses model_type for dispatch and sub_configs to recognize
    # the nested pretrained backbone configuration during loading/serialization.
    model_type = "slideon"
    sub_configs = {"backbone_config": AutoConfig}

    # Accept config objects or the dictionaries read from a saved config.json.
    # These dictionaries describe the backbone, optional token projections, SAE
    # head shape, text preprocessing, and query/document inference settings.
    def __init__(self, backbone_config=None, projection_configs=None, head_config=None,
                 inference=None, tokenizer_settings=None, skip_token_ids=None,
                 checkpoint_type="retrieval", **kwargs):
        super().__init__(**kwargs)
        # Turn the nested backbone dictionary into the appropriate Transformers
        # config class before constructing its AutoModel. Copy before removing
        # model_type so the caller's dictionary is not modified in place.
        if isinstance(backbone_config, dict):
            backbone_config = dict(backbone_config)
            model_type = backbone_config.pop("model_type", "modernbert")
            backbone_config = AutoConfig.for_model(model_type, **backbone_config)
        self.backbone_config = backbone_config or AutoConfig.for_model("modernbert")
        self.projection_configs = projection_configs or []
        self.head_config = head_config or dict(d_model=768, n_latents=32768, d_lift=0)

        # token_k is a per-token budget. query/document_max_active_dims are
        # separate final-vector budgets, with None meaning no final pruning in
        # these defaults. Sequence length limits and chunk size are independent.
        self.inference = inference or dict(
            token_k=32, query_max_active_dims=None, document_max_active_dims=None,
            query_length=8192, document_length=8192, head_chunk_size=2048,
            prompts={"query": "", "document": ""}, pooling="max_relu", score="dot")
        self.tokenizer_settings = tokenizer_settings or {}
        self.skip_token_ids = skip_token_ids or []
        self.checkpoint_type = checkpoint_type
        # Fail on incompatible checkpoint semantics instead of silently loading
        # weights into a model with a different pooling or scoring definition.
        if self.inference.get("pooling") != "max_relu" or self.inference.get("score") != "dot":
            raise ValueError("SlideOn supports max_relu pooling and dot scoring")


# Retained for compatible multilingual/legacy checkpoints. English SlideOn has
# an empty projection_configs list, so it never constructs these extra layers.
class TokenProjection(torch.nn.Module):
    """The token-wise residual Dense layers used by mSlideOn."""

    # Map each token from in_features to out_features, optionally adding a
    # residual path. A width change needs a learned residual projection too.
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = torch.nn.Linear(
            config["in_features"], config["out_features"], bias=config["bias"]
        )
        self.use_residual = config.get("use_residual", False)
        self.residual = None
        if self.use_residual and config["in_features"] != config["out_features"]:
            self.residual = torch.nn.Linear(
                config["in_features"], config["out_features"], bias=False
            )
        if config["activation_function"] != "torch.nn.modules.linear.Identity":
            raise ValueError("Unsupported projection activation")

    # Linear layers act on the last axis; batch/sequence axes are preserved.
    # Identity activation is required by the constructor, so no extra token
    # nonlinearity is introduced on this optional projection path.
    def forward(self, tokens):
        projected = self.linear(tokens)
        if self.use_residual:
            projected = projected + (
                self.residual(tokens) if self.residual is not None else tokens
            )
        return projected


# Retrieval-side encoder of the SAE. It emits pre-activation scores [N, V];
# the enclosing model, not this class, applies Top-K, ReLU, pooling and gates.
# Raw reconstruction training instead uses TopKSAE in train_slideon_sae.py.
class LatentHead(torch.nn.Module):
    # W_enc has shape [H, V] for English, b_enc [V], and b_dec [H]. The center
    # b_dec is used when encoding even though its name comes from the decoder.
    # gate_relu_phi [V] is a retrieval-specific learnable post-pooling scale.
    def __init__(self, config):
        super().__init__()
        d_model, n_latents = config["d_model"], config["n_latents"]
        d_lift = config.get("d_lift") or 0
        self.W_enc = torch.nn.Parameter(torch.empty(d_lift or d_model, n_latents))
        self.b_enc = torch.nn.Parameter(torch.zeros(n_latents))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))
        self.gate_relu_phi = torch.nn.Parameter(torch.ones(n_latents))
        self.lift = None

        # Optional shared lift/residual-MLP encoder: project the centered token
        # before scoring latent features. English d_lift=0 skips this entirely.
        if d_lift:
            hidden = config["hidden_dim"]
            self.lift = torch.nn.Linear(d_model, d_lift, bias=False)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d_lift, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, d_lift))
        if config.get("has_decoder", False):
            # Preserve the pretrained SAE decoder without optimizing it in retrieval SFT.
            self.W_dec = torch.nn.Parameter(torch.empty(n_latents, d_model), requires_grad=False)
            self.register_buffer("steps_since_fired", torch.zeros(n_latents, dtype=torch.long))

    # Inputs are already normalized token states. Subtract the learned center
    # and compute all latent scores; retaining signed scores here matters for
    # the later Top-K selection and training-time softmax balance statistic.
    def forward(self, tokens):
        centered = tokens - self.b_dec
        if self.lift is not None:
            lifted = self.lift(centered)
            centered = lifted + self.mlp(lifted)
        return centered @ self.W_enc + self.b_enc


# Full Transformers network: contextual token backbone plus sparse latent head.
# Its forward result is a dictionary consumed by the outer SlideOn InputModule.
# No reconstruction loss/decoder call appears in this retrieval forward path.
class SlideOnModel(PreTrainedModel):
    # Library integration flags describe loading, attention and device placement.
    # Keep head parameters in FP32; its matmul can still run under BF16 autocast.
    config_class = SlideOnConfig
    base_model_prefix = "backbone"
    _supports_sdpa = True
    _supports_flash_attn = True
    _no_split_modules = ["ModernBertEncoderLayer", "LatentHead", "TokenProjection"]
    _keep_in_fp32_modules_strict = ["head"]

    # Construct architecture from config. Actual pretrained parameter values
    # are restored by from_pretrained; AutoModel.from_config alone initializes
    # a backbone rather than downloading DenseOn weights here.
    def __init__(self, config):
        super().__init__(config)
        config.backbone_config._attn_implementation = config._attn_implementation
        self.backbone = AutoModel.from_config(config.backbone_config)
        self.projections = torch.nn.ModuleList(
            TokenProjection(item) for item in config.projection_configs)
        self.head = LatentHead(config.head_config)
        # Share the config's settings dictionary so adapter/trainer updates are
        # visible to forward and preserved when the checkpoint is saved.
        self.settings = config.inference
        self.post_init()

    # Transformers calls this initialization hook before/around weight loading.
    # Fresh latent matrices use a small normal distribution; a loaded checkpoint
    # replaces them with its saved values, including its learned encoder bias.
    def _init_weights(self, module):
        if isinstance(module, LatentHead):
            torch.nn.init.normal_(module.W_enc, std=0.02)
            if hasattr(module, "W_dec"):
                torch.nn.init.normal_(module.W_dec, std=0.02)
        else:
            super()._init_weights(module)

    # Preserve Transformers' normal return contract, including optional loading
    # information. Apply two compatibility adjustments after restoring weights.
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        result = super().from_pretrained(*args, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        # Match the original PyLate encoder's RoPE buffer precision.
        model.backbone.to(dtype=model.backbone.dtype)
        # Loading tensors can reset the decoder's requires_grad flag.
        if hasattr(model.head, "W_dec"):
            model.head.W_dec.requires_grad_(False)
        return result

    # V is the vocabulary width of every output row. Selecting fewer active
    # latents leaves the same coordinate system and embedding dimension.
    def get_embedding_dimension(self):
        return self.config.head_config["n_latents"]

    # Mirror the adapter's role rules for callers using the core directly.
    # Role selects document skip tokens and the optional final pruning cap.
    @staticmethod
    def _task(task):
        task = task or "document"
        if task not in ("query", "document"):
            raise ValueError("task must be 'query' or 'document'")
        return task


    # Input token tensors are [B, L]. Return sentence_embedding [B, V] and,
    # only while training, a scalar load_balance auxiliary term. The function
    # performs encoding/pooling; the fine-tuning script defines ranking losses.
    def forward(self, input_ids, attention_mask, token_type_ids=None, task=None,
                slideon_task=None, max_active_dims=None, token_k=None, **kwargs):
        if "top_k" in kwargs:
            raise TypeError("top_k has been removed; use max_active_dims (None for unpruned)")
        # Carry the preprocessing role along with tensor fields. An explicit
        # forward task may be provided too, but it must agree with that role.
        features = {"input_ids": input_ids, "attention_mask": attention_mask}
        if slideon_task is not None or task is not None:
            features["slideon_task"] = slideon_task or task
        if token_type_ids is not None:
            features["token_type_ids"] = token_type_ids
        task = self._task(task or features.get("slideon_task"))
        if features.get("slideon_task", task) != task:
            raise ValueError("Preprocessing and forward tasks must match")
        n_latents = self.get_embedding_dimension()

        # None defers to the saved per-role cap, which is itself None for the
        # released unpruned defaults. This is distinct from token_k, which always
        # selects K coordinates independently for each valid token.
        cap = self.settings.get(task + "_max_active_dims") if max_active_dims is None else max_active_dims
        k = self.settings["token_k"] if token_k is None else token_k
        if cap is not None and (type(cap) is not int or cap <= 0):
            raise ValueError("max_active_dims must be a positive integer or None")
        if not isinstance(k, int) or not 1 <= k <= n_latents:
            raise ValueError(f"token_k must be an integer between 1 and {n_latents}")
        model_inputs = {
            key: features[key]
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in features
        }

        # Use contextual states [B, L, H], not DenseOn's pooled sentence vector.
        # The optional projection stack changes only the last dimension.
        tokens = self.backbone(**model_inputs).last_hidden_state
        for projection in self.projections:
            tokens = projection(tokens)

        # Attention masking removes padding. A configured document skip list
        # can additionally remove token IDs; English backbone-only conversion
        # clears that list, preserving ordinary special and prompt tokens.
        mask = features["attention_mask"].bool()
        if task == "document" and self.config.skip_token_ids:
            skip_ids = torch.tensor(self.config.skip_token_ids, device=input_ids.device)
            mask = mask & ~torch.isin(input_ids, skip_ids)

        # Flatten valid positions while recording which sequence owns each one.
        # owners [N] lets max pooling later restore the B separate text vectors.
        # L2 normalization is per token over H, before learned centering.
        owners = torch.arange(tokens.shape[0], device=tokens.device)[:, None].expand_as(
            mask
        )[mask]
        tokens = torch.nn.functional.normalize(tokens[mask], p=2, dim=-1)
        # Match the evaluation loader's FP16 token transfer before head pooling.
        if not self.training:
            tokens = tokens.half().float()

        # These two V-wide statistics implement training-time load balancing:
        # softmax probability mass and detached counts of selected coordinates.
        # They aggregate over valid tokens within this forward call, not across
        # distributed ranks or future gradient-accumulation microbatches.
        probability_sum = tokens.new_zeros(n_latents, dtype=torch.float32)
        usage_sum = tokens.new_zeros(n_latents, dtype=torch.float32)
        # Pool into a flattened [B * V] array so sequence ownership can be added
        # to latent indices. Zero initialization makes unused features zero and
        # makes an all-padding sequence return a zero vector.
        pooled = torch.zeros(
            mask.shape[0] * n_latents, dtype=torch.float32, device=tokens.device
        )

        # Bound the largest token-by-vocabulary activation matrix to [chunk, V].
        # head_chunk_size trades working memory against repeated kernel calls;
        # it does not change which tokens belong to the same pooled sequence.
        for start in range(0, tokens.shape[0], self.settings["head_chunk_size"]):
            end = start + self.settings["head_chunk_size"]
            # The reference evaluates the head under BF16 autocast on CUDA.
            # Compute pre-activations, Top-K values/indices, and an optional
            # differentiable softmax sum. All V scores contribute to the balance
            # probabilities, even though only K values contribute to pooling.
            def head_chunk(chunk):
                with torch.autocast(chunk.device.type, dtype=torch.bfloat16, enabled=chunk.is_cuda):
                    activations = self.head(chunk)
                    values, indices = activations.topk(k, dim=-1)
                probabilities = activations.float().softmax(-1).sum(0) if self.training else chunk.new_zeros(n_latents)
                return values, indices, probabilities

            # Activation checkpointing recomputes this head chunk on backward
            # instead of storing all [chunk, V] intermediates. This is separate
            # from optional backbone gradient checkpointing enabled by SFT.
            if self.training and torch.is_grad_enabled():
                values, indices, probabilities = activation_checkpoint(head_chunk, tokens[start:end], use_reentrant=False)
            else:
                values, indices, probabilities = head_chunk(tokens[start:end])
            # Count Top-K selections before ReLU: a selected negative score can
            # count toward usage even though its eventual retrieval weight is
            # zero. Counts have no gradient; probabilities carry the gradient.
            if self.training:
                probability_sum = probability_sum + probabilities
                usage_sum = usage_sum + torch.bincount(indices.flatten(), minlength=n_latents)

            # Enforce nonnegative impacts after selecting features. With V-wide
            # offsets, equal latent IDs in different sequences never collide;
            # repeated IDs within one sequence take their maximum token value.
            values = values.float().relu()
            indices = owners[start:end, None] * n_latents + indices
            pooled = pooled.scatter_reduce(
                0, indices.flatten(), values.flatten(), reduce="amax", include_self=True
            )

        # Gates rescale the pooled features and are themselves clamped by ReLU.
        # There is no log1p transformation or final vector L2 normalization.
        pooled = (
            pooled.view(mask.shape[0], n_latents)
            * self.head.gate_relu_phi.float().relu()
        )

        # Optional sequence-level Top-K comes after token pooling and gating.
        # It limits positive nonzeros to at most cap, without compressing V.
        # Caps larger than V are accepted and simply keep all coordinates.
        if cap is not None:
            # Prune before FP16 rounding
            values, indices = pooled.topk(min(cap, n_latents), dim=-1)
            pooled = torch.zeros_like(pooled).scatter(1, indices, values)
        # The reference stores retrieval weights in FP16; retain that rounding.
        features["sentence_embedding"] = pooled if self.training else pooled.half().float()

        # Balance = V * sum(mean_probability * selected_frequency). Frequencies
        # divide by N*K; a uniform distribution gives a reference value of one.
        # The caller weights this auxiliary term separately from ranking losses.
        # These are token-level selections before ReLU, independent of final
        # pooled-vector gates/pruning and the loss's activation-based FLOPS term.
        if self.training:
            count = max(tokens.shape[0], 1)
            features["load_balance"] = n_latents * (
                probability_sum / count * usage_sum.detach() / (count * k)
            ).sum()
        # Inference returns only embeddings. The adapter merges this compact
        # result back into its input feature dictionary for the outer wrapper.
        return {key: features[key] for key in ("sentence_embedding", "load_balance") if key in features}


# Register saved-code mappings so a self-contained checkpoint can reconstruct
# these classes with AutoConfig/AutoModel and trust_remote_code=True. Loading a
# Hub revision then follows that artifact's code, not edits in this checkout.
SlideOnConfig.register_for_auto_class()
SlideOnModel.register_for_auto_class("AutoModel")
