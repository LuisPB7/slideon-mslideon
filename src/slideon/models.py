"""Sentence Transformers adapter for the complete, flat SlideOn checkpoint."""

# This file connects two libraries: Sentence Transformers owns the outer
# SparseEncoder interface, while Transformers owns the tokenizer and neural
# model loaded below. The adapter turns text into tensors and passes results
# back through Sentence Transformers' feature dictionary.
#
# export.py copies this file into a checkpoint as slideon.py. A Hub checkpoint
# can therefore execute its bundled adapter, independently of this local file.
from pathlib import Path

import torch
from sentence_transformers.base.modules import InputModule
from transformers import AutoConfig, AutoModel, AutoTokenizer


# InputModule is the text-facing component inside SparseEncoder. The similarly
# named SlideOnModel is the tensor-facing core and is loaded through AutoModel.
class SlideOn(InputModule):
    # Sentence Transformers uses these declarations when saving/loading modules
    # and deciding which encode-time arguments to forward into the neural model.
    config_file_name = "config.json"
    forward_kwargs = {"task", "max_active_dims", "token_k"}
    save_in_root = True

    # checkpoint may be a local flat directory or a Hub model identifier.
    # Constructor kwargs are split by destination: model, tokenizer/processor,
    # config, and shared Hub options such as revision or cache location.
    def __init__(self, checkpoint, model_kwargs=None, processor_kwargs=None,
                 config_kwargs=None, **hub_kwargs):
        super().__init__()
        # Read the checkpoint's custom config first. trust_remote_code allows
        # its saved Python implementation to supply the config/model classes;
        # editing local modeling_slideon.py does not rewrite a pinned Hub model.
        config = AutoConfig.from_pretrained(
            checkpoint, trust_remote_code=True, **hub_kwargs, **(config_kwargs or {}))
        if config.model_type != "slideon":
            raise ValueError("Expected a flat SlideOn checkpoint; use slideon.export for legacy checkpoints")
        self.model = AutoModel.from_pretrained(
            checkpoint, config=config, trust_remote_code=True,
            **hub_kwargs, **(model_kwargs or {}))
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True, **hub_kwargs, **(processor_kwargs or {}))

        # Preserve the upstream padding-token convention. Attention masks, not
        # the identity of this token, determine which positions are discarded.
        if self.tokenizer.mask_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.mask_token_id
        elif self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Keep references to the model's config dictionaries. Fine-tuning edits
        # these settings in place, and saving the model then preserves them.
        # max_seq_length is an outer cap; preprocess also uses the role's cap.
        self.settings = self.model.config.inference
        self.encoder_config = self.model.config.tokenizer_settings
        self.max_seq_length = max(self.settings["query_length"], self.settings["document_length"])

    # Return the latent vocabulary width V, not the backbone hidden width H.
    # A final sparsity budget changes nonzeros, never this embedding dimension.
    def get_embedding_dimension(self):
        return self.model.get_embedding_dimension()

    # Missing roles default to document. Explicit role validation prevents
    # silently applying a query prompt or length to document inputs.
    @staticmethod
    def _task(task):
        task = task or "document"
        if task not in ("query", "document"):
            raise ValueError("task must be 'query' or 'document'")
        return task

    # Convert a list of strings into a batch feature dictionary. Tensor fields
    # have shape [B, L]; slideon_task is a string carried into forward so the
    # correct role-specific settings can be checked and applied later.
    def preprocess(
        self, inputs, prompt=None, task=None, processing_kwargs=None, **kwargs
    ):
        task = self._task(task)
        if not all(isinstance(text, str) for text in inputs):
            raise TypeError("SlideOn accepts text strings only")
        if prompt is None:
            prompt = self.settings["prompts"][task]

        # A textual prompt is prepended before tokenization. A separate optional
        # marker token (prefix below) is inserted after tokenization; English
        # backbone-only SAE conversion disables those marker prefixes.
        texts = [(prompt + text).strip() for text in inputs]
        prefix = self.encoder_config.get(task + "_prefix", "")
        length = min(self.max_seq_length, self.settings[task + "_length"])
        options = dict(
            padding=True,
            truncation=True,
            max_length=length - bool(prefix),
            return_tensors="pt",
        )
        # Caller-supplied tokenizer options override these defaults, including
        # lengths/padding. Leave one position for the optional marker first.
        options.update(processing_kwargs or {})
        features = dict(self.tokenizer(texts, **options))

        # Insert the marker after the first token and extend every aligned
        # tensor: token ID for input_ids, one for its mask, zero otherwise.
        if prefix:
            prefix_id = self.tokenizer.convert_tokens_to_ids(prefix)
            for name, value in list(features.items()):
                fill = (
                    prefix_id
                    if name == "input_ids"
                    else (1 if name == "attention_mask" else 0)
                )
                marker = torch.full(
                    (value.shape[0], 1), fill, dtype=value.dtype, device=value.device
                )
                features[name] = torch.cat((value[:, :1], marker, value[:, 1:]), dim=1)
        features["slideon_task"] = task
        return features

    # The core returns sentence_embedding [B, V] and, during training, a scalar
    # load_balance term. Merge them into the original feature dictionary so
    # downstream Sentence Transformers losses/modules retain the input fields.
    # token_k acts per token; max_active_dims acts after sequence pooling.
    def forward(self, features, task=None, max_active_dims=None, token_k=None, **kwargs):
        if "top_k" in kwargs:
            raise TypeError("top_k has been removed; use max_active_dims (None for unpruned)")
        inputs = {key: features[key] for key in
                  ("input_ids", "attention_mask", "token_type_ids", "slideon_task") if key in features}
        features.update(self.model(**inputs, task=task, max_active_dims=max_active_dims, token_k=token_k))
        return features

    # Save the core's weights/config and tokenizer in this module's root.
    # SparseEncoder's outer saving machinery handles its own module metadata.
    def save(self, output_path, safe_serialization=True, **kwargs):
        self.model.save_pretrained(output_path, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(output_path)

    # Sentence Transformers calls this hook when reconstructing the module
    # described by modules.json. Translate its cache_folder argument to the
    # cache_dir keyword expected by the Transformers/Hub loading functions.
    @classmethod
    def load(cls, model_name_or_path, subfolder="", token=None, cache_folder=None,
             revision=None, local_files_only=False, model_kwargs=None,
             processor_kwargs=None, config_kwargs=None, backend="torch", **kwargs):
        if backend != "torch":
            raise ValueError("SlideOn currently supports the torch backend only")
        return cls(model_name_or_path, model_kwargs, processor_kwargs, config_kwargs,
                   subfolder=subfolder, token=token, cache_dir=cache_folder,
                   revision=revision, local_files_only=local_files_only)
