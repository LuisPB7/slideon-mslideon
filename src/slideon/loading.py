"""Import raw SAE checkpoints without a dependency on the research repository."""

# Checkpoint formats are the main concern here, rather than neural operations.
# A flat checkpoint already bundles the backbone, sparse head and custom code.
# A raw SAE contains learned dictionary tensors plus a recipe and needs its
# backbone attached before it can encode text through SparseEncoder.
import json
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file
from sentence_transformers import SparseEncoder

# General loader for complete retrieval/SAE models or an older encoder/head
# directory. Returns a Sentence Transformers SparseEncoder in every branch.
# This accepts local directories as well as Hub IDs. The evaluation CLI directly
# loads complete ST checkpoints; raw SAE conversion remains this loader's job.
def from_pretrained(checkpoint, revision=None, **kwargs):
    # Inspect only config.json first to distinguish formats before downloading
    # all weights. Existing directories take the local path; other strings are
    # treated as Hub repository identifiers.
    source = Path(checkpoint)
    hub_kwargs = dict(token=kwargs.get("token"), cache_dir=kwargs.get("cache_folder"),
                      local_files_only=kwargs.get("local_files_only", False))
    config_path = source / "config.json" if source.is_dir() else Path(
        hf_hub_download(str(checkpoint), "config.json", revision=revision, **hub_kwargs))

    # Modern flat models declare their own AutoConfig/AutoModel implementation.
    # Their saved custom code is loaded here, not replaced by the local package.
    if json.loads(config_path.read_text()).get("model_type") == "slideon":
        return SparseEncoder(str(checkpoint), revision=revision, trust_remote_code=True, **kwargs)
    if not source.is_dir():
        source = Path(snapshot_download(str(checkpoint), revision=revision, **hub_kwargs))

    # Older layouts need a temporary flat export. Export copies the local model
    # implementation into that directory, then the normal wrapper loads it.
    # This conversion does not rewrite the original checkpoint directory.
    from .export import export_legacy_checkpoint
    with tempfile.TemporaryDirectory(prefix="slideon-export-") as directory:
        export_legacy_checkpoint(source, directory)
        return SparseEncoder(directory, trust_remote_code=True, **kwargs)


# Build the initializer used by retrieval fine-tuning. Raw SAE input and flat
# SAE input take different paths, but both return a full text encoder with the
# requested sequence limits. The training script overrides these helper defaults
# with its own JSON values (8,192 for each role in the English baseline).
def from_sae(checkpoint, revision=None, backbone_revision=None, query_length=128,
             document_length=512):
    source = Path(checkpoint)
    if not source.is_dir():
        config_path = Path(hf_hub_download(str(checkpoint), "config.json", revision=revision))
    else:
        config_path = source / "config.json"
    config = json.loads(config_path.read_text())

    # A complete SAE checkpoint already carries its backbone; refuse a
    # different backbone revision rather than silently pairing incompatible
    # token features and dictionary weights. Only sequence limits change here.
    if config.get("model_type") == "slideon":
        if config.get("checkpoint_type") != "sae":
            raise ValueError("from_sae requires an SAE initialization checkpoint")
        if backbone_revision is not None and backbone_revision != config.get("base_model_revision"):
            raise ValueError("The flat SAE bundles its backbone; a different revision requires a raw SAE")
        model = from_pretrained(checkpoint, revision=revision)
        model[0].settings.update(query_length=query_length, document_length=document_length)
        model[0].max_seq_length = max(query_length, document_length)
        return model

    # Raw SAE branch: only the recipe and dictionary weights are needed from
    # the SAE repository. The separately resolved base model supplies text
    # encoding, so the raw checkpoint is not itself an executable retriever.
    if not source.is_dir():
        source = Path(snapshot_download(checkpoint, revision=revision,
                      allow_patterns=["config.json", "sae.safetensors"]))
    if not config.get("normalized_inputs", True):
        raise ValueError("Only normalized-token SAEs are supported")
    # Prefer an explicit backbone pin, then the raw checkpoint's recorded pin.
    # The DenseOn fallback keeps old unpinned raw SAE recipes tied to the known
    # initialization instead of following a moving default branch.
    base_revision = backbone_revision or config.get("base_model_revision")
    if base_revision is None and config["base_model"] == "lightonai/DenseOn":
        base_revision = "cb9947ebccb33862d24e3c7ca2edb25e51acd887"
    base = Path(snapshot_download(config["base_model"], revision=base_revision))
    state = load_file(str(source / "sae.safetensors"))

    # Construct the retrieval-facing head without its reconstruction decoder or
    # training-only firing counters. export_legacy_checkpoint later reattaches
    # these raw SAE tensors for preservation; retrieval never calls the decoder.
    state = {key: value for key, value in state.items()
             if key not in ("W_dec", "steps_since_fired")}
    # Retrieval adds one learnable scale per latent after pooling. Starting
    # every gate at one preserves the pretrained encoder's initial magnitudes.
    state["gate_relu_phi"] = torch.ones(config["n_latents"])
    with tempfile.TemporaryDirectory(prefix="slideon-sae-") as directory:
        root = Path(directory)
        encoder = root / "encoder"
        shutil.copytree(base, encoder)
        # Rebuild the legacy intermediate format expected by the exporter.
        # Keep the transformer and only the configured token projection layers;
        # English truncate_after=0 excludes pooled sentence-embedding modules.
        modules = json.loads((encoder / "modules.json").read_text())
        modules = modules[:config.get("truncate_after", 0) + 1]
        if modules[0]["path"]:
            raise ValueError("Expected a backbone with its Transformer at the root")
        (encoder / "modules.json").write_text(json.dumps(modules))
        encoder_config = json.loads((encoder / "config_sentence_transformers.json").read_text())
        # English uses ordinary prompted token states, including special tokens.
        # Disable the marker/skip-list conventions of the optional other path.
        if config.get("backbone_only"):
            encoder_config.update(query_prefix="", document_prefix="", skiplist_words=[])
        (encoder / "config_sentence_transformers.json").write_text(json.dumps(encoder_config))

        # These are inference/model settings, not optimizer hyperparameters.
        # Final pooled vectors begin unpruned; fine-tuning later sets its chunk
        # size and applies Matryoshka budgets inside the retrieval loss.
        settings = dict(token_k=config["k"], query_max_active_dims=None, document_max_active_dims=None,
                        query_length=query_length, document_length=document_length,
                        head_chunk_size=256, prompts={
                            "query": config.get("query_prompt", ""),
                            "document": config.get("document_prompt", "")})
        (root / "slideon_config.json").write_text(json.dumps(settings))
        # Carry initialization provenance through conversion, alongside the
        # max-pooling/dot-product contract used by the retrieval model.
        (root / "config.json").write_text(json.dumps(dict(
            config, pooling="max_relu", score="dot", sae_source=str(checkpoint),
            sae_revision=revision, resolved_backbone_revision=base_revision or base.name)))
        save_file(state, str(root / "sparse_head.safetensors"))
        from .export import export_legacy_checkpoint
        # Passing sae_checkpoint preserves decoder weights in the flat artifact
        # even though they are excluded from retrieval computation. The exporter
        # also copies local adapter/core Python files into this new checkpoint.
        with tempfile.TemporaryDirectory(prefix="slideon-export-") as flat:
            export_legacy_checkpoint(root, flat, sae_checkpoint=source)
            return SparseEncoder(flat, trust_remote_code=True)
