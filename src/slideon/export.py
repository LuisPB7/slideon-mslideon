"""Convert legacy encoder/head directories to a self-contained flat checkpoint."""

# This is a format conversion utility, not another training stage. The legacy
# format separates encoder/, sparse_head.safetensors and retrieval settings;
# the flat format packages them as one Transformers/Sentence Transformers model.
import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from .modeling_slideon import SlideOnConfig


# Read an existing legacy directory and write a new self-contained directory.
# sae_checkpoint optionally supplies the raw reconstruction decoder/counters to
# preserve alongside the retrieval head. Return the output Path for the loader.
def export_legacy_checkpoint(source, output, sae_checkpoint=None):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Export into a separate directory before replacing the source")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("The export directory must be empty")

    # Historical exports used different metadata filenames. Keep model recipe
    # and provenance separate from runtime inference settings while reading.
    metadata_path = source / "retrieval_config.json"
    metadata = json.loads((metadata_path if metadata_path.exists() else source / "config.json").read_text())
    settings = json.loads((source / "slideon_config.json").read_text())
    # Translate historical checkpoint metadata; the public API uses max_active_dims.
    for task in ("query", "document"):
        legacy_cap = settings.pop(task + "_top_k", None)
        settings.setdefault(task + "_max_active_dims", legacy_cap or None)
    settings.update(pooling=metadata["pooling"], score=metadata["score"])

    # Rename tensor keys to match SlideOnModel's three submodule namespaces:
    # backbone.*, optional projections.*, and head.*. This changes packaging,
    # not tensor values or the order of neural operations.
    encoder = source / "encoder"
    tokenizer = AutoTokenizer.from_pretrained(encoder, local_files_only=True)
    encoder_config = json.loads((encoder / "config_sentence_transformers.json").read_text())
    modules = json.loads((encoder / "modules.json").read_text())
    state = {"backbone." + k: v for k, v in load_file(str(encoder / "model.safetensors")).items()}
    projection_configs = []
    # English backbone-only exports have no entries after the transformer.
    # The shared converter retains support for optional token projection layers
    # in other compatible checkpoints without enabling them for English.
    for index, entry in enumerate(modules[1:]):
        folder = encoder / entry["path"]
        projection_configs.append(json.loads((folder / "config.json").read_text()))
        state.update({f"projections.{index}." + k: v
                      for k, v in load_file(str(folder / "model.safetensors")).items()})
    head = load_file(str(source / "sparse_head.safetensors"))

    # A reconstruction decoder is unnecessary for retrieval, but retaining it
    # makes a flat SAE initialization complete. Check overlapping encoder tensors
    # before merging so two inconsistent sources cannot be combined silently.
    sae_path = Path(sae_checkpoint or source) / "sae.safetensors"
    if sae_path.exists():
        sae = load_file(str(sae_path))
        for key in head.keys() & sae.keys():
            if not torch.equal(head[key], sae[key]):
                raise ValueError(f"Raw SAE and inference head disagree: {key}")
        head.update(sae)

    # Derive architectural sizes from saved tensors, rather than assuming the
    # English defaults. H=d_model is the token width and V=n_latents the output
    # vocabulary size; d_lift=0 selects the ordinary linear SAE encoder.
    head_config = dict(d_model=head["b_dec"].numel(), n_latents=head["b_enc"].numel(),
                       d_lift=head["lift.weight"].shape[0] if "lift.weight" in head else 0,
                       has_decoder="W_dec" in head)
    if head_config["d_lift"]:
        head_config["hidden_dim"] = head["mlp.0.weight"].shape[0]
    state.update({"head." + k: v for k, v in head.items()})

    # Preserve where the weights came from. A backbone pin and an SAE pin refer
    # to different repositories/stages and should not be treated as one revision.
    provenance = {k: metadata[k] for k in ("base_model", "sae_path", "sae_source", "sae_revision") if k in metadata}
    provenance["base_model_revision"] = metadata.get("base_model_revision") or metadata.get("model_revision") or metadata.get("resolved_backbone_revision")
    # This checkpoint config describes the saved architecture and inference
    # behavior, unlike scripts/training-configs/*/config.json which describes a
    # training run. auto_map tells Transformers which bundled classes to load.
    # checkpoint_type is inferred here from decoder presence; it is metadata,
    # not a switch that changes the retrieval forward pass.
    config = SlideOnConfig(
        backbone_config=json.loads((encoder / "config.json").read_text()),
        projection_configs=projection_configs, head_config=head_config,
        inference=settings,
        tokenizer_settings={k: encoder_config[k] for k in
                            ("query_prefix", "document_prefix", "skiplist_words") if k in encoder_config},
        skip_token_ids=tokenizer.convert_tokens_to_ids(encoder_config.get("skiplist_words", [])),
        checkpoint_type="sae" if head_config["has_decoder"] else "retrieval",
        architectures=["SlideOnModel"],
        auto_map={"AutoConfig": "modeling_slideon.SlideOnConfig", "AutoModel": "modeling_slideon.SlideOnModel"},
        **provenance)

    # Write the flat model config, weights and tokenizer, then wrapper metadata.
    # No optimizer state or token cache is required to use the flat model.
    config.save_pretrained(output)
    save_file(state, str(output / "model.safetensors"), metadata={"format": "pt"})
    tokenizer.save_pretrained(output)

    # Bundle the local Python implementation with the exported weights.
    # models.py becomes slideon.py, exposing the InputModule adapter SlideOn;
    # modeling_slideon.py contains the distinct Transformers SlideOnModel core.
    # Later loading this artifact with trust_remote_code uses these saved files,
    # so subsequent edits in a separate local checkout do not update it.
    code = Path(__file__).parent
    shutil.copy2(code / "modeling_slideon.py", output / "modeling_slideon.py")
    shutil.copy2(code / "models.py", output / "slideon.py")
    # Sentence Transformers sees one module at the checkpoint root. That module
    # itself owns the entire backbone/head model through Transformers AutoModel.
    (output / "modules.json").write_text(json.dumps([
        dict(idx=0, name="0", path="", type="slideon.SlideOn")], indent=2) + "\n")
    # Reuse outer-wrapper settings when supplied; otherwise create the sparse
    # encoder's prompt and dot-product defaults from the legacy inference data.
    st_config = source / "config_sentence_transformers.json"
    st = json.loads(st_config.read_text()) if st_config.exists() else dict(
        model_type="SparseEncoder", prompts=settings["prompts"],
        default_prompt_name="document", similarity_fn_name="dot")
    (output / "config_sentence_transformers.json").write_text(json.dumps(st, indent=2) + "\n")
    return output


# Optional command-line conversion entry point (python -m slideon.export).
# Unlike from_sae(), this expects an already assembled legacy directory.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    export_legacy_checkpoint(args.source, args.output)
