import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForPreTraining,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "py_common"))
from dataset_paths import dataset_paths  # noqa: E402
from parquet_io import save_embeds_parquet  # noqa: E402


MODEL_TO_CLASS = {
    "DeepChem/ChemBERTa-77M-MLM": AutoModelForMaskedLM,
    "ibm/MoLFormer-XL-both-10pct": AutoModel,
    "zjunlp/MolGen-large": AutoModelForSeq2SeqLM,
    "unikei/bert-base-smiles": AutoModelForPreTraining,
}

# Models whose tokenizer vocabulary is SELFIES rather than SMILES. Feeding raw SMILES
# to these silently produces an empty token sequence (`<s></s>`) for every molecule, so
# every compound gets an identical embedding. MolGen-large's vocabulary is 180/185
# SELFIES bracket tokens and contains no SMILES characters.
SELFIES_MODELS = {"zjunlp/MolGen-large"}


def get_unimol_embeddings(args, smiles_csv):
    # Imported lazily: unimol_tools is only needed for the UniMol models, and it is not
    # installed in every env. A top-level import makes the HuggingFace path fail too.
    from unimol_tools import UniMolRepr

    model_name, model_size = args.pretrained_name_or_path.split("/")
    df = pd.read_csv(smiles_csv)

    clf = UniMolRepr(data_type="molecule", remove_hs=False, model_name=model_name, model_size=model_size)
    smiles = list(df["smiles"].values)
    unimol_repr = clf.get_repr(smiles, return_atomic_reprs=False)

    embedded_dataset = {
        "drug": df["drug"].values,
        "smiles": smiles,
        "embedding": unimol_repr,
    }

    return embedded_dataset


def to_selfies(smiles: str) -> str | None:
    """SMILES -> SELFIES, or None if the molecule cannot be encoded."""
    import selfies as sf
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def get_hf_embeddings(args, device, smiles_csv):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path, trust_remote_code=True)
    use_selfies = args.pretrained_name_or_path in SELFIES_MODELS

    # Model
    if args.pretrained_name_or_path in MODEL_TO_CLASS:
        model = MODEL_TO_CLASS[args.pretrained_name_or_path].from_pretrained(args.pretrained_name_or_path, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(args.pretrained_name_or_path, trust_remote_code=True)

    model.to(device)

    # Data
    dataset = load_dataset("csv", data_files=str(smiles_csv), split="all")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda x: x)

    # Generate embeddings
    embedded_dataset = {"drug": [], "smiles": [], "embedding": []}
    n_skipped = 0
    for batch in tqdm(dataloader):
        # SMILES are stripped before use: stray leading/trailing whitespace makes both
        # RDKit and the SELFIES encoder reject otherwise valid molecules.
        smiles = [str(item["smiles"]).strip() for item in batch]
        drugs = [item["drug"] for item in batch]

        if use_selfies:
            converted = [to_selfies(s) for s in smiles]
            keep = [i for i, c in enumerate(converted) if c is not None]
            n_skipped += len(converted) - len(keep)
            if not keep:
                continue
            drugs = [drugs[i] for i in keep]
            smiles = [smiles[i] for i in keep]
            model_input = [converted[i] for i in keep]
        else:
            model_input = smiles

        inputs = tokenizer(model_input, padding=True, truncation=True,
                           max_length=args.max_length, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        if "pooler_output" in outputs.keys():
            embeddings = outputs.pooler_output
        else:
            if "encoder_last_hidden_state" in outputs.keys():
                hidden = outputs.encoder_last_hidden_state
            elif "last_hidden_state" in outputs.keys():
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs.hidden_states[-1]
            # Mean over real tokens only. An unmasked mean averages in the padding
            # positions, so a molecule's embedding would depend on the longest
            # sequence that happened to share its batch.
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        embeddings = embeddings.detach().cpu()

        embedded_dataset["drug"].extend(drugs)
        embedded_dataset["smiles"].extend(smiles)
        embedded_dataset["embedding"].extend(embeddings.numpy())

    if use_selfies and n_skipped:
        print(f"WARNING: skipped {n_skipped} molecules that could not be encoded as SELFIES.")

    return embedded_dataset


if __name__ == "__main__":
    parser = ArgumentParser(description="Compute molecule embeddings.")
    parser.add_argument(
        "--pretrained_name_or_path",
        type=str,
        required=True,
        help="Model name.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="BIOPERT_OUTDIR: base directory for all pipeline data.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["lincs", "tahoe"],
        help="Dataset whose SMILES to embed.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max tokenized sequence length.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device. 'auto' uses CUDA when available, else CPU.",
    )
    args = parser.parse_args()
    is_unimol = "unimol" in args.pretrained_name_or_path

    paths = dataset_paths(args.outdir, args.dataset)
    smiles_csv = paths["smiles_csv"]

    if is_unimol:
        model_outdir = paths["molec_embeds_dir"] / args.pretrained_name_or_path.replace("/", "_")
    else:
        model_outdir = paths["molec_embeds_dir"] / args.pretrained_name_or_path.split("/")[-1]
    model_outdir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if is_unimol:
        embedded_dataset = get_unimol_embeddings(args, smiles_csv)
    else:
        embedded_dataset = get_hf_embeddings(args, device, smiles_csv)

    df = pd.DataFrame(embedded_dataset)
    save_embeds_parquet(df, model_outdir / "embeds.parquet")

    embedded_dataset = Dataset.from_dict(embedded_dataset)
    embedded_dataset.save_to_disk(model_outdir)