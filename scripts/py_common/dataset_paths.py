import tomllib
from pathlib import Path


BIOPERT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = BIOPERT_ROOT / "configs" / "default_paths.toml"


def dataset_paths(outdir: str, dataset: str) -> dict[str, Path]:
    """Resolve the default paths for `dataset` ("lincs" or "tahoe") under `outdir`
    (BIOPERT_OUTDIR), as configured in configs/default_paths.toml."""
    with open(CONFIG_FILE, "rb") as f:
        config = tomllib.load(f)

    if dataset not in config:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of: {', '.join(config)}")

    outdir = Path(outdir)
    return {key: outdir / value for key, value in config[dataset].items()}
