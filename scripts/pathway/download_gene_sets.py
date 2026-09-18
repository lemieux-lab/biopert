"""
Downloads gene set collections (as GMT files) used for GSEA prerank analysis
comparing BioPert's predicted delta profiles to measured ones.

Gene sets are pulled from Enrichr (https://maayanlab.cloud/Enrichr/), which
mirrors the underlying MSigDB Hallmark, Reactome, and KEGG collections in
plain GMT text format (one gene set per line: name, description, gene...).
"""
import argparse
from pathlib import Path
from urllib.request import urlopen

DEFAULT_LIBRARIES = ["MSigDB_Hallmark_2020", "Reactome_2022", "KEGG_2021_Human"]
ENRICHR_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={library}"


def download_gmt(library: str, outdir: Path) -> Path:
    outfile = outdir / f"{library}.gmt"
    with urlopen(ENRICHR_URL.format(library=library)) as resp:
        outfile.write_bytes(resp.read())
    return outfile


def main():
    parser = argparse.ArgumentParser(description="Download Enrichr gene-set libraries (GMT) for GSEA.")
    parser.add_argument("--outdir", required=True, help="BIOPERT_OUTDIR: base directory for all pipeline data.")
    parser.add_argument("--libraries", nargs="+", default=DEFAULT_LIBRARIES,
                        help=f"Enrichr library names (default: {DEFAULT_LIBRARIES})")
    args = parser.parse_args()

    gene_sets_dir = Path(args.outdir) / "gene_sets"
    gene_sets_dir.mkdir(parents=True, exist_ok=True)

    for library in args.libraries:
        outfile = download_gmt(library, gene_sets_dir)
        n_sets = sum(1 for _ in outfile.open())
        print(f"{library}: {n_sets} gene sets -> {outfile}")


if __name__ == "__main__":
    main()
