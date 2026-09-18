#!/usr/bin/env python3
"""Fetch and freeze the public pathway libraries used by the Tahoe case study."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import gseapy


LIBRARIES = {
    "hallmark": "MSigDB_Hallmark_2020",
    "reactome": "Reactome_2022",
}


def write_gmt(path: Path, gene_sets: dict[str, list[str]]) -> None:
    with path.open("w") as handle:
        for term in sorted(gene_sets):
            genes = sorted(set(gene_sets[term]))
            handle.write("\t".join([term, "Enrichr", *genes]) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "gseapy_version": gseapy.__version__,
        "organism": "Human",
        "source": "Enrichr public gene-set API",
        "libraries": {},
    }
    for label, library_name in LIBRARIES.items():
        gene_sets = gseapy.get_library(name=library_name, organism="Human")
        path = args.output_dir / f"{library_name}.gmt"
        write_gmt(path, gene_sets)
        manifest["libraries"][label] = {
            "name": library_name,
            "path": str(path.resolve()),
            "n_gene_sets": len(gene_sets),
        }

    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
