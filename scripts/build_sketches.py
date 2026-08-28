#!/usr/bin/env python3
"""
Build MinHash sketches from raw data.

This is the step that turns a raw corpus (memory dumps, PDF files, images, DNA
sequences, news articles) into the CSV feature matrix consumed by the two
experiment scripts. The shingle parameters of every dataset are stored in
``configs/datasets.yaml`` and reproduce Tables 2 and 3 of the paper.

Expected input layout: one directory per class, e.g.

    raw/pdf/benign/*.pdf
    raw/pdf/malicious/*.pdf

Usage
-----
    # Experiment 1 sketch (400 features, primary overlap)
    python scripts/build_sketches.py --dataset pdf --input raw/pdf \
        --out data/minhash/pdf_400.csv

    # Experiment 2 sketches (200 features each, two overlaps)
    python scripts/build_sketches.py --dataset pdf --input raw/pdf \
        --variant secondary --n-functions 200 --out data/minhash/pdf_b_200.csv

Note on the memory-dump dataset: the malware samples themselves are not
redistributed (see data/README.md); this script is provided so that the sketches
can be regenerated locally by anyone holding the samples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minhash_fs.datasets import save_minhash_csv
from minhash_fs.rabin import make_fingerprint_family
from minhash_fs.shingles import extract_shingles
from minhash_fs.sketch import minhash_sketch, partition_minhash_sketch

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "datasets.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="dataset key in configs/datasets.yaml "
                        "(memory_dumps, pdf, chest_xray, dna, fake_news)")
    p.add_argument("--input", required=True,
                   help="root directory holding one sub-directory per class")
    p.add_argument("--variant", default="primary", choices=["primary", "secondary"],
                   help="overlap setting: primary = Experiment 1, "
                        "secondary = the second sketch of Experiment 2")
    p.add_argument("--n-functions", type=int, default=400,
                   help="sketch size (400 for Experiment 1, 200 per sketch for Exp. 2)")
    p.add_argument("--degree", type=int, default=61,
                   help="degree of the Rabin irreducible polynomials")
    p.add_argument("--seed", type=int, default=42,
                   help="seed of the fingerprint family; keep it fixed across datasets")
    p.add_argument("--out", required=True, help="destination CSV")
    return p.parse_args()


def load_config(dataset: str) -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if dataset not in config["datasets"]:
        raise KeyError(f"'{dataset}' not in {CONFIG_PATH}. "
                       f"Available: {sorted(config['datasets'])}")
    return config["datasets"][dataset]


def read_item(path: Path, modality: str):
    if modality == "bytes":
        return path.read_bytes()
    if modality in ("words", "characters"):
        text = path.read_text(errors="ignore")
        if modality == "characters":     # FASTA: drop headers and whitespace
            text = "".join(line.strip() for line in text.splitlines()
                           if not line.startswith(">"))
        return text
    if modality == "image_2d":
        from PIL import Image
        return np.asarray(Image.open(path).convert("L"))   # grayscale, per Table 1
    raise ValueError(f"Unknown modality '{modality}'")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.dataset)
    modality = cfg["modality"]
    length = cfg["shingle_length"]
    overlap = cfg["overlap"][args.variant]

    print(f"dataset={args.dataset} modality={modality} "
          f"shingle={length} overlap={overlap} ({args.variant}) "
          f"k={args.n_functions}")

    family = make_fingerprint_family(args.n_functions, degree=args.degree,
                                     random_state=args.seed)

    root = Path(args.input)
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class sub-directories found under {root}")

    sketches, labels = [], []
    for class_dir in class_dirs:
        files = sorted(f for f in class_dir.rglob("*") if f.is_file())
        print(f"  class '{class_dir.name}': {len(files)} files")
        for index, file_path in enumerate(files):
            item = read_item(file_path, modality)
            if modality == "image_2d" and cfg.get("partition_minhash", False):
                sketch = partition_minhash_sketch(item, family, length, overlap)
            else:
                sketch = minhash_sketch(extract_shingles(item, modality, length, overlap),
                                        family)
            sketches.append(sketch)
            labels.append(class_dir.name)
            if (index + 1) % 200 == 0:
                print(f"    {index + 1}/{len(files)} sketched", flush=True)

    if not sketches:
        raise SystemExit(f"No readable files found under {root}. Expected one "
                         f"sub-directory per class, each holding the raw items.")

    X = np.vstack(sketches)
    y = np.array(labels)
    save_minhash_csv(args.out, X, y)
    print(f"Wrote {X.shape[0]} x {X.shape[1]} sketch matrix to {args.out}")


if __name__ == "__main__":
    main()
