#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_low_conf_dims.py

Replace specified dimensions of the embedding in each record of the target JSONL file
with the given "low-confidence mean embedding" vector values.

Input:
  --lowconf : JSON file containing two fields:
               "img_important_indices": [int...]
               "img_mean_features_lowconfidence": [float...]  (length = D)
  --input   : Target JSONL file, each line must have "id" and "v" (original embedding list)
  --output  : Output JSONL path, embedding "v" will be replaced with the modified vector

Example:
  python apply_low_conf_dims.py \
    --lowconf low_conf_stats.json \
    --input  target_embeddings.jsonl \
    --output target_fixed.jsonl
"""

import json
import warnings
warnings.filterwarnings("ignore")
import argparse
from pathlib import Path


def load_lowconf(path):
    """
    Load low-confidence statistics JSON, read list of indices to replace and mean vector.
    Return: indices (List[int]), mean_vec (List[float])
    """
    data = json.load(open(path, 'r', encoding='utf-8'))
    indices = data.get("img_important_indices")
    mean_vec = data.get("img_mean_features_lowconfidence")
    if indices is None or mean_vec is None:
        raise KeyError("lowconf JSON must contain 'img_important_indices' and 'img_mean_features_lowconfidence' fields.")
    D = len(mean_vec)
    # Check index validity
    for idx in indices:
        if not (0 <= idx < D):
            raise ValueError(f"Index {idx} out of range for vector length {D}.")
    return indices, mean_vec


def main():
    parser = argparse.ArgumentParser(
        description="Replace specified dimensions with low-confidence mean embedding"
    )
    parser.add_argument(
        "-l", "--lowconf", required=True,
        help="Low-confidence statistics JSON file"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Original embeddings JSONL file, fields: id, v"
    )
    parser.add_argument(
        "-o", "--output", default="fixed.jsonl",
        help="Output JSONL file path (v will be replaced with low-confidence values)"
    )
    args = parser.parse_args()

    # Load low-confidence vector and indices
    indices, mean_vec = load_lowconf(args.lowconf)
    D = len(mean_vec)
    # print(f"Loaded low-conf vector of dim={D}, will replace {len(indices)} indices.")

    in_path = Path(args.input)
    out_path = Path(args.output)

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with in_path.open('r', encoding='utf-8') as fin, \
         out_path.open('w', encoding='utf-8') as fout:

        for line in fin:
            rec = json.loads(line)
            if 'v' not in rec:
                raise KeyError("Each record must contain 'v' field.")
            v = rec['v']
            if len(v) != D:
                raise ValueError(f"Embedding length {len(v)} != expected {D}.")

            # Replace specified dimensions
            for idx in indices:
                rec['v'][idx] = mean_vec[idx]

            # Write back to JSONL (only id, v and other original fields)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    # print(f"✅ Completed replacement for {count} records. Saved to {out_path}")


if __name__ == "__main__":
    main()
