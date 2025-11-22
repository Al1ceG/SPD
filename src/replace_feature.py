#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replace_low_conf_with_axes.py

Use multiple axes U extracted by INLP and a given low-confidence mean vector bar_x,
to replace the projection on each axis, generating new embedding v.
"""

import json, argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path

def load_embeddings(path):
    """Load JSONL with fields: id, v, plus other attrs."""
    recs = [json.loads(line) for line in open(path, 'r', encoding='utf-8')]
    df = pd.DataFrame(recs)
    X = np.vstack(df['v'].values)  # (N, D)
    return df, X

def load_axes(path, attribute):
    """
    Load U axes from JSON:
      { "gender": [[...],...], "race": [...], ... }
    Returns U as np.ndarray shape (k, D), orthonormal rows.
    """
    raw = json.load(open(path, 'r', encoding='utf-8'))
    W = np.array(raw[attribute])     # maybe (k,D) or (D,) 
    if W.ndim == 1:
        W = W[np.newaxis, :]
    # Orthogonalize rows: QR on W^T
    Q, _ = np.linalg.qr(W.T)         # Q: (D, k)
    U = Q.T[:W.shape[0], :]          # take first k rows → (k, D)
    return U

def load_lowconf(path):
    """
    Load low-confidence mean vector JSON:
      { "img_important_indices": [...], 
        "img_mean_features_lowconfidence": [ ... length D ... ] }
    Returns mean_vec (length D).
    """
    data = json.load(open(path, 'r', encoding='utf-8'))
    mean_vec = np.array(data["img_mean_features_lowconfidence"], dtype=float)
    return mean_vec

def replace_with_axes(X, U, mean_vec):
    """
    X: (N, D)
    U: (k, D) orthonormal axes
    mean_vec: (D,) the low-conf mean embedding
    Returns X_fixed: (N, D)
    """
    # 1) compute low-conf projections s_low = U @ mean_vec
    s_low = U.dot(mean_vec)           # (k,)

    # 2) compute original projections S = X @ U^T
    S = X.dot(U.T)                    # (N, k)

    # 3) delta = s_low - S_j for each sample j
    Delta = s_low[np.newaxis, :] - S  # (N, k)

    # 4) reconstruct fixed embeddings
    X_fixed = X + Delta.dot(U)        # (N, D)
    return X_fixed

def main():
    p = argparse.ArgumentParser(
        description="Replace projections using INLP axes & low-confidence mean vector"
    )
    p.add_argument("-i","--input", required=True,
                   help="Original JSONL embeddings file")
    p.add_argument("-a","--axes", required=True,
                   help="INLP output axes JSON")
    p.add_argument("-t","--attribute", required=True,
                   help="Attribute name (e.g. gender, race, age_grp)")
    p.add_argument("-l","--lowconf", required=True,
                   help="Low-confidence mean vector JSON file")
    p.add_argument("-o","--output", default="fixed_axes.jsonl",
                   help="Output JSONL file")
    args = p.parse_args()

    # 1. load embeddings
    df, X = load_embeddings(args.input)
    N, D = X.shape
    # print(f"Loaded {N} samples with embedding dim = {D}")

    # 2. load axes U (k, D)
    U = load_axes(args.axes, args.attribute)
    k = U.shape[0]
    # print(f"Loaded {k} axes for attribute '{args.attribute}'")

    # 3. load low-conf mean embedding
    mean_vec = load_lowconf(args.lowconf)
    if mean_vec.shape[0] != D:
        raise ValueError(f"mean_vec dim {mean_vec.shape[0]} != embedding dim {D}")

    # 4. replace
    X_fixed = replace_with_axes(X, U, mean_vec)
    # print("Computed fixed embeddings via axes replacement.")

    # 5. write JSONL
    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as fw:
        for idx, row in df.iterrows():
            rec = {
                "id":     row.get("id", None),
                "v":      X_fixed[idx].tolist(),                   
            }
            # copy any other metadata (attributes)
            for c in df.columns:
                if c not in ("id","v"):
                    rec[c] = row[c]
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # print(f"✅ Saved fixed embeddings to {out_path}")

if __name__ == "__main__":
    main()
