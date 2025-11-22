#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py

Evaluate the discriminability of embeddings stored in "v" in a JSONL file by attribute:
  - Read JSONL, each line contains id, v, plus attribute columns (e.g. age, gender, race)
  - For each attribute:
      1. Use LabelEncoder to encode string labels into integers
      2. Randomly split training/testing sets (test set ratio adjustable)
      3. Train a LogisticRegression classifier on the original v
      4. Evaluate ACC and AUC on the test set (binary/multiclass)
  - Write all metrics to ./result/{input_basename}_result.json
"""

import os
import warnings
warnings.filterwarnings("ignore")
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def load_data(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if "v" not in rec:
                raise KeyError("Missing embedding field 'v' in record")
            records.append(rec)
    df = pd.DataFrame(records)
    X = np.vstack(df["v"].values)
    return df, X

def evaluate(df, X, test_size=0.2, random_state=42):
    # Attribute columns = all columns except id, v
    attrs = [c for c in df.columns if c not in ("id", "v")]
    results = {}
    for attr in attrs:
        y = df[attr].astype(str).values
        # print(pd.Series(y).value_counts())
        le = LabelEncoder().fit(y)
        y_enc = le.transform(y)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_enc,
            test_size=test_size,
            stratify=y_enc,
            random_state=random_state
        )

        clf = LogisticRegression(
            solver='lbfgs',
            multi_class='multinomial',
            max_iter=500,
            n_jobs=-1
        )
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)

        proba = clf.predict_proba(X_te)
        if proba.shape[1] == 2:
            auc = roc_auc_score(y_te, proba[:, 1])
        else:
            auc = roc_auc_score(y_te, proba, multi_class='ovr', average='macro')

        # print(f"Attribute: {attr}")
        # print(f"  Classes: {le.classes_.tolist()}")
        # print(f"  Test size: {len(y_te)}")
        # print(f"  ACC = {acc:.4f}")
        # print(f"  AUC = {auc:.4f}\n")

        results[attr] = {
            "classes": le.classes_.tolist(),
            "test_size": len(y_te),
            "ACC": acc,
            "AUC": auc
        }

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate discriminability on embeddings stored in 'v'"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input JSONL file, must contain field 'v' and attribute columns"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Test set ratio (default=0.2)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed (default=42)"
    )
    args = parser.parse_args()

    df, X = load_data(args.input)
    # print(f"Loaded {X.shape[0]} samples, embedding dim = {X.shape[1]}\n")

    results = evaluate(
        df, X,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Prepare output directory and filename
    base = os.path.basename(args.input)
    name, _ = os.path.splitext(base)
    outdir = "./result"
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{name}_result.json")

    # Write JSON
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"All results have been saved to {outfile}")

if __name__ == "__main__":
    main()
