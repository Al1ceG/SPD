#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_inlp_axes.py

Use Iterative Nullspace Projection (INLP) method to iteratively extract attribute discriminant axes (directions) from embedding data.

Input: JSONL, each JSON record must contain:
  - "id": Any unique identifier
  - "v": list/array of length D representing embedding
  - Other columns are attribute labels (e.g. gender, race, age_grp)

Output: JSON file, format:
{
  "gender": [
    [w0_1, w0_2, ..., w0_D],
    [w1_1, w1_2, ..., w1_D],
    ...
  ],
  "race": [
    ...
  ],
  ...
}
Each vector is of unit length.

Algorithm Flow (run independently for each attribute):
1. Load X, y
2. Let X_curr = X
3. for t in [0..max_iter):
     a. Train linear classifier clf on X_curr to predict y
     b. Calculate training accuracy acc
     c. If acc <= 1/K + tol: break
     d. Extract clf.coef_ matrix W (shape (num_classes or 1, D))
     e. Orthonormalize rows of W -> U (shape (k_t, D), row orthonormal unit vectors)
     f. Record each row of U as "t-th axis"
     g. Construct projection matrix P_t = I - U.T @ U
     h. Update X_curr = X_curr @ P_t
4. Write all axes to output JSON

Reference: Ravfogel et al., “Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection” (ACL 2020).
"""
import argparse
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def load_embeddings(jsonl_path):
    """
    Load embeddings from JSONL:
      - id: Any unique identifier
      - v: list of float, length D
      - Other columns = attribute labels
    Return: df (pandas.DataFrame), X (np.ndarray shape (N,D))
    """
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    if 'v' not in df.columns:
        raise ValueError("Each record must contain field 'v' representing embedding")
    X = np.vstack(df['v'].values)
    return df, X


def orthonormalize_rows(W):
    """
    Orthonormalize and unitize rows of matrix W (shape (k, D)):
    Perform QR decomposition on W.T, take first k columns of Q, then transpose to get U (k, D):
      Q, R = qr(W.T)  =>  Q: (D, k)
      U = Q[:, :k].T => (k, D)
    Ensures U @ U.T = I_k
    """
    # W: (k, D)
    # QR decomposition
    Q, _ = np.linalg.qr(W.T)
    U = Q[:, :W.shape[0]].T
    return U


def compute_projection_to_nullspace(U):
    """
    Given orthonormal unit vector matrix U (k, D), construct projection to the Nullspace of its row space:
      P = I_D - U.T @ U
    Where U.T @ U: (D, D) projects to row-space(U),
    I - that projects to null-space(U).
    """
    D = U.shape[1]
    P = np.eye(D) - U.T.dot(U)
    return P


def extract_inlp_axes(X, y, max_iter=50, tol=1e-6, C=1.0):
    """
    Iteratively perform INLP on (X, y) to extract multiple discriminant axes.
    Args:
      X: ndarray (N, D)
      y: ndarray (N,) already encoded labels 0..K-1
      max_iter: Maximum iterations
      tol: Stopping threshold, stop when train_accuracy <= (1/K + tol)
      C: LogisticRegression regularization strength parameter (default 1.0)
    Return:
      axes: list of np.ndarray, each element shape (D,) unit vector
    """
    axes = []
    X_curr = X.copy()
    N, D = X_curr.shape
    K = len(np.unique(y))
    # Baseline accuracy for random classifier
    stop_acc = 1.0 / K + tol

    for it in range(max_iter):
        # One-vs-Rest Logistic Regression
        clf = LogisticRegression(
            solver='lbfgs', multi_class='ovr',
            C=C, max_iter=1000
        )
        clf.fit(X_curr, y)
        acc = clf.score(X_curr, y)
        # print(f"  [iter {it:02d}] train acc = {acc:.4f}  vs. stop_acc = {stop_acc:.4f}")
        if acc <= stop_acc:
            # print("  Accuracy dropped to random level, stopping iteration.")
            break

        W = clf.coef_  # shape (K, D) or (1, D) for binary
        # Orthonormalize and unitize
        U = orthonormalize_rows(W)
        # Record each discriminant axis
        for w in U:
            axes.append(w.copy())
        # Construct projection matrix to null-space(U)
        P = compute_projection_to_nullspace(U)
        # Update X_curr
        X_curr = X_curr.dot(P)

    return axes


def main():
    p = argparse.ArgumentParser(description="Extract multiple discriminant axes from embedding using INLP")
    p.add_argument("--input", "-i", required=True,
                   help="JSONL file path, fields: id, v, <attribute columns>")
    p.add_argument("--attributes", "-a", nargs="+", required=True,
                   help="List of attribute column names to extract, e.g. gender race age_grp")
    p.add_argument("--output", "-o", default="axes_inlp.json",
                   help="Output JSON file path")
    p.add_argument("--max_iter", type=int, default=10,
                   help="INLP maximum iterations")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="Stopping threshold, difference from random accuracy")
    p.add_argument("--C", type=float, default=1.0,
                   help="LogisticRegression regularization strength parameter")
    args = p.parse_args()

    # 1. Load data
    df, X = load_embeddings(args.input)

    # 2. Perform INLP for each attribute
    all_axes = {}
    for attr in args.attributes:
        if attr not in df.columns:
            # print(f"[Warning] Attribute '{attr}' not in data, skipping.")
            continue
        # print(f"=== Attribute: {attr} ===")
        # Label encoding
        y = LabelEncoder().fit_transform(df[attr].astype(str).values)
        axes = extract_inlp_axes(X, y,
                                 max_iter=args.max_iter,
                                 tol=args.tol,
                                 C=args.C)
        # print(f"Extracted {len(axes)} axes.\n")
        all_axes[attr] = [w.tolist() for w in axes]

    # 3. Write JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_axes, f, ensure_ascii=False, indent=2)
    # print(f"Saved axes for all attributes to '{args.output}'.")


if __name__ == "__main__":
    main()
