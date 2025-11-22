#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_pt_to_jsonl.py

Automatically convert PyTorch .pt checkpoint to JSONL:
  - Automatically detect tensor keys, shapes, and label information in the checkpoint
  - Select embedding tensor (most likely 2D tensor, largest feature dimension)
  - Select all 1D tensors with length equal to N as label fields
  - Output JSONL, each line contains id, v, <label_keys>

Usage example:
  python convert_pt_to_jsonl.py --pt model.pt --out data.jsonl
"""
import argparse
import warnings
warnings.filterwarnings("ignore")
import json
from pathlib import Path
import torch

def inspect_checkpoint(ckpt):
    """
    Print keys, shapes, and dtypes of all tensors in the checkpoint,
    and return them as a dict.
    """
    tensors = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
    # print("Detected the following tensor fields in Checkpoint:")
    for k, v in tensors.items():
        shape = tuple(v.shape)
        dtype = v.dtype
        # print(f"  - {k}: shape={shape}, dtype={dtype}")
        if v.dim() == 1:
            uniq = torch.unique(v).tolist()
            # print(f"      Unique values ({len(uniq)}): {uniq[:10]}{'...' if len(uniq)>10 else ''}")
    # print("")
    return tensors


def select_embedding_key(tensors):
    """
    Select the most likely embedding tensor from tensors: 2D and second dimension is largest.
    Return (key, tensor).
    """
    candidates = [(k, v) for k, v in tensors.items() if v.dim() == 2]
    if not candidates:
        raise ValueError("No 2D tensor found as embedding.")
    emb_key, emb_tensor = max(candidates, key=lambda kv: kv[1].shape[1])
    # print(f"Selected '{emb_key}' as embedding, shape={tuple(emb_tensor.shape)}\n")
    return emb_key, emb_tensor


def select_label_keys(tensors, N):
    """
    Select all 1D tensors with length N as label fields.
    Return list of label_keys.
    """
    label_keys = [k for k, v in tensors.items() if v.dim() == 1 and v.shape[0] == N]
    # print(f"Selected {len(label_keys)} label fields: {label_keys}\n")
    return label_keys


def main():
    parser = argparse.ArgumentParser(description="Automatically convert .pt to JSONL")
    parser.add_argument('--pt',    required=True, help='Input .pt checkpoint file path')
    parser.add_argument('--out',   required=True, help='Output JSONL file path')
    args = parser.parse_args()

    pt_path = Path(args.pt)
    out_path = Path(args.out)

    # Load checkpoint
    ckpt = torch.load(pt_path, map_location='cpu')
    # Print and collect all tensors
    tensors = inspect_checkpoint(ckpt)

    # Select embedding
    emb_key, emb_tensor = select_embedding_key(tensors)
    N, D = emb_tensor.shape

    # Select labels
    label_keys = select_label_keys(tensors, N)

    # Write JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open('w', encoding='utf-8') as fout:
        for i in range(N):
            rec = {'id': i + 1, 'v': emb_tensor[i].tolist()}
            
            # Handle sensitive_attributes if present
            if "sensitive_attributes" in tensors:
                sa = tensors["sensitive_attributes"][i]
                # Mapping based on unique value counts: 0->age(9), 1->gender(2), 2->race(7)
                rec["age_grp"] = sa[0].item()
                rec["gender"] = sa[1].item()
                rec["race"] = sa[2].item()

            for lk in label_keys:
                rec[lk] = tensors[lk][i].item()
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            count += 1

    # print(f"✅ Conversion complete, wrote {count} records to {out_path}")

if __name__ == '__main__':
    main()
