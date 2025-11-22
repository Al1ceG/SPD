# SPD: Sensitive Subspace Projection for Feature Erasure

This repository contains the implementation of **SPD** and the baseline method **SFID** on attribute-replacement experiments on FairFace.

This experiment uses the SPD and SFID methods on the FairFace dataset to replace features across three dimensions, age, gender, and race, with low-confidence mean embeddings. Logistic regression is then used to evaluate the classification performance of these replaced embeddings. This verifies the effectiveness and completeness of the debiasing process and measures the level of feature entanglement (i.e., whether replacing one attribute significantly affects the accuracy of others) to ensure semantic integrity.

## Folder Structure

- `src/`: Source code (Python scripts).
- `embedding/`: Raw PyTorch `.pt` checkpoint files. Obtained from the SFID repo.
- `low_conf/`: Pre-computed low-confidence statistics (`output_*.json`). Obtained from the SFID repo.
- `artifacts/`: Intermediate generated files (`.jsonl`, `axes.json`).
- `result/`: Final evaluation results.

## Setup

Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

---

## Method 1: SPD (Proposed Method)

**To run the full SPD workflow:**
```bash
sh run_spd.sh
```
To experiment with different values of $r$ in our paper, adjust the value of `MAX_AXES` in line 4.

---

## Method 2: SFID (Baseline Method)

**To run the full SFID workflow:**
```bash
sh run_sfid.sh
```

