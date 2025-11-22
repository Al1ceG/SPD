#!/bin/bash
set -e

export MAX_AXES=5

echo "Starting SPD (INLP Feature Replacement) Workflow..."

python src/read_pt.py --pt embedding/fairface_ViTB32_train.pt --out artifacts/fairface_ViTB32_train.jsonl

# Extract INLP Axes
echo "Extracting $MAX_AXES INLP Axes..."
python src/extract_inlp_axes.py --input artifacts/fairface_ViTB32_train.jsonl --attributes gender race age_grp --output artifacts/axes.json --max_iter $MAX_AXES

# Replace Features
echo "Replacing Features... age_grp"
python src/replace_feature.py \
  --input artifacts/fairface_ViTB32_train.jsonl \
  --axes artifacts/axes.json \
  --attribute age_grp \
  --lowconf low_conf/output_age.json \
  --output artifacts/clean_fairface_age_spd.jsonl

echo "Replacing Features... gender"
python src/replace_feature.py \
  --input artifacts/fairface_ViTB32_train.jsonl \
  --axes artifacts/axes.json \
  --attribute gender \
  --lowconf low_conf/output_gender.json \
  --output artifacts/clean_fairface_gender_spd.jsonl

echo "Replacing Features... race"
python src/replace_feature.py \
  --input artifacts/fairface_ViTB32_train.jsonl \
  --axes artifacts/axes.json \
  --attribute race \
  --lowconf low_conf/output_race.json \
  --output artifacts/clean_fairface_race_spd.jsonl

echo "Evaluating..."
python src/classifer.py --input artifacts/clean_fairface_age_spd.jsonl
python src/classifer.py --input artifacts/clean_fairface_gender_spd.jsonl
python src/classifer.py --input artifacts/clean_fairface_race_spd.jsonl

echo "SPD Workflow Completed Successfully!"
