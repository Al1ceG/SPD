#!/bin/bash
set -e

echo "Starting SFID (Low-Confidence Mean Replacement) Workflow..."

python src/read_pt.py --pt embedding/fairface_ViTB32_train.pt --out artifacts/fairface_ViTB32_train.jsonl

# Replace Dimensions
echo "Replacing Dimensions with Low-Confidence Means... age"
python src/replace_k.py \
  --input artifacts/fairface_ViTB32_train.jsonl \
  --lowconf low_conf/output_age.json \
  --output artifacts/clean_fairface_age_sfid.jsonl

echo "Replacing Dimensions with Low-Confidence Means... gender"
python src/replace_k.py \
  --input artifacts/fairface_ViTB32_train.jsonl \
  --lowconf low_conf/output_gender.json \
  --output artifacts/clean_fairface_gender_sfid.jsonl

echo "Replacing Dimensions with Low-Confidence Means... race"
python src/replace_k.py \
  --input artifacts/fairface_ViTB32_train.jsonl \
  --lowconf low_conf/output_race.json \
  --output artifacts/clean_fairface_race_sfid.jsonl

# Evaluate
echo "Evaluating..."
python src/classifer.py --input artifacts/clean_fairface_age_sfid.jsonl
python src/classifer.py --input artifacts/clean_fairface_gender_sfid.jsonl
python src/classifer.py --input artifacts/clean_fairface_race_sfid.jsonl

echo "SFID Workflow Completed Successfully!"
