import argparse
import os
import sys
import json
import random
import pickle

import tempfile

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

from transformers import GPT2Tokenizer

# import things from the unified paper
from clip_debiasing.models.clipcap import model_clipcap
from clip_debiasing.models.clipcap.clipcap_utils import decide_gender, generate
import clip
from clip_debiasing.models.model_vl_debiasing import DebiasedCLIP
from unified_debiasing.evaluation import evaluate_image_captioning


def train_or_load_decoder_sfid(
    train_path,
    val_path,
    rf_checkpoint_path,
    threshold,
    prune_num,
):
    """Train/load decoder RF and return SFID decoder intervention tensors."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Decoder SFID train embedding file not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Decoder SFID val embedding file not found: {val_path}")

    train_embedding = torch.load(train_path, map_location="cpu")
    val_embedding = torch.load(val_path, map_location="cpu")

    required_keys = {"hidden_states", "sensitive_attributes"}
    if not required_keys.issubset(train_embedding.keys()):
        raise KeyError(f"Train embedding must contain keys: {required_keys}")
    if not required_keys.issubset(val_embedding.keys()):
        raise KeyError(f"Val embedding must contain keys: {required_keys}")

    x_train = train_embedding["hidden_states"]
    y_train = train_embedding["sensitive_attributes"]
    x_val = val_embedding["hidden_states"]

    if isinstance(x_train, torch.Tensor):
        x_train = x_train.cpu().numpy()
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.cpu().numpy()
    if isinstance(x_val, torch.Tensor):
        x_val = x_val.cpu().numpy()

    os.makedirs(os.path.dirname(rf_checkpoint_path), exist_ok=True)
    if os.path.exists(rf_checkpoint_path):
        decoder_clf = load(rf_checkpoint_path)
        print(f"Loaded decoder RF model from {rf_checkpoint_path}")
    else:
        decoder_clf = RandomForestClassifier(n_estimators=100, random_state=0)
        decoder_clf.fit(x_train, y_train)
        dump(decoder_clf, rf_checkpoint_path)
        print(f"Trained and saved decoder RF model to {rf_checkpoint_path}")

    probabilities = decoder_clf.predict_proba(x_val)
    max_probabilities = probabilities.max(axis=1)
    low_confidence_samples = x_val[max_probabilities < threshold]

    if low_confidence_samples.shape[0] == 0:
        print(
            "Warning: no low-confidence decoder samples found for threshold "
            f"{threshold}. Falling back to mean over all validation hidden states."
        )
        low_confidence_samples = x_val

    decoder_mean_features_lowconfidence = torch.tensor(
        low_confidence_samples.mean(axis=0),
        dtype=torch.float32,
    )
    decoder_importances = decoder_clf.feature_importances_
    text_important_indices = np.argsort(decoder_importances)[-prune_num:]
    text_important_indices = torch.tensor(text_important_indices, dtype=torch.long)

    return text_important_indices, decoder_mean_features_lowconfidence


def _orthonormalize_rows(W: np.ndarray) -> np.ndarray:
    """
    Orthonormalize rows of W using QR on W^T.
    If W has shape (k, D), returns U with shape (k, D) and U @ U^T == I_k.
    """
    Q, _ = np.linalg.qr(W.T)  # Q: (D, k)
    U = Q[:, : W.shape[0]].T  # (k, D)
    return U


def train_or_load_decoder_spd(
    train_path,
    val_path,
    rf_checkpoint_path,
    threshold,
    max_iter,
    tol,
    C,
    inlp_axes_checkpoint_path=None,
    max_axes=None,
):
    """
    Train/load decoder RF to compute low-confidence mean vector (SFID-style),
    then extract INLP axes U for SPD on decoder hidden states.
    Returns:
      - text_inlp_axes: torch.Tensor of shape (k, D)
      - decoder_mean_features_lowconfidence: torch.Tensor of shape (D,)
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Decoder SPD train embedding file not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Decoder SPD val embedding file not found: {val_path}")

    train_embedding = torch.load(train_path, map_location="cpu")
    val_embedding = torch.load(val_path, map_location="cpu")

    required_keys = {"hidden_states", "sensitive_attributes"}
    if not required_keys.issubset(train_embedding.keys()):
        raise KeyError(f"Train embedding must contain keys: {required_keys}")
    if not required_keys.issubset(val_embedding.keys()):
        raise KeyError(f"Val embedding must contain keys: {required_keys}")

    x_train = train_embedding["hidden_states"]
    y_train = train_embedding["sensitive_attributes"]
    x_val = val_embedding["hidden_states"]

    if isinstance(x_train, torch.Tensor):
        x_train = x_train.cpu().numpy()
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.cpu().numpy()
    if isinstance(x_val, torch.Tensor):
        x_val = x_val.cpu().numpy()

    # Ensure shapes are what sklearn expects: X is (N, D), y is (N,)
    if y_train.ndim > 1:
        y_train = np.squeeze(y_train)
    if x_train.ndim != 2:
        raise ValueError(f"Expected decoder hidden_states to be 2D (N, D), got {x_train.shape}")
    if x_val.ndim != 2:
        raise ValueError(f"Expected decoder hidden_states to be 2D (N, D), got {x_val.shape}")

    y_train = y_train.astype(np.int64)

    # 1) Compute low-confidence mean vector (same RF approach as SFID)
    os.makedirs(os.path.dirname(rf_checkpoint_path), exist_ok=True)
    if os.path.exists(rf_checkpoint_path):
        decoder_clf = load(rf_checkpoint_path)
        print(f"Loaded decoder RF model from {rf_checkpoint_path}")
    else:
        decoder_clf = RandomForestClassifier(n_estimators=100, random_state=0)
        decoder_clf.fit(x_train, y_train)
        dump(decoder_clf, rf_checkpoint_path)
        print(f"Trained and saved decoder RF model to {rf_checkpoint_path}")

    probabilities = decoder_clf.predict_proba(x_val)
    max_probabilities = probabilities.max(axis=1)
    low_confidence_samples = x_val[max_probabilities < threshold]
    if low_confidence_samples.shape[0] == 0:
        print(
            "Warning: no low-confidence decoder samples found for threshold "
            f"{threshold}. Falling back to mean over all validation hidden states."
        )
        low_confidence_samples = x_val

    decoder_mean_features_lowconfidence = torch.tensor(
        low_confidence_samples.mean(axis=0),
        dtype=torch.float32,
    )

    # 2) Extract INLP axes U on x_train/y_train
    if inlp_axes_checkpoint_path is not None and os.path.exists(inlp_axes_checkpoint_path):
        inlp_axes = load(inlp_axes_checkpoint_path)
        if isinstance(inlp_axes, np.ndarray):
            U_axes = torch.tensor(inlp_axes, dtype=torch.float32)
        else:
            U_axes = torch.tensor(inlp_axes, dtype=torch.float32)
        print(f"Loaded INLP axes from {inlp_axes_checkpoint_path}")
        return U_axes, decoder_mean_features_lowconfidence

    le = LabelEncoder()
    y = le.fit_transform(y_train.astype(str))

    X_curr = x_train.copy()
    _, D = X_curr.shape
    K = len(np.unique(y))
    stop_acc = 1.0 / K + tol

    axes = []
    for it in range(max_iter):
        clf = LogisticRegression(
            solver="lbfgs",
            multi_class="ovr",
            C=C,
            max_iter=1000,
        )
        clf.fit(X_curr, y)
        acc = clf.score(X_curr, y)
        if acc <= stop_acc:
            break

        W = clf.coef_  # (K, D) for multiclass, (1, D) for binary
        U = _orthonormalize_rows(W)  # (k_t, D)

        for row in U:
            axes.append(row.copy())

        # Project X_curr into the nullspace of the current extracted axes
        P = np.eye(D) - (U.T @ U)  # (D, D)
        X_curr = X_curr @ P

        if max_axes is not None and len(axes) >= max_axes:
            axes = axes[:max_axes]
            break

    if len(axes) == 0:
        raise RuntimeError("INLP produced 0 axes; consider lowering max_iter or tol.")

    U_axes = np.stack(axes, axis=0).astype(np.float32)  # (k, D)
    if inlp_axes_checkpoint_path is not None:
        os.makedirs(os.path.dirname(inlp_axes_checkpoint_path), exist_ok=True)
        dump(U_axes, inlp_axes_checkpoint_path)
        print(f"Saved INLP axes to {inlp_axes_checkpoint_path}")

    return torch.tensor(U_axes, dtype=torch.float32), decoder_mean_features_lowconfidence


def main():
    
    scratch_dir = os.environ.get("SCRATCH", f"/disk/scratch/s2142414")
    os.makedirs(scratch_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Measure baseline bias of ClipCap (no debiasing).")
    parser.add_argument('--gpu_id', default='0', type=str, help='GPU id to use')
    parser.add_argument(
        '--image_dir',
        default="VL_Debiasing/data/COCO/images/val2014",
        type=str,
        help='Directory containing COCO val2014 images',
    )

    parser.add_argument(
        '--results_filename',
        default="VL_Debiasing/results/clipcap_debiased.csv",
        type=str,
        help='Path to save baseline captioning results',
    )
    parser.add_argument(
        '--decoder_sfid',
        action='store_true',
        help='Enable decoder-only SFID intervention during caption generation.',
    )
    parser.add_argument(
        '--decoder_spd',
        action='store_true',
        help='Enable decoder-only SPD intervention during caption generation.',
    )
    parser.add_argument(
        '--decoder_train_embedding',
        default="VL_Debiasing/unified_debiasing/embedding/clip_cap_decoder_fairface_train.pt",
        type=str,
        help='Path to decoder SFID training embedding file (.pt).',
    )
    parser.add_argument(
        '--decoder_val_embedding',
        default="VL_Debiasing/unified_debiasing/embedding/clip_cap_decoder_fairface_test.pt",
        type=str,
        help='Path to decoder SFID validation/test embedding file (.pt).',
    )
    parser.add_argument(
        '--decoder_rf_checkpoint',
        default="VL_Debiasing/unified_debiasing/checkpoint/CLIPCAP_text_decoder_random_forest_model.joblib",
        type=str,
        help='Path to save/load decoder SFID random forest checkpoint.',
    )
    parser.add_argument(
        '--decoder_prune_num',
        default=50,
        type=int,
        help='Top-k decoder hidden dimensions to intervene on for SFID.',
    )
    parser.add_argument(
        '--decoder_threshold',
        default=0.9,
        type=float,
        help='Confidence threshold for selecting low-confidence samples in SFID.',
    )
    parser.add_argument(
        '--decoder_inlp_max_iter',
        default=5,
        type=int,
        help='Max INLP iterations for decoder SPD.',
    )
    parser.add_argument(
        '--decoder_inlp_tol',
        default=1e-6,
        type=float,
        help='INLP stopping tolerance (train acc threshold).',
    )
    parser.add_argument(
        '--decoder_inlp_C',
        default=1.0,
        type=float,
        help='LogisticRegression regularization strength for INLP.',
    )
    parser.add_argument(
        '--decoder_inlp_max_axes',
        default=None,
        type=int,
        help='Optional cap on number of INLP axes to use for SPD.',
    )
    parser.add_argument(
        '--decoder_inlp_axes_checkpoint',
        default="VL_Debiasing/checkpoint/CLIPCAP_text_decoder_inlp_axes.joblib",
        type=str,
        help='Path to cache INLP axes for decoder SPD.',
    )
    args = parser.parse_args()

    # Set CUDA device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Ensure imports can find project root
    sys.path.append('./')

    # Get NLTK resources (to scratch)
    nltk_data_dir = os.path.join(scratch_dir, "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('punkt_tab', download_dir=nltk_data_dir)


    # Reproducibility
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmarks = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Use scratch
    os.environ["HF_HOME"] = os.path.join(scratch_dir, "hf_cache")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(scratch_dir, "hf_cache")

    # # Load ClipCap model and CLIP image encoder
    prefix_length = 10
    model_path = 'VL_Debiasing/clip_debiasing/models/clipcap/clip_cap_coco_weight.pt'
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = model_clipcap.ClipCaptionModel(prefix_length, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    # THIS IS THE CODE FOR LOADING THE DEBIASED CLIP
    # clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model = model.eval().to(device)
    
    Debiased_clip_path = 'VL_Debiasing/vitb32_debiased_model/latest/Exp_512_1024_0.3_5e-06/best.pth'
    backbone = 'ViT-B/32'
    mlp1_hidden_size = 512
    mlp2_hidden_size = 1024
    alpha = 0.3
    clip_model = DebiasedCLIP(backbone, device=device, mlp1_hidden_size=mlp1_hidden_size, mlp2_hidden_size=mlp2_hidden_size, alpha=alpha).to(device)
    preprocess = clip_model.preprocess
    clip_model.load_state_dict(torch.load(Debiased_clip_path, map_location=device))
    # Unsure if we need this evaluation mode 
    # clip_model.eval()

    # Load COCO captions annotations
    with open('VL_Debiasing/data/COCO/annotations/captions_val2014.json', 'r') as json_data:
        d = json.load(json_data)
    annotations = d['annotations']

    # Map image_id -> list of ground-truth captions
    id_to_captions = {}
    for ann in annotations:
        image_id = ann['image_id']
        caption = ann['caption']
        if image_id not in id_to_captions:
            id_to_captions[image_id] = []
        id_to_captions[image_id].append(caption)

    # TODO: not just for gender
    # Ground-truth gender per image
    imid_2_gender = pickle.load(open('VL_Debiasing/clip_debiasing/models/clipcap/val_imid_gender.pkl', 'rb'))
    filtered_image_ids = set(imid_2_gender.keys())

    # Only keep images that have a gender label
    filtered_id_to_captions = {
        image_id: captions
        for image_id, captions in id_to_captions.items()
        if image_id in filtered_image_ids
    }

    # IDs to remove (e.g., problematic images)
    remove_id = pd.read_csv("VL_Debiasing/clip_debiasing/models/clipcap/remove_df.csv")['remove_id']

    results_filename = args.results_filename
    results = []
    text_important_indices = None
    text_inlp_axes = None
    text_mean_features_lowconfidence = None
    decoder_mode = None

    if args.decoder_sfid and args.decoder_spd:
        raise ValueError("Use only one of --decoder_sfid or --decoder_spd (mutually exclusive).")

    if args.decoder_sfid:
        print("Preparing decoder-only SFID statistics...")
        if args.decoder_prune_num <= 0:
            raise ValueError("--decoder_prune_num must be > 0 when --decoder_sfid is enabled.")
        text_important_indices, text_mean_features_lowconfidence = train_or_load_decoder_sfid(
            train_path=args.decoder_train_embedding,
            val_path=args.decoder_val_embedding,
            rf_checkpoint_path=args.decoder_rf_checkpoint,
            threshold=args.decoder_threshold,
            prune_num=args.decoder_prune_num,
        )
        text_important_indices = text_important_indices.to(device)
        text_mean_features_lowconfidence = text_mean_features_lowconfidence.to(device)
        decoder_mode = "sfid"
        print(
            "Decoder SFID ready: "
            f"prune_num={args.decoder_prune_num}, threshold={args.decoder_threshold}"
        )
    elif args.decoder_spd:
        print("Preparing decoder-only SPD statistics...")
        text_inlp_axes, text_mean_features_lowconfidence = train_or_load_decoder_spd(
            train_path=args.decoder_train_embedding,
            val_path=args.decoder_val_embedding,
            rf_checkpoint_path=args.decoder_rf_checkpoint,
            threshold=args.decoder_threshold,
            max_iter=args.decoder_inlp_max_iter,
            tol=args.decoder_inlp_tol,
            C=args.decoder_inlp_C,
            inlp_axes_checkpoint_path=args.decoder_inlp_axes_checkpoint,
            max_axes=args.decoder_inlp_max_axes,
        )
        text_inlp_axes = text_inlp_axes.to(device)
        text_mean_features_lowconfidence = text_mean_features_lowconfidence.to(device)
        decoder_mode = "spd"
        print(
            "Decoder SPD ready: "
            f"axes={tuple(text_inlp_axes.shape)}, threshold={args.decoder_threshold}"
        )

    # Only run generation if file does not already exist
    if not os.path.exists(results_filename):
        print("Results file does not exist")
        for image_id, gt_captions in tqdm(filtered_id_to_captions.items()):
            with torch.no_grad():
                if image_id in remove_id.values:
                    continue

                image_path = os.path.join(
                    args.image_dir,
                    f"COCO_val2014_{str(image_id).zfill(12)}.jpg",
                )
                image = Image.open(image_path).convert('RGB')
                ground_truth_gender = imid_2_gender[image_id]

                # Get CLIP image embedding (prefix)
                prefix = clip_model.encode_image(
                    preprocess(image).unsqueeze(0).to(device)
                ).float()

                # Generate caption without any debiasing
                generated_text = generate(
                    model,
                    tokenizer,
                    text_important_indices=text_important_indices,
                    text_mean_features_lowconfidence=text_mean_features_lowconfidence,
                    text_inlp_axes=text_inlp_axes,
                    embed=prefix,
                    mode=decoder_mode,
                )

                # Detect gender in generated text
                detected_gender = decide_gender(nltk.word_tokenize(generated_text))

                # Store result
                results.append(
                    {
                        'image_id': image_id,
                        'ground_truth_gender': ground_truth_gender,
                        'detected_gender': detected_gender,
                        'gt_captions': gt_captions,
                        'generated_text': generated_text,
                    }
                )

        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)
        df.to_csv(results_filename, index=False)
    else:
        print(f"Results file {results_filename} already exists. Skipping generation.")

    # Evaluate baseline image captioning and bias
    evaluate_image_captioning(results_filename)


if __name__ == "__main__":
    main()

