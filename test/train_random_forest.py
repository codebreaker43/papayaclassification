"""Train a Random Forest Classifier for Papaya Maturity Classification.

- Dataset: Combines papaya_curated_dataset (multi-angle, realistic & studio backgrounds)
           + papaya_dataset (clean studio cutouts)
- Exact pixel-level parity with Flutter/Dart RandomForestService (96x96 grid, perimeter background
  rejection, shadow filtering, and PIL-scale HSV math).
- Outputs:
    assets/models/random_forest_papaya.joblib
    assets/models/random_forest_papaya.json
"""
import os
import glob
import json
import math
import joblib
import random
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CURATED_DIR = os.path.join(BASE_DIR, "papaya_curated_dataset")
STUDIO_DIR = os.path.join(BASE_DIR, "papaya_dataset")
MODELS_DIR = os.path.join(BASE_DIR, "assets", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CLASSES = ["overripe", "ripe", "ripening", "unripe"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

FEATURE_NAMES = [
    "mean_r", "std_r", "mean_g", "std_g", "mean_b", "std_b",
    "rg_ratio", "rb_ratio", "greenness", "yellowness",
    "mean_h", "std_h", "mean_s", "std_s", "mean_v", "std_v",
    "h_green", "h_yellow", "h_orange", "h_amber",
    "dark_spot_ratio"
]


def _rgb_to_hsv_pil(rf, gf, bf):
    mx = max(rf, gf, bf)
    mn = min(rf, gf, bf)
    d = mx - mn
    h = 0.0
    if d > 0:
        if mx == rf:
            h = 60.0 * (((gf - bf) / d) % 6.0)
        elif mx == gf:
            h = 60.0 * (((bf - rf) / d) + 2.0)
        else:
            h = 60.0 * (((rf - gf) / d) + 4.0)
        if h < 0:
            h += 360.0
    val_h = (h / 360.0) * 255.0
    val_s = 0.0 if mx == 0 else (d / mx) * 255.0
    val_v = mx
    return val_h, val_s, val_v


def extract_features(img_path_or_pil):
    if isinstance(img_path_or_pil, str):
        im = Image.open(img_path_or_pil).convert("RGBA")
    else:
        im = img_path_or_pil.convert("RGBA")

    dim = 96
    im = im.resize((dim, dim), Image.Resampling.BILINEAR)
    raw = np.array(im, dtype=np.uint8)

    # 8 perimeter reference points to detect background color (wood, counter, foliage, etc.)
    ref_coords = [
        (3, 3), (3, 48), (3, 92),
        (48, 3),         (48, 92),
        (92, 3), (92, 48), (92, 92)
    ]
    ref_pts = [
        (float(raw[r, c, 0]), float(raw[r, c, 1]), float(raw[r, c, 2]))
        for (r, c) in ref_coords
    ]

    def run_pass(min_coord, max_coord, use_bg_filter, min_sat, min_val, white_thr):
        sR = sG = sB = sR2 = sG2 = sB2 = 0.0
        sH = sS = sV = sH2 = sS2 = sV2 = 0.0
        gn = yw = orr = am = dark_spots = 0
        n = 0

        for row in range(min_coord, max_coord):
            for col in range(min_coord, max_coord):
                r = int(raw[row, col, 0])
                g = int(raw[row, col, 1])
                b = int(raw[row, col, 2])
                a = int(raw[row, col, 3])
                if a < 50:
                    continue

                dist_white = math.sqrt((255 - r) ** 2 + (255 - g) ** 2 + (255 - b) ** 2)
                if dist_white < white_thr:
                    continue

                rf, gf, bf = float(r), float(g), float(b)
                val_h, val_s, val_v = _rgb_to_hsv_pil(rf, gf, bf)

                if val_v < min_val or val_s < min_sat:
                    continue

                if use_bg_filter:
                    is_bg = False
                    for (pr, pg, pb) in ref_pts:
                        d_bg = math.sqrt((rf - pr) ** 2 + (gf - pg) ** 2 + (bf - pb) ** 2)
                        if d_bg < 26.0:
                            is_bg = True
                            break
                    if is_bg:
                        continue

                sR += rf; sG += gf; sB += bf
                sR2 += rf * rf; sG2 += gf * gf; sB2 += bf * bf

                sH += val_h; sS += val_s; sV += val_v
                sH2 += val_h * val_h; sS2 += val_s * val_s; sV2 += val_v * val_v

                if 43.0 <= val_h <= 105.0:
                    gn += 1
                elif 25.0 <= val_h < 43.0:
                    yw += 1
                elif 18.0 <= val_h < 25.0:
                    orr += 1
                elif val_h < 18.0 or val_h > 240.0:
                    am += 1

                if val_v < 115.0 and val_s > 45.0 and val_h < 32.0:
                    dark_spots += 1

                n += 1

        return n, (sR, sG, sB, sR2, sG2, sB2, sH, sS, sV, sH2, sS2, sV2, gn, yw, orr, am, dark_spots)

    # Pass 1: Center region with background perimeter rejection
    n, stats = run_pass(18, 78, True, 52.0, 45.0, 32.0)
    # Pass 2: Inner core fallback if fruit fills the corners or matches background
    if n < 60:
        n, stats = run_pass(24, 72, False, 35.0, 40.0, 25.0)
    if n < 15:
        n, stats = run_pass(28, 68, False, 0.0, 0.0, 0.0)
    if n < 5:
        return [0.0] * 21

    sR, sG, sB, sR2, sG2, sB2, sH, sS, sV, sH2, sS2, sV2, gn, yw, orr, am, dark_spots = stats
    nd = float(n)
    mR, mG, mB = sR / nd, sG / nd, sB / nd
    mH, mS, mV = sH / nd, sS / nd, sV / nd

    return [
        mR, math.sqrt(max(0.0, sR2 / nd - mR * mR)),
        mG, math.sqrt(max(0.0, sG2 / nd - mG * mG)),
        mB, math.sqrt(max(0.0, sB2 / nd - mB * mB)),
        mR / (mG + 1e-5),
        mR / (mB + 1e-5),
        (2.0 * mG - mR - mB) / (2.0 * mG + mR + mB + 1e-5),
        (mR + mG - 2.0 * mB) / (mR + mG + 2.0 * mB + 1e-5),
        mH, math.sqrt(max(0.0, sH2 / nd - mH * mH)),
        mS, math.sqrt(max(0.0, sS2 / nd - mS * mS)),
        mV, math.sqrt(max(0.0, sV2 / nd - mV * mV)),
        gn / nd, yw / nd, orr / nd, am / nd,
        dark_spots / nd,
    ]


if __name__ == "__main__":
    print("=== Step 1: Extracting Features from Multi-Background & Studio Datasets ===")
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for c in CLASSES:
        label = CLASS_TO_IDX[c]
        train_files = sorted(glob.glob(os.path.join(CURATED_DIR, "train", c, "*.jpg")))
        # Also include 150 clean studio images per class in training
        studio_files = sorted(glob.glob(os.path.join(STUDIO_DIR, c, "*.jpg")))[:150]
        val_files = sorted(glob.glob(os.path.join(CURATED_DIR, "val", c, "*.jpg")))
        test_files = sorted(glob.glob(os.path.join(CURATED_DIR, "test", c, "*.jpg")))

        print(f"Class '{c}': {len(train_files) + len(studio_files)} train, {len(val_files)} val, {len(test_files)} test")
        for p in train_files + studio_files:
            X_train.append(extract_features(p))
            y_train.append(label)
        for p in val_files:
            X_val.append(extract_features(p))
            y_val.append(label)
        for p in test_files:
            X_test.append(extract_features(p))
            y_test.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)

    print("\n=== Step 2: Training Random Forest Classifier ===")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    val_preds = rf.predict(X_val)
    test_preds = rf.predict(X_test)
    print(f"Validation Accuracy: {accuracy_score(y_val, val_preds) * 100:.2f}%")
    print(f"Holdout Test Accuracy (Realistic & Studio Backgrounds): {accuracy_score(y_test, test_preds) * 100:.2f}%\n")

    print("Confusion Matrix (Holdout Test Set):")
    print(f"Classes: {CLASSES}")
    print(confusion_matrix(y_test, test_preds))

    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=CLASSES, digits=4))

    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("Top 10 Most Important Features:")
    for rank, idx in enumerate(sorted_idx[:10]):
        print(f"  {rank+1}. {FEATURE_NAMES[idx]:18s}: {importances[idx]*100:.2f}%")

    joblib_path = os.path.join(MODELS_DIR, "random_forest_papaya.joblib")
    joblib.dump(rf, joblib_path)
    print(f"\nSaved joblib model to: {joblib_path}")

    def serialize_forest(forest, feature_names, class_names):
        trees = []
        for dt in forest.estimators_:
            tree = dt.tree_
            trees.append({
                "children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": np.round(tree.threshold, 5).tolist(),
                "value": np.round(tree.value.squeeze(axis=1), 4).tolist(),
            })
        return {
            "algorithm": "RandomForestClassifier",
            "n_estimators": len(trees),
            "n_classes": len(class_names),
            "classes": class_names,
            "feature_names": feature_names,
            "trees": trees,
        }

    json_path = os.path.join(MODELS_DIR, "random_forest_papaya.json")
    with open(json_path, "w") as f:
        json.dump(serialize_forest(rf, FEATURE_NAMES, CLASSES), f, separators=(",", ":"))
    print(f"Saved JSON model to: {json_path}")
