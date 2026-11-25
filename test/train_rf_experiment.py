import os
import glob
import random
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

random.seed(42)
np.random.seed(42)

CLASSES = ['overripe', 'ripe', 'ripening', 'unripe']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def extract_features(img_path_or_im):
    if isinstance(img_path_or_im, str):
        im = Image.open(img_path_or_im).convert('RGB')
    else:
        im = img_path_or_im.convert('RGB')
        
    w, h = im.size
    # Center crop 60% of the image to focus on the papaya inside the viewfinder
    crop_x1 = int(w * 0.15)
    crop_y1 = int(h * 0.15)
    crop_x2 = int(w * 0.85)
    crop_y2 = int(h * 0.85)
    cropped = im.crop((crop_x1, crop_y1, crop_x2, crop_y2)).resize((96, 96))
    
    arr = np.array(cropped, dtype=np.float32)
    # Reject extreme white background (distance to white < 35)
    dist_white = np.sqrt(np.sum((255.0 - arr) ** 2, axis=2))
    fruit_mask = dist_white > 30.0
    
    # If the image doesn't have white background (e.g. camera photo), use all pixels except extreme white highlights
    if np.sum(fruit_mask) < 200:
        fruit_mask = dist_white > 15.0
    if np.sum(fruit_mask) < 100:
        fruit_mask = np.ones((96, 96), dtype=bool)
        
    fp = arr[fruit_mask]
    r, g, b = fp[:, 0], fp[:, 1], fp[:, 2]
    
    mr, sr = float(np.mean(r)), float(np.std(r))
    mg, sg = float(np.mean(g)), float(np.std(g))
    mb, sb = float(np.mean(b)), float(np.std(b))
    
    rg_ratio = float(mr / (mg + 1e-5))
    rb_ratio = float(mr / (mb + 1e-5))
    greenness = float((2 * mg - mr - mb) / (2 * mg + mr + mb + 1e-5))
    yellowness = float((mr + mg - 2 * mb) / (mr + mg + 2 * mb + 1e-5))
    
    hsv = np.array(cropped.convert('HSV'), dtype=np.float32)[fruit_mask]
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2] # PIL scale: H in 0..255 (0..360 is h * 360/255)
    
    mh, sh = float(np.mean(h)), float(np.std(h))
    ms, ss = float(np.mean(s)), float(np.std(s))
    mv, sv = float(np.mean(v)), float(np.std(v))
    
    # In PIL HSV:
    # Red: 0..10 and 240..255
    # Orange/Amber: 11..25
    # Yellow: 26..42
    # Green: 43..100
    h_green = float(np.mean((h >= 43) & (h <= 100)))
    h_yellow = float(np.mean((h >= 26) & (h < 43)))
    h_orange = float(np.mean((h >= 11) & (h < 26)))
    h_amber = float(np.mean((h < 11) | (h > 240)))
    
    # Necrotic dark spots ON RIPE/OVERRIPE skin:
    # Only counts if the pixel is dark (v < 80) and surrounded by amber/orange, OR dark brown
    dark_spots = float(np.mean((v < 85) & (s > 30)))
    
    return [
        mr, sr, mg, sg, mb, sb,
        rg_ratio, rb_ratio, greenness, yellowness,
        mh, sh, ms, ss, mv, sv,
        h_green, h_yellow, h_orange, h_amber,
        dark_spots
    ]

print("Loading dataset samples...")
samples = []
for c in CLASSES:
    files = sorted(glob.glob(f"papaya_dataset/{c}/*.jpg"))
    print(f"  Class {c}: {len(files)} files")
    for f in files:
        samples.append((f, CLASS_TO_IDX[c]))

random.shuffle(samples)
print(f"Total samples: {len(samples)}")

X = []
y = []
for idx, (f, label) in enumerate(samples):
    X.append(extract_features(f))
    y.append(label)
    if (idx + 1) % 500 == 0:
        print(f"  Processed {idx + 1}...")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# Train/Val/Test split per class
X_train_l, y_train_l = [], []
X_val_l, y_val_l = [], []
X_test_l, y_test_l = [], []

for c_idx in range(4):
    c_indices = np.where(y == c_idx)[0]
    n = len(c_indices)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    X_train_l.append(X[c_indices[:n_train]])
    y_train_l.append(y[c_indices[:n_train]])
    
    X_val_l.append(X[c_indices[n_train:n_train + n_val]])
    y_val_l.append(y[c_indices[n_train:n_train + n_val]])
    
    X_test_l.append(X[c_indices[n_train + n_val:]])
    y_test_l.append(y[c_indices[n_train + n_val:]])

X_train = np.vstack(X_train_l)
y_train = np.concatenate(y_train_l)
X_val = np.vstack(X_val_l)
y_val = np.concatenate(y_val_l)
X_test = np.vstack(X_test_l)
y_test = np.concatenate(y_test_l)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    min_samples_split=6,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

val_acc = accuracy_score(y_val, rf.predict(X_val))
test_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"\n--- Random Forest Performance ---")
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Test Accuracy:       {test_acc * 100:.2f}%")
print("\nClassification Report (Test):")
print(classification_report(y_test, rf.predict(X_test), target_names=CLASSES))
print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, rf.predict(X_test)))

# Save model
os.makedirs("assets/models", exist_ok=True)
joblib.dump(rf, "assets/models/random_forest_papaya.joblib")
print("Saved assets/models/random_forest_papaya.joblib")
