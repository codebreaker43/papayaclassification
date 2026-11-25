import os
import glob
import joblib
import numpy as np
from train_random_forest import extract_features, CLASSES

rf = joblib.load("assets/models/random_forest_papaya.joblib")

test_imgs = sorted(
    glob.glob("scratch_test_*.png")
    + glob.glob("assets/images/papaya_unripe.png")
    + glob.glob("assets/images/papaya_ripening.png")
    + glob.glob("assets/images/papaya_ripe.png")
    + glob.glob("assets/images/papaya_overripe.png")
)
print("Found test images:", len(test_imgs))
for p in test_imgs:
    f = extract_features(p)
    probs = rf.predict_proba([f])[0]
    pred = CLASSES[np.argmax(probs)]
    print(f"{os.path.basename(p):35} -> {pred.upper():10} (Conf: {np.max(probs)*100:.1f}%) Probs: {np.round(probs, 3)}")
