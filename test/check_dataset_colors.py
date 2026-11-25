import glob
import numpy as np
from PIL import Image
import joblib

for stage in ['unripe', 'ripening', 'ripe', 'overripe']:
    files = glob.glob(f'papaya_dataset/{stage}/*.jpg')[:5]
    print(f"=== Stage: {stage} (sample {len(files)}) ===")
    for f in files:
        im = Image.open(f).convert('RGB')
        arr = np.array(im, dtype=np.float32)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        mr, mg, mb = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
        hsv = np.array(im.convert('HSV'), dtype=np.float32)
        h = hsv[:,:,0] # 0..255 in PIL
        print(f"  {f}: Mean R={mr:.1f}, G={mg:.1f}, B={mb:.1f}, R/G={mr/(mg+1e-5):.2f}, Mean H={float(np.mean(h)):.1f}")
