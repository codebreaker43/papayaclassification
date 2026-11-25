"""Replace the old ripe papaya images in the dataset with the new ripe papaya.

- Backs up the old files to scratch/backup_old_ripe/
- Regenerates papaya_dataset/ripe/*.jpg (same filenames, angle semantics honored)
- Regenerates unique_bases/ripe/*.jpg (8 base views)
"""
import os
import glob
import random
import shutil
from PIL import Image, ImageEnhance

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "assets", "images", "papaya_ripe.png")  # new transparent cutout
BACKUP = os.path.join(ROOT, "scratch", "backup_old_ripe")
CANVAS = (1024, 900)

random.seed(7)
fruit = Image.open(SRC).convert("RGBA")
fruit = fruit.crop(fruit.getbbox())


def render(angle, scale=0.86, brightness=1.0, dx=0, dy=0):
    f = fruit
    if angle == "flip":
        f = f.transpose(Image.FLIP_LEFT_RIGHT)
    elif angle in ("090", "180", "270"):
        f = f.rotate(int(angle), resample=Image.BICUBIC, expand=True)

    cw, ch = CANVAS if angle not in ("090", "270") else (CANVAS[1], CANVAS[0])
    k = min(cw * scale / f.width, ch * scale / f.height)
    f = f.resize((int(f.width * k), int(f.height * k)), Image.LANCZOS)

    bg = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
    bg.alpha_composite(f, ((cw - f.width) // 2 + dx, (ch - f.height) // 2 + dy))
    out = bg.convert("RGB")
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)
    return out


def backup(folder):
    dst = os.path.join(BACKUP, os.path.relpath(folder, ROOT))
    if not os.path.exists(dst):
        shutil.copytree(folder, dst)
        print(f"Backed up {folder} -> {dst}")


# 1. papaya_dataset/ripe
ds = os.path.join(ROOT, "papaya_dataset", "ripe")
backup(ds)
files = sorted(glob.glob(os.path.join(ds, "*.jpg")))
for p in files:
    angle = os.path.splitext(p)[0].rsplit("_angle_", 1)[-1]
    img = render(
        angle,
        scale=random.uniform(0.78, 0.90),
        brightness=random.uniform(0.95, 1.05),
        dx=random.randint(-20, 20),
        dy=random.randint(-15, 15),
    )
    img.save(p, "JPEG", quality=92)
print(f"Regenerated {len(files)} images in {ds}")

# 2. unique_bases/ripe (8 base views)
ub = os.path.join(ROOT, "unique_bases", "ripe")
backup(ub)
view_map = {
    "front": "000", "side": "flip", "back": "flip", "opposite_side": "000",
    "45_angled": "000", "three_quarter": "flip", "high_angle": "000", "top": "000",
}
ub_files = sorted(glob.glob(os.path.join(ub, "*.jpg")))
for i, p in enumerate(ub_files):
    name = os.path.basename(p)
    angle = next((a for v, a in view_map.items() if f"_{v}_view" in name), "000")
    img = render(angle, scale=0.84 + 0.01 * (i % 4), brightness=0.97 + 0.02 * (i % 3))
    img.save(p, "JPEG", quality=95)
print(f"Regenerated {len(ub_files)} images in {ub}")
