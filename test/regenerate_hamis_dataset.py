"""Regenerate ripe dataset using the exact smooth, spotless papaya from user photo.

- ONLY ONE pure white background (#FFFFFF)
- Zero inner white boxes / zero shadow artifacts
- Spotless, smooth skin ('hamis nga walay lagom lagom')
- Canonical orientation:
  1. Front View (Standing vertical, stem at top)
  2. Side View (Horizontal lying flat)
  3. Back View (Standing reversed)
  4. Opposite-Side View (Horizontal reversed)
  5. 45° Angled View (Original pose)
  6. Three-Quarter View (3/4 perspective)
  7. Top View (Stem aerial)
  8. Bottom View (Base aerial)
"""
import os
import glob
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
BRAIN_DIR = r"C:\Users\Jess No Limit\.gemini\antigravity-ide\brain\c3d57fdf-9afe-4161-9368-cf5cc4af6fff"
CANVAS = (1024, 900)

random.seed(42)
np.random.seed(42)

# Load the user's exact spotless transparent papaya
fruit = Image.open(os.path.join(ROOT, "scratch", "test_flawless_ripe.png")).convert("RGBA")
fruit = fruit.crop(fruit.getbbox())

# Top and Bottom spotless cutouts
top_cut = Image.open(os.path.join(ROOT, "scratch", "pure_white_top_view.png")).convert("RGBA")
top_cut = top_cut.crop(top_cut.getbbox())

bottom_cut = Image.open(os.path.join(ROOT, "scratch", "pure_white_bottom_view.png")).convert("RGBA")
bottom_cut = bottom_cut.crop(bottom_cut.getbbox())

# Orientations
front_cut = fruit.rotate(52, resample=Image.BICUBIC, expand=True) # Standing vertical upright
front_cut = front_cut.crop(front_cut.getbbox())

side_cut = fruit.rotate(-38, resample=Image.BICUBIC, expand=True) # Horizontal lying flat
side_cut = side_cut.crop(side_cut.getbbox())

back_cut = front_cut.transpose(Image.FLIP_LEFT_RIGHT)
opp_side_cut = side_cut.transpose(Image.FLIP_LEFT_RIGHT)
angled_45_cut = fruit # Original uploaded angle
three_quarter_cut = fruit.rotate(18, resample=Image.BICUBIC, expand=True)
three_quarter_cut = three_quarter_cut.crop(three_quarter_cut.getbbox())

canonical_views = {
    "ripe_01_front_view": {
        "title": "1. Front View (Standing)",
        "cutout": front_cut,
        "desc": "Vertical upright perspective"
    },
    "ripe_02_side_view": {
        "title": "2. Side View (Horizontal)",
        "cutout": side_cut,
        "desc": "Horizontal lateral perspective"
    },
    "ripe_03_back_view": {
        "title": "3. Back View (Reversed)",
        "cutout": back_cut,
        "desc": "180° reverse vertical perspective"
    },
    "ripe_04_opposite_side_view": {
        "title": "4. Opposite-Side View",
        "cutout": opp_side_cut,
        "desc": "Opposite horizontal perspective"
    },
    "ripe_05_45_angled_view": {
        "title": "5. 45° Angled View (Original)",
        "cutout": angled_45_cut,
        "desc": "45° diagonal perspective"
    },
    "ripe_06_three_quarter_view": {
        "title": "6. Three-Quarter View",
        "cutout": three_quarter_cut,
        "desc": "Three-quarter perspective"
    },
    "ripe_07_top_view": {
        "title": "7. Top View (Stem)",
        "cutout": top_cut,
        "desc": "Aerial top-down stem view"
    },
    "ripe_08_bottom_view": {
        "title": "8. Bottom View (Base)",
        "cutout": bottom_cut,
        "desc": "Base / blossom end view"
    }
}

# 1. Update unique_bases/ripe, assets/images/angles, assets/ripe_variants
ub_dir = os.path.join(ROOT, "unique_bases", "ripe")
angles_dir = os.path.join(ROOT, "assets", "images", "angles")
variants_dir = os.path.join(ROOT, "assets", "ripe_variants")

for d in [ub_dir, angles_dir, variants_dir]:
    os.makedirs(d, exist_ok=True)

print("Updating canonical base files...")
for key, data in canonical_views.items():
    cut = data["cutout"]
    
    # Scale to canvas
    cw, ch = CANVAS
    k = min((cw * 0.82) / cut.width, (ch * 0.82) / cut.height)
    nw = max(10, int(cut.width * k))
    nh = max(10, int(cut.height * k))
    res_cut = cut.resize((nw, nh), Image.Resampling.LANCZOS)
    
    # Pure white background (only ONE white bg, no borders, no inner boxes)
    white_bg = Image.new("RGB", (cw, ch), (255, 255, 255))
    px = (cw - nw) // 2
    py = (ch - nh) // 2
    white_bg.paste(res_cut, (px, py), res_cut.split()[3])
    
    # Save to unique_bases
    white_bg.save(os.path.join(ub_dir, f"{key}.jpg"), "JPEG", quality=95)
    
    # Save to assets/images/angles
    white_bg.save(os.path.join(angles_dir, f"{key}.jpg"), "JPEG", quality=95)
    res_cut.save(os.path.join(angles_dir, f"{key}.png"), "PNG")
    
    # Save to assets/ripe_variants
    res_cut.save(os.path.join(variants_dir, f"{key}.png"), "PNG")
    print(f"  Saved {key}")

# 2. Regenerate 500 images in papaya_dataset/ripe/
print("\nRegenerating 500 images in papaya_dataset/ripe/...")
dataset_dir = os.path.join(ROOT, "papaya_dataset", "ripe")
files = sorted(glob.glob(os.path.join(dataset_dir, "*.jpg")))

cutout_list = [d["cutout"] for d in canonical_views.values()]

for idx, p in enumerate(files):
    angle_type = os.path.splitext(p)[0].rsplit("_angle_", 1)[-1]
    
    base = cutout_list[idx % len(cutout_list)]
    subj = base.copy()
    
    if angle_type == "flip":
        subj = subj.transpose(Image.FLIP_LEFT_RIGHT)
    elif angle_type in ("090", "180", "270"):
        subj = subj.rotate(int(angle_type), resample=Image.BICUBIC, expand=True)
        
    cw, ch = CANVAS if angle_type not in ("090", "270") else (CANVAS[1], CANVAS[0])
    scale = random.uniform(0.78, 0.88)
    k = min(cw * scale / subj.width, ch * scale / subj.height)
    nw = max(10, int(subj.width * k))
    nh = max(10, int(subj.height * k))
    subj_res = subj.resize((nw, nh), Image.Resampling.LANCZOS)
    
    canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
    dx = random.randint(-15, 15)
    dy = random.randint(-10, 10)
    px = (cw - nw) // 2 + dx
    py = (ch - nh) // 2 + dy
    
    canvas.paste(subj_res, (px, py), subj_res.split()[3])
    
    b_mult = random.uniform(0.97, 1.04)
    canvas = ImageEnhance.Brightness(canvas).enhance(b_mult)
    canvas.save(p, "JPEG", quality=93)

print(f"Regenerated {len(files)} dataset images with pure white background!")

# 3. Build Overview Image on ONE single pure white background with subtle clean borders (NO inner boxes)
print("\nGenerating overview contact sheet...")
sheet_w = 1200
sheet_h = 660
overview = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255)) # ONE pure white background
draw = ImageDraw.Draw(overview)

try:
    font_title = ImageFont.truetype("arialbd.ttf", 22)
    font_sub = ImageFont.truetype("arial.ttf", 13)
    font_label = ImageFont.truetype("arialbd.ttf", 12)
except:
    font_title = ImageFont.load_default()
    font_sub = ImageFont.load_default()
    font_label = ImageFont.load_default()

draw.text((40, 20), "Papaya Ripe • Hamis & Spotless Multi-Angle Dataset", fill=(15, 23, 42), font=font_title)
draw.text((40, 50), "Single pure white background (#FFFFFF) • Hamis nga panit (Spotless) • All 8 Camera Angles", fill=(100, 116, 139), font=font_sub)

cw, ch = 260, 250
start_x, start_y = 40, 85
pad_x, pad_y = 26, 20

for i, (key, data) in enumerate(canonical_views.items()):
    col = i % 4
    row = i // 4
    x = start_x + col * (cw + pad_x)
    y = start_y + row * (ch + pad_y)
    
    # Elegant light border outline on pure white (NO inner box)
    draw.rounded_rectangle([x, y, x + cw, y + ch], radius=12, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
    
    # Orange label pill
    draw.rounded_rectangle([x + 10, y + 10, x + cw - 10, y + 32], radius=6, fill=(234, 88, 12))
    draw.text((x + 16, y + 14), data["title"], fill=(255, 255, 255), font=font_label)
    
    # Thumbnail
    cut = data["cutout"]
    max_dim = 160
    k_scale = min(max_dim / cut.width, max_dim / cut.height)
    nw = int(cut.width * k_scale)
    nh = int(cut.height * k_scale)
    thumb = cut.resize((nw, nh), Image.Resampling.LANCZOS)
    
    tx = x + (cw - nw) // 2
    ty = y + 42 + (max_dim - nh) // 2
    
    overview.paste(thumb, (tx, ty), thumb.split()[3])

out_path = os.path.join(BRAIN_DIR, "all_ripe_angles_hamis_overview.png")
overview.save(out_path)
print(f"Saved overview to {out_path}")
print("Done!")
