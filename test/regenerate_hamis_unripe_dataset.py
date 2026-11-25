"""Regenerate unripe dataset and assets using the user's exact papaya photo adjusted to super green.

- ONLY ONE pure white background (#FFFFFF)
- Zero inner white boxes / zero checkerboard artifacts
- Super green, spotless, smooth skin ('hamis nga walay lagom lagom')
- All 8 canonical camera views:
  1. Front View (Standing vertical, stem at top)
  2. Side View (Horizontal lying flat)
  3. Back View (Standing reversed)
  4. Opposite-Side View (Horizontal reversed)
  5. 45° Angled View (45° diagonal)
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

# Load the super green spotless transparent unripe papaya
fruit = Image.open(os.path.join(ROOT, "scratch", "test_opt_a.png")).convert("RGBA")
fruit = fruit.crop(fruit.getbbox())

# Top and Bottom spotless super green cutouts
top_cut = Image.open(os.path.join(ROOT, "scratch", "pure_white_unripe_top_view.png")).convert("RGBA")
top_cut = top_cut.crop(top_cut.getbbox())

bottom_cut = Image.open(os.path.join(ROOT, "scratch", "pure_white_unripe_bottom_view.png")).convert("RGBA")
bottom_cut = bottom_cut.crop(bottom_cut.getbbox())

# 8 canonical camera orientations
front_cut = fruit.rotate(-90, resample=Image.BICUBIC, expand=True) # Standing vertical upright (stem at top)
front_cut = front_cut.crop(front_cut.getbbox())

side_cut = fruit # Horizontal lying flat (stem on left, pointed tip on right)

back_cut = front_cut.transpose(Image.FLIP_LEFT_RIGHT) # Standing vertical reversed
opp_side_cut = side_cut.transpose(Image.FLIP_LEFT_RIGHT) # Horizontal reversed

angled_45_cut = fruit.rotate(45, resample=Image.BICUBIC, expand=True)
angled_45_cut = angled_45_cut.crop(angled_45_cut.getbbox())

three_quarter_cut = fruit.rotate(22, resample=Image.BICUBIC, expand=True)
three_quarter_cut = three_quarter_cut.crop(three_quarter_cut.getbbox())

canonical_views = {
    "unripe_01_front_view": {
        "title": "1. Front View (Standing)",
        "cutout": front_cut,
        "desc": "Vertical upright perspective"
    },
    "unripe_02_side_view": {
        "title": "2. Side View (Horizontal)",
        "cutout": side_cut,
        "desc": "Horizontal lateral perspective"
    },
    "unripe_03_back_view": {
        "title": "3. Back View (Reversed)",
        "cutout": back_cut,
        "desc": "180° reverse vertical perspective"
    },
    "unripe_04_opposite_side_view": {
        "title": "4. Opposite-Side View",
        "cutout": opp_side_cut,
        "desc": "Opposite horizontal perspective"
    },
    "unripe_05_45_angled_view": {
        "title": "5. 45° Angled View",
        "cutout": angled_45_cut,
        "desc": "45° diagonal perspective"
    },
    "unripe_06_three_quarter_view": {
        "title": "6. Three-Quarter View",
        "cutout": three_quarter_cut,
        "desc": "Three-quarter perspective"
    },
    "unripe_07_top_view": {
        "title": "7. Top View (Stem)",
        "cutout": top_cut,
        "desc": "Aerial top-down stem view"
    },
    "unripe_08_bottom_view": {
        "title": "8. Bottom View (Base)",
        "cutout": bottom_cut,
        "desc": "Base / blossom end view"
    }
}

# 1. Update unique_bases/unripe, assets/images/angles, assets/unripe_variants
ub_dir = os.path.join(ROOT, "unique_bases", "unripe")
angles_dir = os.path.join(ROOT, "assets", "images", "angles")
variants_dir = os.path.join(ROOT, "assets", "unripe_variants")

for d in [ub_dir, angles_dir, variants_dir]:
    os.makedirs(d, exist_ok=True)

print("Updating canonical base files...")
cw, ch = CANVAS
cutouts_list = []
for key, data in canonical_views.items():
    cut = data["cutout"]
    cutouts_list.append(cut)
    
    # Scale to canvas
    k = min((cw * 0.82) / cut.width, (ch * 0.82) / cut.height)
    nw = max(10, int(cut.width * k))
    nh = max(10, int(cut.height * k))
    res_cut = cut.resize((nw, nh), Image.Resampling.LANCZOS)
    
    # Pure white background (only ONE white bg, no borders, no inner boxes)
    white_bg = Image.new("RGB", (cw, ch), (255, 255, 255))
    px = (cw - nw) // 2
    py = (ch - nh) // 2
    white_bg.paste(res_cut, (px, py), res_cut.split()[3])
    
    # Save to unique_bases as unripe_XX_...
    white_bg.save(os.path.join(ub_dir, f"{key}.jpg"), "JPEG", quality=95)
    
    # Save to assets/images/angles
    white_bg.save(os.path.join(angles_dir, f"{key}.jpg"), "JPEG", quality=95)
    res_cut.save(os.path.join(angles_dir, f"{key}.png"), "PNG")
    
    # Save to assets/unripe_variants
    res_cut.save(os.path.join(variants_dir, f"{key}.png"), "PNG")
    print(f"  Saved {key}")

# Also update base_001.jpg .. base_075.jpg in unique_bases/unripe/
print("Updating base_001.jpg .. base_075.jpg in unique_bases/unripe/...")
for i in range(1, 76):
    cut = cutouts_list[(i - 1) % len(cutouts_list)]
    k = min((cw * 0.80) / cut.width, (ch * 0.80) / cut.height)
    nw = max(10, int(cut.width * k))
    nh = max(10, int(cut.height * k))
    res_cut = cut.resize((nw, nh), Image.Resampling.LANCZOS)
    
    white_bg = Image.new("RGB", (cw, ch), (255, 255, 255))
    px = (cw - nw) // 2
    py = (ch - nh) // 2
    white_bg.paste(res_cut, (px, py), res_cut.split()[3])
    white_bg.save(os.path.join(ub_dir, f"base_{i:03d}.jpg"), "JPEG", quality=95)

# 2. Update app images (transparent PNGs and white JPG)
print("\nUpdating app images (assets/images/)...")
pad = 28
cropped_fruit = fruit.crop(fruit.getbbox())
app_asset = Image.new("RGBA", (cropped_fruit.width + pad * 2, cropped_fruit.height + pad * 2), (0, 0, 0, 0))
app_asset.paste(cropped_fruit, (pad, pad), cropped_fruit.split()[3])

app_png_path = os.path.join(ROOT, "assets", "images", "papaya_unripe.png")
stage_png_path = os.path.join(ROOT, "assets", "images", "stage_unripe.png")
app_asset.save(app_png_path, "PNG")
app_asset.save(stage_png_path, "PNG")
print(f"  Saved {app_png_path} and {stage_png_path} ({app_asset.size})")

# Pure white JPG
app_jpg = Image.new("RGB", (cw, ch), (255, 255, 255))
k = min((cw * 0.82) / cropped_fruit.width, (ch * 0.82) / cropped_fruit.height)
nw = int(cropped_fruit.width * k)
nh = int(cropped_fruit.height * k)
res_fruit = cropped_fruit.resize((nw, nh), Image.Resampling.LANCZOS)
px = (cw - nw) // 2
py = (ch - nh) // 2
app_jpg.paste(res_fruit, (px, py), res_fruit.split()[3])
app_jpg_path = os.path.join(ROOT, "assets", "images", "papaya_unripe.jpg")
app_jpg.save(app_jpg_path, "JPEG", quality=95)
print(f"  Saved {app_jpg_path}")

# 3. Regenerate 500 images in papaya_dataset/unripe/
print("\nRegenerating 500 images in papaya_dataset/unripe/...")
dataset_dir = os.path.join(ROOT, "papaya_dataset", "unripe")
files = sorted(glob.glob(os.path.join(dataset_dir, "*.jpg")))

for idx, p in enumerate(files):
    base = cutouts_list[idx % len(cutouts_list)]
    subj = base.copy()
    
    # Apply natural subtle angle rotation / flip
    angle_choice = idx % 5
    if angle_choice == 1:
        subj = subj.rotate(90, resample=Image.BICUBIC, expand=True)
    elif angle_choice == 2:
        subj = subj.rotate(180, resample=Image.BICUBIC, expand=True)
    elif angle_choice == 3:
        subj = subj.rotate(270, resample=Image.BICUBIC, expand=True)
    elif angle_choice == 4:
        subj = subj.transpose(Image.FLIP_LEFT_RIGHT)
        
    cw_i, ch_i = CANVAS if angle_choice not in (1, 3) else (CANVAS[1], CANVAS[0])
    scale = random.uniform(0.78, 0.88)
    k = min(cw_i * scale / subj.width, ch_i * scale / subj.height)
    nw = max(10, int(subj.width * k))
    nh = max(10, int(subj.height * k))
    subj_res = subj.resize((nw, nh), Image.Resampling.LANCZOS)
    
    canvas = Image.new("RGB", (cw_i, ch_i), (255, 255, 255))
    dx = random.randint(-15, 15)
    dy = random.randint(-10, 10)
    px = (cw_i - nw) // 2 + dx
    py = (ch_i - nh) // 2 + dy
    
    canvas.paste(subj_res, (px, py), subj_res.split()[3])
    
    b_mult = random.uniform(0.97, 1.04)
    canvas = ImageEnhance.Brightness(canvas).enhance(b_mult)
    canvas.save(p, "JPEG", quality=93)

print(f"Regenerated {len(files)} dataset images with single pure white background!")

# 4. Build Overview Image on ONE single pure white background with subtle clean borders (NO inner boxes)
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

draw.text((40, 20), "Papaya Unripe • Super Green Multi-Angle Dataset", fill=(15, 23, 42), font=font_title)
draw.text((40, 50), "Single pure white background (#FFFFFF) • Super Green & Hamis (Spotless) • All 8 Camera Angles", fill=(100, 116, 139), font=font_sub)

cw_b, ch_b = 260, 250
start_x, start_y = 40, 85
pad_x, pad_y = 26, 20

for i, (key, data) in enumerate(canonical_views.items()):
    col = i % 4
    row = i // 4
    x = start_x + col * (cw_b + pad_x)
    y = start_y + row * (ch_b + pad_y)
    
    # Elegant light border outline on pure white (NO inner box)
    draw.rounded_rectangle([x, y, x + cw_b, y + ch_b], radius=10, outline=(226, 232, 240), width=1)
    
    cut = data["cutout"]
    box_w, box_h = cw_b - 30, ch_b - 55
    k = min(box_w / cut.width, box_h / cut.height)
    nw = max(10, int(cut.width * k))
    nh = max(10, int(cut.height * k))
    res = cut.resize((nw, nh), Image.Resampling.LANCZOS)
    
    px = x + (cw_b - nw) // 2
    py = y + 15 + (box_h - nh) // 2
    
    overview.paste(res, (px, py), res.split()[3])
    
    bbox_t = draw.textbbox((0, 0), data["title"], font=font_label)
    tw = bbox_t[2] - bbox_t[0]
    draw.text((x + (cw_b - tw) // 2, y + ch_b - 28), data["title"], fill=(30, 41, 59), font=font_label)

# Save overview to brain directory and scratch
brain_overview = os.path.join(BRAIN_DIR, "all_unripe_angles_hamis_overview.png")
overview.save(brain_overview)
overview.save(os.path.join(ROOT, "scratch", "all_unripe_angles_hamis_overview.png"))
print(f"Overview saved to {brain_overview}")

# Also verify card transparency on pastel green card (#DCFCE7) used in app
card = Image.new("RGBA", (340, 140), (220, 252, 231, 255))
card_draw = ImageDraw.Draw(card)
card_draw.rounded_rectangle([0, 0, 339, 139], radius=18, outline=(187, 247, 208, 255), width=2)
app_res = app_asset.resize((90, int(90 * app_asset.height / app_asset.width)), Image.Resampling.LANCZOS)
card.paste(app_res, (18, (140 - app_res.height) // 2), app_res.split()[3])

card_preview_path = os.path.join(BRAIN_DIR, "preview_unripe_card_verified.png")
card.save(card_preview_path)
print(f"Card preview saved to {card_preview_path}")

print("\nAll tasks completed successfully!")
