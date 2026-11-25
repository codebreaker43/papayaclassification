"""Apply front, side, top, and bottom views to the ripe papaya dataset with clean transparency.

Updates:
- unique_bases/ripe/ (8 canonical views including bottom view)
- assets/images/angles/ (JPG and transparent PNG)
- assets/ripe_variants/ (transparent PNGs)
- papaya_dataset/ripe/ (500 balanced training images with angle semantics)
- Artifact overview image
"""
import os
import glob
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from scipy.ndimage import binary_fill_holes

ROOT = os.path.dirname(os.path.abspath(__file__))
BRAIN_DIR = r"C:\Users\Jess No Limit\.gemini\antigravity-ide\brain\c3d57fdf-9afe-4161-9368-cf5cc4af6fff"
CANVAS = (1024, 900)

random.seed(42)
np.random.seed(42)

# Source images
front_src = os.path.join(BRAIN_DIR, "papaya_ripe_front_view_1792032509258.jpg")
side_src = os.path.join(BRAIN_DIR, "papaya_ripe_side_view_1792032533679.jpg")
top_src = os.path.join(ROOT, "scratch", "top_view_enhanced.png")
bottom_src = os.path.join(BRAIN_DIR, "papaya_ripe_bottom_view_1792032469217.jpg")

def make_clean_cutout(img_path):
    im_rgb = Image.open(img_path).convert("RGB")
    arr = np.array(im_rgb, dtype=np.float32)
    hsv = np.array(im_rgb.convert("HSV"))
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    
    is_fg = (s > 22) | ((v < 130) & (s > 10))
    is_fg = binary_fill_holes(is_fg).astype(np.float32)
    
    m_img = Image.fromarray((is_fg * 255).astype(np.uint8))
    m_blurred = m_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    alpha = np.array(m_blurred, dtype=np.float32)
    
    out = np.zeros((im_rgb.height, im_rgb.width, 4), dtype=np.uint8)
    for c in range(3):
        out[:, :, c] = np.clip(arr[:, :, c], 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    res = Image.fromarray(out)
    bbox = res.getbbox()
    return res.crop(bbox) if bbox else res

print("Processing clean cutouts...")
front_cut = make_clean_cutout(front_src)
side_cut = make_clean_cutout(side_src)
top_cut = make_clean_cutout(top_src)
bottom_cut = make_clean_cutout(bottom_src)

# 8 canonical angle variants
canonical_views = {
    "ripe_01_front_view": {
        "title": "1. Front View (Standing)",
        "cutout": front_cut,
        "desc": "Vertical upright perspective"
    },
    "ripe_02_side_view": {
        "title": "2. Side View (Horizontal)",
        "cutout": side_cut,
        "desc": "Lateral horizontal perspective"
    },
    "ripe_03_back_view": {
        "title": "3. Back View",
        "cutout": front_cut.transpose(Image.FLIP_LEFT_RIGHT),
        "desc": "180° reverse vertical perspective"
    },
    "ripe_04_opposite_side_view": {
        "title": "4. Opposite-Side View",
        "cutout": side_cut.transpose(Image.FLIP_LEFT_RIGHT),
        "desc": "Opposite lateral perspective"
    },
    "ripe_05_45_angled_view": {
        "title": "5. 45° Angled View",
        "cutout": front_cut.rotate(-45, resample=Image.BICUBIC, expand=True),
        "desc": "45° diagonal angle"
    },
    "ripe_06_three_quarter_view": {
        "title": "6. Three-Quarter View",
        "cutout": front_cut.rotate(-22, resample=Image.BICUBIC, expand=True),
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

def place_on_canvas(cutout, target_size=CANVAS, scale=0.82, dx=0, dy=0):
    cw, ch = target_size
    k = min(cw * scale / cutout.width, ch * scale / cutout.height)
    new_w = max(10, int(cutout.width * k))
    new_h = max(10, int(cutout.height * k))
    resized = cutout.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    bg = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
    px = (cw - new_w) // 2 + dx
    py = (ch - new_h) // 2 + dy
    bg.paste(resized, (px, py), resized)
    return bg.convert("RGB"), resized

# 1. Update unique_bases/ripe, assets/images/angles, assets/ripe_variants
ub_dir = os.path.join(ROOT, "unique_bases", "ripe")
angles_dir = os.path.join(ROOT, "assets", "images", "angles")
variants_dir = os.path.join(ROOT, "assets", "ripe_variants")

for d in [ub_dir, angles_dir, variants_dir]:
    os.makedirs(d, exist_ok=True)

print("\nUpdating canonical base files...")
for key, data in canonical_views.items():
    cut = data["cutout"]
    rgb_img, _ = place_on_canvas(cut)
    
    # unique_bases
    ub_path = os.path.join(ub_dir, f"{key}.jpg")
    rgb_img.save(ub_path, "JPEG", quality=95)
    
    # assets/images/angles
    angles_jpg = os.path.join(angles_dir, f"{key}.jpg")
    angles_png = os.path.join(angles_dir, f"{key}.png")
    rgb_img.save(angles_jpg, "JPEG", quality=95)
    cut.save(angles_png, "PNG")
    
    # assets/ripe_variants
    var_png = os.path.join(variants_dir, f"{key}.png")
    cut.save(var_png, "PNG")
    
    print(f"  Saved {key}")

# 2. Regenerate papaya_dataset/ripe/*.jpg (500 files)
print("\nRegenerating 500 images in papaya_dataset/ripe/...")
dataset_dir = os.path.join(ROOT, "papaya_dataset", "ripe")
all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.jpg")))

cutout_pool = [
    front_cut,
    side_cut,
    top_cut,
    bottom_cut,
    canonical_views["ripe_05_45_angled_view"]["cutout"],
    canonical_views["ripe_06_three_quarter_view"]["cutout"],
]

for idx, p in enumerate(all_files):
    angle_type = os.path.splitext(p)[0].rsplit("_angle_", 1)[-1]
    
    base = cutout_pool[idx % len(cutout_pool)]
    subj = base.copy()
    
    if angle_type == "flip":
        subj = subj.transpose(Image.FLIP_LEFT_RIGHT)
    elif angle_type in ("090", "180", "270"):
        subj = subj.rotate(int(angle_type), resample=Image.BICUBIC, expand=True)
    
    scale = random.uniform(0.76, 0.88)
    dx = random.randint(-20, 20)
    dy = random.randint(-15, 15)
    
    canvas_dim = CANVAS if angle_type not in ("090", "270") else (CANVAS[1], CANVAS[0])
    img_rgb, _ = place_on_canvas(subj, target_size=canvas_dim, scale=scale, dx=dx, dy=dy)
    
    b_factor = random.uniform(0.95, 1.05)
    img_rgb = ImageEnhance.Brightness(img_rgb).enhance(b_factor)
    img_rgb.save(p, "JPEG", quality=92)

print(f"Successfully regenerated {len(all_files)} images in {dataset_dir}")

# 3. Create visual overview contact sheet
print("\nGenerating overview contact sheet...")
sheet_w = 1200
sheet_h = 680
canvas_sheet = Image.new("RGB", (sheet_w, sheet_h), (248, 250, 252))
draw = ImageDraw.Draw(canvas_sheet)

try:
    font_h = ImageFont.truetype("arialbd.ttf", 22)
    font_sub = ImageFont.truetype("arial.ttf", 13)
    font_card = ImageFont.truetype("arialbd.ttf", 12)
    font_desc = ImageFont.truetype("arial.ttf", 11)
except:
    font_h = ImageFont.load_default()
    font_sub = ImageFont.load_default()
    font_card = ImageFont.load_default()
    font_desc = ImageFont.load_default()

draw.text((36, 20), "Papaya Ripe • Multi-Angle Dataset (Front, Side, Top & Bottom View)", fill=(15, 23, 42), font=font_h)
draw.text((36, 50), "All 8 camera angles applied to unique_bases, assets, and 500-image training dataset", fill=(100, 116, 139), font=font_sub)

cw, ch = 260, 260
start_x, start_y = 36, 85
pad_x, pad_y = 25, 20

view_items = list(canonical_views.items())
for idx, (key, data) in enumerate(view_items):
    col = idx % 4
    row = idx // 4
    x = start_x + col * (cw + pad_x)
    y = start_y + row * (ch + pad_y)
    
    # Card bg
    draw.rounded_rectangle([x, y, x + cw, y + ch], radius=14, fill=(255, 237, 213), outline=(254, 215, 170), width=1)
    
    # Header badge
    draw.rounded_rectangle([x + 12, y + 10, x + cw - 12, y + 34], radius=8, fill=(234, 88, 12))
    draw.text((x + 18, y + 14), data["title"], fill=(255, 255, 255), font=font_card)
    
    # Image thumbnail (floating seamlessly with transparent alpha)
    cut = data["cutout"]
    max_dim = 160
    k = min(max_dim / cut.width, max_dim / cut.height)
    thumb = cut.resize((int(cut.width * k), int(cut.height * k)), Image.Resampling.LANCZOS)
    
    tx = x + (cw - thumb.width) // 2
    ty = y + 42 + (max_dim - thumb.height) // 2
    canvas_sheet.paste(thumb, (tx, ty), thumb)
    
    # Description footer
    draw.text((x + 16, y + ch - 22), data["desc"], fill=(154, 52, 18), font=font_desc)

out_overview = os.path.join(BRAIN_DIR, "all_ripe_angles_overview.png")
canvas_sheet.save(out_overview)
print(f"Saved overview to {out_overview}")
print("ALL TASKS COMPLETED!")
