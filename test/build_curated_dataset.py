import os
import glob
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

# Seed for deterministic and reproducible generation
random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "papaya_curated_dataset")
TARGET_SIZE = (224, 224)


def make_pure_white_background(w, h):
    tone = random.choice([255, 253, 250, 246])
    return Image.new("RGB", (w, h), (tone, tone, tone))


def make_wood_background(w, h):
    wood_types = [
        ([175, 125, 80], 8.0, 10.0),   # Oak
        ([110, 70, 45], 6.0, 12.0),    # Walnut / Mahogany
        ([205, 160, 105], 10.0, 8.0),  # Light Pine
    ]
    base_color, grain_freq, grain_amp = random.choice(wood_types)
    y_coords = np.linspace(0, 10, h)[:, None]
    grain = np.sin(y_coords * grain_freq + random.uniform(0, 5)) * grain_amp
    noise = np.random.normal(0, 5.0, (h, w, 1))

    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        channel = base_color[c] + grain + noise[:, :, 0]
        bg[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
    return Image.fromarray(bg)


def make_kitchen_counter(w, h):
    counter_styles = [
        [215, 215, 215],  # Light Granite
        [50, 55, 60],     # Dark Slate / Charcoal
        [190, 185, 175],  # Travertine Marble
        [135, 138, 142],  # Concrete Grey
    ]
    base = random.choice(counter_styles)
    noise = np.random.normal(0, 10.0, (h, w, 3))
    bg = np.clip(np.array(base, dtype=float) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(bg)


def make_foliage_background(w, h):
    bg = np.zeros((h, w, 3), dtype=float)
    bg[:, :, 0] = np.random.uniform(40, 75)
    bg[:, :, 1] = np.random.uniform(85, 135)
    bg[:, :, 2] = np.random.uniform(30, 60)

    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    for _ in range(5):
        bx = random.randint(0, w)
        by = random.randint(0, h)
        rad = random.randint(30, 80)
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.ellipse(
            [bx - rad, by - rad, bx + rad, by + rad],
            fill=(
                random.randint(85, 150),
                random.randint(140, 200),
                random.randint(45, 95),
                75,
            ),
        )
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return img.filter(ImageFilter.GaussianBlur(radius=8))


def make_neutral_studio(w, h):
    studio_tones = [
        [245, 245, 245],  # Clean off-white
        [220, 222, 225],  # Soft grey
        [170, 172, 175],  # Neutral studio grey
        [32, 34, 38],     # Matte black
    ]
    base = random.choice(studio_tones)
    x = np.linspace(-1, 1, w)[None, :]
    y = np.linspace(-1, 1, h)[:, None]
    vignette = 1.0 - 0.22 * (x**2 + y**2)
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        channel = base[c] * vignette + np.random.normal(0, 3.0, (h, w))
        bg[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
    return Image.fromarray(bg)


def make_cardboard_market(w, h):
    base_color = np.array([185, 148, 105], dtype=float)
    noise = np.random.normal(0, 9.0, (h, w, 3))
    return Image.fromarray(np.clip(base_color + noise, 0, 255).astype(np.uint8))


BACKGROUND_GENERATORS = [
    make_pure_white_background,
    make_pure_white_background,  # 2x weight for clean studio/white backgrounds
    make_wood_background,
    make_kitchen_counter,
    make_foliage_background,
    make_neutral_studio,
    make_cardboard_market,
]


def extract_fruit_mask(img_rgba):
    if img_rgba.mode != "RGBA":
        img_rgba = img_rgba.convert("RGBA")
    raw = np.array(img_rgba)

    alpha = raw[:, :, 3]
    if np.min(alpha) < 50:
        # Also clean any residual near-white corner pixels
        r, g, b = raw[:, :, 0], raw[:, :, 1], raw[:, :, 2]
        is_white = (r > 242) & (g > 242) & (b > 242)
        raw[is_white, 3] = 0
        out = Image.fromarray(raw)
        bbox = out.getbbox()
        return out.crop(bbox) if bbox else out

    r, g, b = raw[:, :, 0], raw[:, :, 1], raw[:, :, 2]
    brightness = (r.astype(float) + g.astype(float) + b.astype(float)) / 3.0
    channel_range = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

    is_bg = (brightness > 232) & (channel_range < 24)
    raw[is_bg, 3] = 0
    out = Image.fromarray(raw)
    bbox = out.getbbox()
    return out.crop(bbox) if bbox else out


def composite_fruit(
    fruit_rgba,
    target_size=(224, 224),
    scale_range=(0.64, 0.88),
    rot_range=(-35, 35),
    force_white_bg=False,
):
    tw, th = target_size
    bg_func = make_pure_white_background if force_white_bg else random.choice(BACKGROUND_GENERATORS)
    bg = bg_func(tw, th).convert("RGBA")

    fruit = fruit_rgba.copy()
    if random.random() > 0.5:
        fruit = fruit.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.8:
        fruit = fruit.transpose(Image.FLIP_TOP_BOTTOM)

    angle = random.uniform(rot_range[0], rot_range[1])
    fruit = fruit.rotate(angle, resample=Image.BICUBIC, expand=True)
    bbox = fruit.getbbox()
    if bbox:
        fruit = fruit.crop(bbox)

    target_fruit_dim = int(min(tw, th) * random.uniform(scale_range[0], scale_range[1]))
    fw, fh = fruit.size
    scale_factor = target_fruit_dim / max(fw, fh, 1)
    new_fw = max(10, int(fw * scale_factor))
    new_fh = max(10, int(fh * scale_factor))
    fruit = fruit.resize((new_fw, new_fh), resample=Image.LANCZOS)

    max_jitter_x = max(0, (tw - new_fw) // 5)
    max_jitter_y = max(0, (th - new_fh) // 5)
    pos_x = (tw - new_fw) // 2 + (random.randint(-max_jitter_x, max_jitter_x) if max_jitter_x > 0 else 0)
    pos_y = (th - new_fh) // 2 + (random.randint(-max_jitter_y, max_jitter_y) if max_jitter_y > 0 else 0)

    if not force_white_bg and bg_func != make_pure_white_background:
        shadow_mask = fruit.split()[3]
        shadow = Image.new("RGBA", (new_fw, new_fh), (0, 0, 0, 90))
        shadow.putalpha(shadow_mask)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=5))
        shadow_x = pos_x + random.randint(-2, 2)
        shadow_y = pos_y + random.randint(3, 7)
        bg.paste(shadow, (shadow_x, shadow_y), shadow)

    bg.paste(fruit, (pos_x, pos_y), fruit)
    result = bg.convert("RGB")

    # Realistic camera brightness & contrast variations
    brightness_factor = random.uniform(0.78, 1.18)
    result = ImageEnhance.Brightness(result).enhance(brightness_factor)

    contrast_factor = random.uniform(0.88, 1.14)
    result = ImageEnhance.Contrast(result).enhance(contrast_factor)

    if random.random() > 0.45:
        r_mult = random.uniform(0.95, 1.05)
        b_mult = random.uniform(0.95, 1.05)
        arr = np.array(result, dtype=float)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * r_mult, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * b_mult, 0, 255)
        result = Image.fromarray(arr.astype(np.uint8))

    if random.random() > 0.55:
        noise = np.random.normal(0, random.uniform(1.5, 4.0), (th, tw, 3))
        res_arr = np.clip(np.array(result, dtype=float) + noise, 0, 255).astype(np.uint8)
        result = Image.fromarray(res_arr)

    return result


def load_base_subjects():
    # Clean, verified transparent cutouts covering ALL 8 canonical angles + test variants
    raw_sources = {
        "unripe": [
            "assets/images/papaya_unripe.png",
            "scratch_test_unripe.png",
            "scratch_test_unripe_perfect.png",
            "assets/images/papaya_unripe_orig.png",
            "assets/images/papaya_unripe_white.png",
        ] + sorted(glob.glob("assets/images/angles/unripe_*.png")),
        "ripening": [
            "assets/images/papaya_ripening.png",
            "scratch_test_ripening.png",
            "scratch_test_ripening_perfect.png",
            "scratch_ripening_clean.png",
        ] + sorted(glob.glob("assets/images/angles/ripening_*.png")),
        "ripe": [
            "assets/images/papaya_ripe.png",
            "scratch_test_ripe.png",
            "scratch_test_ripe_perfect.png",
        ] + sorted(glob.glob("assets/images/angles/ripe_*.png")),
        "overripe": [
            "assets/images/papaya_overripe.png",
        ] + sorted(glob.glob("assets/images/angles/overripe_*.png")),
    }

    classes_subjects = {}
    for c, paths in raw_sources.items():
        classes_subjects[c] = []
        for p in paths:
            full_p = os.path.join(BASE_DIR, p)
            if os.path.exists(full_p):
                try:
                    img = Image.open(full_p).convert("RGBA")
                    masked = extract_fruit_mask(img)
                    classes_subjects[c].append(masked)
                except Exception as e:
                    print(f"Skipping {p}: {e}")
        print(f"Loaded {len(classes_subjects[c])} verified multi-angle base subjects for {c}")
    return classes_subjects


def generate_curated_dataset(train_per_class=400, val_per_class=80, test_per_class=80):
    print("\n--- Starting Multi-Angle & Multi-Environment Papaya Dataset Generation ---")
    classes_subjects = load_base_subjects()

    splits = {
        "train": train_per_class,
        "val": val_per_class,
        "test": test_per_class,
    }

    for split in splits.keys():
        for c in classes_subjects.keys():
            split_dir = os.path.join(DATASET_ROOT, split, c)
            os.makedirs(split_dir, exist_ok=True)
            for old_f in glob.glob(os.path.join(split_dir, "*.jpg")):
                os.remove(old_f)

    # All splits sample across all 8 camera angles with distinct random seeds/backgrounds/lighting
    for c, subjects in classes_subjects.items():
        print(f"\nClass '{c}' ({len(subjects)} angle/variant sources):")
        for split_name, count in splits.items():
            print(f"   Generating {count} {split_name} images...")
            for i in range(count):
                subj = subjects[i % len(subjects)]
                force_white = (i % 4 == 0)  # 25% pure white, 75% diverse realistic backgrounds
                rot_r = (-45, 45) if split_name == "train" else (-25, 25)
                scale_r = (0.58, 0.90) if split_name == "train" else (0.64, 0.86)
                img = composite_fruit(
                    subj,
                    TARGET_SIZE,
                    scale_range=scale_r,
                    rot_range=rot_r,
                    force_white_bg=force_white,
                )
                save_path = os.path.join(DATASET_ROOT, split_name, c, f"{c}_{split_name}_{i+1:04d}.jpg")
                img.save(save_path, "JPEG", quality=93)

    total_images = sum(splits.values()) * len(classes_subjects)
    print(f"\nSuccessfully generated {total_images} balanced images in {DATASET_ROOT}")


if __name__ == "__main__":
    generate_curated_dataset()
