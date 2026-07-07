import random
import shutil
from pathlib import Path

# ===========================
# 설정
# ===========================

SOURCE_DATASET = Path(r"F:\Datasets\Intel\seg_train")
OUTPUT_DATASET = Path(r"F:\Datasets\demo_dataset")

IMAGES_PER_CLASS = 50

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

random.seed(42)

# ===========================

OUTPUT_DATASET.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("Creating Demo Dataset")
print("=" * 50)

for class_dir in SOURCE_DATASET.iterdir():

    if not class_dir.is_dir():
        continue

    images = []

    for file in class_dir.iterdir():

        if file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file)

    print(f"{class_dir.name}: {len(images)} images found")

    if len(images) == 0:
        continue

    sample_count = min(IMAGES_PER_CLASS, len(images))

    sampled = random.sample(images, sample_count)

    out_class = OUTPUT_DATASET / class_dir.name
    out_class.mkdir(exist_ok=True)

    for img in sampled:
        shutil.copy2(img, out_class / img.name)

    print(f"Copied {sample_count} images")

print()
print("Done!")
print("Demo dataset saved to:")
print(OUTPUT_DATASET)