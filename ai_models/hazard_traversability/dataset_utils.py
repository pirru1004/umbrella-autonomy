from datasets import load_dataset
import numpy as np
from PIL import Image


def mask_to_hazard_label(mask_array):
    """
    Convert a segmentation mask into a single patch-level hazard label.
    Simple heuristic baseline:
    0 = safe
    1 = moderate
    2 = hazardous
    """

    unique, counts = np.unique(mask_array, return_counts=True)
    label_dist = dict(zip(unique.tolist(), counts.tolist()))

    total = mask_array.size
    proportions = {k: v / total for k, v in label_dist.items()}

    # Placeholder assumptions for grouped hazard classes
    sand_like = proportions.get(2, 0.0)
    rock_like = proportions.get(3, 0.0)
    soil_like = proportions.get(1, 0.0)
    bedrock_like = proportions.get(0, 0.0)

    if sand_like + rock_like > 0.45:
        return 2  # hazardous
    elif soil_like + sand_like > 0.35:
        return 1  # moderate
    else:
        return 0  # safe


def preprocess_example(example, image_size=(128, 128)):
    image = example["image"]
    mask = example["label_mask"]

    if mask is None:
        return None, None

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))

    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(np.array(mask).astype(np.uint8))

    image = image.resize(image_size)
    mask = mask.resize(image_size, resample=Image.NEAREST)

    image_np = np.array(image).astype(np.float32) / 255.0
    mask_np = np.array(mask)

    label = mask_to_hazard_label(mask_np)

    return image_np, label


def load_ai4mars_subset(split="train", num_samples=200):
    ds = load_dataset("hassanjbara/AI4MARS", split=split)

    # Keep only samples with labels
    ds = ds.filter(lambda ex: ex["has_labels"] is True and ex["label_mask"] is not None)

    ds = ds.select(range(min(num_samples, len(ds))))
    return ds


def prepare_numpy_dataset(split="train", num_samples=200, image_size=(128, 128)):
    ds = load_ai4mars_subset(split=split, num_samples=num_samples)

    images = []
    labels = []

    for ex in ds:
        img, lbl = preprocess_example(ex, image_size=image_size)

        if img is None or lbl is None:
            continue

        images.append(img)
        labels.append(lbl)

    X = np.stack(images)
    y = np.array(labels)

    return X, y
