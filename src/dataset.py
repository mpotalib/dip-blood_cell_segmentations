import json
from pathlib import Path
from typing import Dict, List, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_mask_from_json(
    annotation_path: Path,
    shape: List[int],
    class_mapping: Dict[str, int],
) -> np.ndarray:
    """Converts a polygon-based JSON annotation into a dense mask."""
    with annotation_path.open("r") as f:
        data = json.load(f)
    mask = np.zeros(shape, dtype=np.uint8)
    shapes = data.get("shapes", [])
    for shape_def in shapes:
        label = shape_def.get("label")
        points = np.array(shape_def.get("points", []), dtype=np.int32)
        if label is None or len(points) == 0:
            continue
        class_id = class_mapping.get(label, 0)
        cv2.fillPoly(mask, [points], class_id)
    return mask


class WBCDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Optional[Path] = None,
        annotation_dir: Optional[Path] = None,
        transforms: Optional[A.Compose] = None,
        class_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        assert mask_dir or annotation_dir, "Either mask_dir or annotation_dir must be provided."
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.annotation_dir = Path(annotation_dir) if annotation_dir else None
        self.transforms = transforms
        self.class_mapping = class_mapping or {"background": 0, "n": 1, "b": 2}

        self.images = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        if self.mask_dir:
            mask_path = self.mask_dir / (image_path.stem + ".png")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found for {image_path.name}")
        else:
            annotation_path = self.annotation_dir / (image_path.stem + ".json")
            if not annotation_path.exists():
                raise FileNotFoundError(f"Annotation not found for {image_path.name}")
            mask = _load_mask_from_json(annotation_path, [h, w], self.class_mapping)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32)
        if self.transforms is None:
            image = image / 255.0
        mask = mask.astype(np.int64)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        return image, mask
