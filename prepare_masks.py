import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert JSON polygon annotations to mask images.")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory with images.")
    parser.add_argument("--annotations-dir", type=str, required=True, help="Directory with JSON files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write mask PNGs.")
    parser.add_argument(
        "--class-map",
        type=str,
        default='{\"n\":1, \"b\":2}',
        help="JSON string mapping label names to integer ids (background is 0).",
    )
    return parser.parse_args()


def load_class_map(raw: str):
    return json.loads(raw)


def convert(images_dir: Path, annotations_dir: Path, output_dir: Path, class_map):
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    for img_path in images:
        ann_path = annotations_dir / f"{img_path.stem}.json"
        if not ann_path.exists():
            print(f"Skipping {img_path.name}: annotation not found")
            continue
        with ann_path.open("r") as f:
            data = json.load(f)
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for shape in data.get("shapes", []):
            label = shape.get("label")
            pts = np.array(shape.get("points", []), dtype=np.int32)
            if label is None or len(pts) == 0:
                continue
            class_id = class_map.get(label, 0)
            cv2.fillPoly(mask, [pts], class_id)
        out_path = output_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), mask)
    print(f"Saved masks to {output_dir}")


def main():
    args = parse_args()
    class_map = load_class_map(args.class_map)
    convert(Path(args.images_dir), Path(args.annotations_dir), Path(args.output_dir), class_map)


if __name__ == "__main__":
    main()
