import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import WBCDataset
from src.models import build_model
from src.transforms import get_val_transforms
from src.utils import dice_coefficient, iou_score, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint.")
    parser.add_argument("--config", type=str, default="experiments/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--save-dir", type=str, default="outputs/predictions")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images to save (0 = all).")
    return parser.parse_args()


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _split_dirs(data_cfg, split: str):
    images = Path(data_cfg[f"{split}_images"])
    masks = data_cfg.get(f"{split}_masks")
    annotations = data_cfg.get(f"{split}_annotations")
    mask_dir = Path(masks) if masks else None
    ann_dir = Path(annotations) if annotations else None
    return images, mask_dir, ann_dir


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    colors = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0]], dtype=np.uint8)
    mask = np.clip(mask, 0, len(colors) - 1)
    return colors[mask]


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    set_seed(cfg["logging"].get("seed", 1337))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_mapping = cfg["data"].get("class_mapping", {"background": 0, "n": 1, "b": 2})
    num_classes = len(class_mapping)
    images, masks, annotations = _split_dirs(cfg["data"], args.split)
    dataset = WBCDataset(images, mask_dir=masks, annotation_dir=annotations, transforms=get_val_transforms(), class_mapping=class_mapping)
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"].get("num_workers", 4), pin_memory=True)

    model = build_model(
        name=cfg["model"].get("name", "unet"),
        in_channels=cfg["model"].get("in_channels", 3),
        num_classes=cfg["model"].get("num_classes", num_classes),
        pretrained_backbone=False,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dices, ious = [], []
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx, (images_t, masks_t) in enumerate(loader):
        images_t = images_t.to(device)
        masks_t = masks_t.to(device)
        outputs = model(images_t)
        logits = outputs["out"] if isinstance(outputs, dict) else outputs
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

        dice = dice_coefficient(preds, masks_t, num_classes)
        iou = iou_score(preds, masks_t, num_classes)
        dices.append(dice)
        ious.append(iou)

        if args.limit == 0 or saved < args.limit:
            for b in range(images_t.size(0)):
                if args.limit and saved >= args.limit:
                    break
                pred_mask = preds[b].cpu().numpy().astype(np.uint8)
                gt_mask = masks_t[b].cpu().numpy().astype(np.uint8)
                cv2.imwrite(str(save_dir / f"{idx:04d}_{b}_pred.png"), colorize_mask(pred_mask))
                cv2.imwrite(str(save_dir / f"{idx:04d}_{b}_gt.png"), colorize_mask(gt_mask))
                saved += 1

    mean_dice = float(np.mean(dices))
    mean_iou = float(np.mean(ious))
    print(f"Split: {args.split} | Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f}")


if __name__ == "__main__":
    main()
