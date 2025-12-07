import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.dataset import WBCDataset
from src.losses import CombinedLoss
from src.models import build_model
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import (
    AverageMeter,
    count_parameters,
    dice_coefficient,
    iou_score,
    save_checkpoint,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train WBC segmentation models.")
    parser.add_argument("--config", type=str, default="experiments/baseline.yaml", help="Path to config yaml.")
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


def create_dataloaders(cfg):
    data_cfg = cfg["data"]
    aug = cfg.get("augmentation", {})
    class_mapping = data_cfg.get("class_mapping", {"background": 0, "n": 1, "b": 2})

    train_images, train_masks, train_annotations = _split_dirs(data_cfg, "train")
    val_images, val_masks, val_annotations = _split_dirs(data_cfg, "val")

    train_ds = WBCDataset(
        train_images,
        mask_dir=train_masks,
        annotation_dir=train_annotations,
        transforms=get_train_transforms(cfg),
        class_mapping=class_mapping,
    )
    val_ds = WBCDataset(
        val_images,
        mask_dir=val_masks,
        annotation_dir=val_annotations,
        transforms=get_val_transforms(),
        class_mapping=class_mapping,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
    )
    return train_loader, val_loader, len(class_mapping)


def get_logits(output):
    if isinstance(output, dict) and "out" in output:
        return output["out"]
    return output


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg):
    model.train()
    losses = AverageMeter()
    log_every = cfg["logging"].get("log_every", 10)
    use_amp = cfg["train"].get("use_amp", False)
    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            outputs = model(images)
            logits = get_logits(outputs)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        if cfg["train"].get("grad_clip"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.item(), images.size(0))
        if (i + 1) % log_every == 0:
            print(f"Step [{i+1}/{len(loader)}] - loss: {losses.avg:.4f}")
    return losses.avg


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, cfg):
    model.eval()
    losses = AverageMeter()
    dices = AverageMeter()
    ious = AverageMeter()
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        logits = get_logits(outputs)
        loss = criterion(logits, masks)
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        dice = dice_coefficient(preds, masks, num_classes)
        iou = iou_score(preds, masks, num_classes)
        losses.update(loss.item(), images.size(0))
        dices.update(dice, images.size(0))
        ious.update(iou, images.size(0))
    print(f"Validation - loss: {losses.avg:.4f} | dice: {dices.avg:.4f} | iou: {ious.avg:.4f}")
    return losses.avg, dices.avg, ious.avg


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    set_seed(cfg["logging"].get("seed", 1337))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["train"].get("amp", True) and torch.cuda.is_available()
    cfg["train"]["use_amp"] = use_amp
    train_loader, val_loader, num_classes = create_dataloaders(cfg)

    model = build_model(
        name=cfg["model"].get("name", "unet"),
        in_channels=cfg["model"].get("in_channels", 3),
        num_classes=cfg["model"].get("num_classes", num_classes),
        pretrained_backbone=cfg["model"].get("pretrained_backbone", False),
    ).to(device)

    print(f"Model params: {count_parameters(model)/1e6:.2f}M")

    class_weights = cfg["train"].get("class_weights")
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None
    criterion = CombinedLoss(num_classes=num_classes, ce_weight=weight_tensor, dice_weight=cfg["train"].get("dice_weight", 0.5))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"].get("lr", 1e-3),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=cfg["train"].get("lr_patience", 3),
    )
    scaler = GradScaler(enabled=use_amp)

    epochs = cfg["train"]["epochs"]
    val_every = cfg["logging"].get("val_every", 1)
    ckpt_dir = Path(cfg["logging"].get("checkpoint_dir", "outputs/checkpoints"))
    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg)
        elapsed = time.time() - start
        print(f"Epoch {epoch} done in {elapsed/60:.1f} min - train loss {train_loss:.4f}")

        if epoch % val_every == 0:
            val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device, num_classes, cfg)
            scheduler.step(val_dice)
            is_best = val_dice > best_dice
            best_dice = max(best_dice, val_dice)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_dice": best_dice,
                    "config": cfg,
                },
                ckpt_dir / f"epoch_{epoch}.pt",
                is_best=is_best,
            )


if __name__ == "__main__":
    main()
