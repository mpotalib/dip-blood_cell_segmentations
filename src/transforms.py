import albumentations as A


def get_train_transforms(config) -> A.Compose:
    aug_cfg = config.get("augmentation", {}).get("train", {})
    transforms = []
    if aug_cfg.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=0.5))
    if aug_cfg.get("vertical_flip", False):
        transforms.append(A.VerticalFlip(p=0.5))
    rotation = aug_cfg.get("rotation", 0)
    if rotation:
        transforms.append(A.Rotate(limit=rotation, border_mode=0, p=0.5))
    if aug_cfg.get("brightness_contrast", 0):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg["brightness_contrast"],
                contrast_limit=aug_cfg["brightness_contrast"],
                p=0.5,
            )
        )
    if aug_cfg.get("gaussian_noise", 0):
        noise = aug_cfg["gaussian_noise"]
        std_range = (0, noise) if isinstance(noise, (int, float)) else noise
        transforms.append(A.GaussNoise(std_range=std_range, p=0.3))
    transforms.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0))
    return A.Compose(transforms)


def get_val_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
        ]
    )
