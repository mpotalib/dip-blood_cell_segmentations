## White Blood Cell Segmentation (Unstained Smears)

This repository scaffolds the full code path to segment WBC nuclei and cytoplasm/boundary from 128×128 unstained blood smear images, matching the assignment brief and teacher request for a detailed, experiment-heavy presentation.

### 1) Setup
- Install deps: `pip install -r requirements.txt`
- Dataset (images + JSON annotations): https://drive.google.com/drive/folders/1WFXZUG1jVvvtCiEmHuRjLRb7ir3zGJ26?usp=sharing  
  Place under `data/raw/` or directly into `data/train|val|test/{images,annotations}` (see below).
- Hardware: CUDA GPU recommended; AMP is enabled by default.

### 2) Data layout
```
data/
  train/
    images/*.png
    masks/*.png            # optional if you pre-generate
    annotations/*.json     # polygon labels 'n' (nucleus) and 'b' (boundary/cytoplasm)
  val/
    images/
    masks/ or annotations/
  test/
    images/
    masks/ or annotations/
```
If you only have JSON, create masks:  
`python prepare_masks.py --images-dir data/train/images --annotations-dir data/train/annotations --output-dir data/train/masks`

Current split (seed=1337, 15% val, overlay images excluded): train 370, val 65, test 70.

### 3) Core training
- Edit `experiments/baseline.yaml` for your paths/params.
- Train: `python train.py --config experiments/baseline.yaml`
- Checkpoints land in `outputs/baseline/checkpoints`; `best.pt` mirrors the highest validation Dice.

### 4) Evaluation and visuals
- Evaluate on val/test:  
  `python evaluate.py --config experiments/baseline.yaml --checkpoint outputs/baseline/checkpoints/best.pt --split val --save-dir outputs/predictions --limit 12`
- The script writes colorized predicted and GT masks to the save directory for slide/report figures. Dice and IoU are printed.

### 5) Baseline model/criterion
- Models: `unet` (default) or `deeplab` (`torchvision` DeepLabV3-ResNet50 head adapted to 3 classes).
- Loss: CrossEntropy + Dice (weight set via `dice_weight` in the config). Optional class weights mitigate imbalance.
- Metrics: mean Dice and IoU across 3 classes (background, nucleus, boundary).

### 6) Experiment plan (to satisfy “thorough experimentation”)
Run and report at least these, keeping a fixed seed/splits:
1. Baseline UNet (no extra aug); record Dice/IoU and qualitative masks.
2. UNet + augmentations (flip/rot/brightness/noise) — compare gains.
3. Loss ablation: CE-only vs CE+Dice (current) vs Dice-heavy (e.g., `dice_weight=0.7`).
4. DeepLabV3+ (ResNet50 backbone, with/without pretrained) — compare boundary quality vs UNet.
5. Class-weight sweep if nuclei/boundaries are under-segmented (e.g., weights [1.0, 1.3, 1.3]).
6. TTA at inference (horizontal/vertical flips averaged) — note marginal improvements vs cost.
7. Hyperparameter quick sweep: batch size (8/16), LR (5e-4, 1e-3, 2e-3); pick the best combo.

Document for each experiment: config deltas, curves (loss/Dice), final metrics per split/class, and 2–3 qualitative overlays highlighting successes/failures.

### 7) Reporting / presentation checklist
- Cover dataset description, preprocessing (mask generation, normalization, augmentation), model choices, and training strategy.
- Include metric tables (Dice/IoU per class + mean) and qualitative figures (pred vs GT).
- State acceptance thresholds you achieved (suggest >=0.85 Dice nucleus / >=0.80 Dice boundary on val) and note remaining errors.
- Add an appendix slide with future work (e.g., CRF post-processing, small backbones for speed).

### 9) Kaggle GPU notebook
- Ready-to-run notebook: `notebooks/kaggle_wbc.ipynb`.
- Steps on Kaggle: enable GPU, attach/upload a dataset containing `data/{train,val,test}/{images,masks}` (or `annotations`), update `DATASET_BASE` in the notebook to your dataset path, run cells to install deps, link data, train, and evaluate. Checkpoints and qualitative masks are saved under `outputs/`.

### 8) Files of interest
- `train.py` — training loop with AMP, ReduceLROnPlateau scheduler, checkpointing.
- `evaluate.py` — metrics + colored mask exports.
- `src/dataset.py` — image/mask loader; JSON→mask conversion for labelme-style polygons.
- `src/models.py` — UNet and DeepLab builders.
- `src/losses.py` — CE + Dice loss.
- `src/transforms.py` — training/val augmentation pipelines.
- `prepare_masks.py` — standalone JSON-to-mask generator.
