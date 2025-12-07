# Segmentation of White Blood Cells (Unstained) - 10 min Deck

Each section is ~1 slide. Speaker cues under **Notes**.

## 1) Problem & Motivation
- Goal: segment nucleus and boundary/cytoplasm for WBCs in unstained 128x128 smears.
- Clinical value: fast, label-free morphology for counts, screening, virtual staining.
- Challenge: low contrast, subtle scattering, small FOV; need robust masks.
- Deliverables: code, metrics (Dice/IoU), qualitative overlays, short report/presentation.
**Notes:** Emphasize unstained difficulty and why segmentation quality matters downstream.

## 2) Dataset
- Source: provided Google Drive; JSON polygons per image (labels `n`, `b`).
- Counts after cleaning: train 370, val 65, test 70 (seed 1337, 15% val, overlay images removed).
- Class mapping: 0 background, 1 nucleus, 2 boundary/cytoplasm.
- Images centered on WBC; occasional extra cells.
**Notes:** Mention JSON+PNG pairing and small sample size -> need augmentation.

## 3) Preprocessing
- JSON -> mask PNG via `prepare_masks.py` (fills polygons to class IDs).
- Normalization: mean/std 0.5; no resize needed (native 128x128).
- Augmentations (train): H/V flip, rotate 15 deg, brightness/contrast, Gaussian noise.
- Val/test: normalization only.
**Notes:** Show a colorized mask overlay; mention why label IDs look black by default.

## 4) Data Layout
- `data/{train,val,test}/images/*.png`
- `data/{train,val,test}/masks/*.png` (or `annotations/*.json` if you prefer on-the-fly)
- Config points to these paths in `experiments/baseline.yaml`.
**Notes:** Quick diagram of folders; stress reproducible split.

## 5) Models Evaluated
- UNet (encoder-decoder, skip connections), ~few million params.
- DeepLabV3-ResNet50 option (with/without pretrained backbone).
- Input: 3-channel RGB; Output: 3-class logits.
**Notes:** Justify UNet for small images; DeepLab for boundary refinement.

## 6) Loss, Metrics, and Training Loop
- Loss: CrossEntropy + Dice (weight 0.5 default); optional class weights.
- Metrics: mean Dice and IoU over 3 classes; per-class breakdown recommended.
- Training: AdamW, LR 1e-3, wd 1e-4, epochs 50, batch 16, AMP on, grad clip 1.0.
- Scheduler: ReduceLROnPlateau on val Dice.
**Notes:** Explain why Dice helps thin boundaries; AMP for speed.

## 7) Baseline Results (fill after runs)
- Report table: val Dice/IoU (mean + per class).
- Show 2-3 overlays: success (clean boundary) and failure (missed nucleus/leakage).
**Notes:** Keep concise; highlight acceptance thresholds if met (e.g., Dice >=0.85 nucleus, >=0.80 boundary).

## 8) Experiment Matrix (to satisfy “thorough experimentation”)
- UNet baseline (no aug) vs UNet + aug.
- Loss ablation: CE-only vs CE+Dice vs Dice-heavy (0.7).
- DeepLabV3 variants: scratch vs pretrained backbone.
- Class-weight sweep if nucleus/boundary under-segmented.
- Inference TTA (flip averaging) trade-off.
- LR/batch sweep: LR {5e-4, 1e-3, 2e-3}, batch {8,16}.
**Notes:** Show best config highlighted; brief metric deltas.

## 9) Error Analysis
- Common errors: boundary leakage into background; missed faint nuclei; occasional false positives on debris.
- Correlate with: low contrast frames, off-center cells, rare shapes.
- Mitigations: stronger aug on contrast, class weighting, CRF/blur post-process (future).
**Notes:** Pair each error with a visual example.

## 10) Conclusion & Next Steps
- Current best (fill in) meets/near targets on val; test pending/ready.
- Next: run on test, add qualitative panel to report, optional CRF post-proc, try lighter backbones for speed.
- Repo quickstart: `pip install -r requirements.txt`; train `python train.py --config experiments/baseline.yaml`; eval `python evaluate.py --checkpoint outputs/baseline/checkpoints/best.pt --split val --save-dir outputs/predictions`.
**Notes:** End with asks: accept thresholds, approve final model, or request more ablations.
