### Slide Outline (detailed technical deck)

1) Problem & motivation  
- Unstained WBC segmentation, clinical value, constraints (low contrast, small FOV).

2) Dataset  
- Source, split strategy, class definitions (background/nucleus/boundary).  
- Annotation → mask pipeline (polygon fill), example frames.

3) Preprocessing  
- Mask generation command, normalization, resizing (128×128), data augmentations.

4) Models evaluated  
- UNet (channels/features), DeepLabV3-ResNet50 (pretrained vs scratch).  
- Params/FLOPs table.

5) Loss & metrics  
- CE + Dice (why), class weighting option.  
- Metrics: mean Dice/IoU + per-class breakdown.

6) Training protocol  
- Hyperparams (batch, LR, wd, epochs, scheduler, AMP, grad clipping), seed policy.

7) Experiments (minimum set)  
- Baseline UNet no aug.  
- UNet + aug.  
- Loss ablation (CE vs CE+Dice vs Dice-heavy).  
- DeepLabV3(+pretrained) comparison.  
- Class-weight sweep.  
- TTA inference.  
- LR/batch sweep.

8) Results  
- Tables: metrics by split/class; best config highlighted.  
- Qualitative: overlays pred vs GT (success/failure cases).

9) Discussion  
- Error modes (boundary leakage, missed nuclei), augment impact, trade-offs.  
- Why chosen model wins; failure analysis vs literature baselines.

10) Conclusion & next steps  
- Acceptance thresholds achieved, risks, future work (CRF, lightweight backbones, semi-supervised).
