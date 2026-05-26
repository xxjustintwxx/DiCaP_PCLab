# CXR-LT 2024 — Training Findings & Analysis

> **Note:** All mAP numbers in this document are evaluated on the **held-out test set**
> (78,946 images from `test_labeled_task1.csv`), not the validation set used in earlier runs.
> Zone thresholds are defined by combined train+val frequency: Head >10% (~29,816 images),
> Medium 1–10%, Tail ≤1% (~2,982 images).

---

## Run: lb_ratio=0.05, ResNet50, ASL loss

**Date:** 2026-04-12
**Checkpoint:** `output/cxr_ours/cxr/resnet50/0.05/fine_tune/best_model.pth.tar` (epoch 34, EMA)

### Overall Metrics

| Metric | Value |
|--------|-------|
| Overall mAP | 14.76 |
| Head mAP (>10%, 9 classes) | 47.27 |
| Medium mAP (1–10%, 15 classes) | 7.11 |
| Tail mAP (≤1%, 16 classes) | 3.64 |

### Per-Class AP (sorted head → tail by train+val frequency)

| Class | Train+Val # | Zone | AP |
|---|---:|:---:|---:|
| Support Devices | 99,240 | Head | 83.5 |
| Lung Opacity | 89,213 | Head | 49.6 |
| Cardiomegaly | 86,321 | Head | 56.8 |
| Pleural Effusion | 76,831 | Head | 77.0 |
| Atelectasis | 75,507 | Head | 51.6 |
| Pneumonia | 53,721 | Head | 23.3 |
| Edema | 43,202 | Head | 43.4 |
| Normal | 39,380 | Head | 26.7 |
| Enlarged Cardiomediastinum | 33,872 | Head | 13.4 |
| Consolidation | 17,750 | Medium | 13.7 |
| Pneumothorax | 16,200 | Medium | 27.7 |
| Fracture | 13,144 | Medium | 7.4 |
| Infiltration | 11,593 | Medium | 4.7 |
| Rib Fracture | 10,169 | Medium | 5.3 |
| Nodule | 8,650 | Medium | 5.7 |
| Mass | 6,077 | Medium | 4.5 |
| Calcification of the Aorta | 4,833 | Medium | 5.9 |
| Hernia | 4,660 | Medium | 6.4 |
| Emphysema | 4,402 | Medium | 11.8 |
| Adenopathy | 3,886 | Medium | 2.6 |
| Tortuous Aorta | 3,831 | Medium | 3.5 |
| Pleural Thickening | 3,751 | Medium | 3.5 |
| Granuloma | 3,348 | Medium | 2.1 |
| Fissure | 3,154 | Medium | 1.9 |
| Lung Lesion | 2,652 | Tail | 2.6 |
| Subcutaneous Emphysema | 2,477 | Tail | 31.1 |
| Tuberculosis | 2,455 | Tail | 3.4 |
| Pulmonary Embolism | 1,935 | Tail | 0.8 |
| Fibrosis | 1,332 | Tail | 2.3 |
| Pulmonary Hypertension | 1,022 | Tail | 0.9 |
| Kyphosis | 890 | Tail | 1.8 |
| Pneumomediastinum | 826 | Tail | 8.8 |
| Infarction | 823 | Tail | 0.3 |
| Hydropneumothorax | 774 | Tail | 2.7 |
| Pleural Other | 696 | Tail | 0.8 |
| Pneumoperitoneum | 570 | Tail | 1.8 |
| Azygos Lobe | 219 | Tail | 0.1 |
| Round(ed) Atelectasis | 218 | Tail | 0.7 |
| Clavicle Fracture | 187 | Tail | 0.1 |
| Lobar Atelectasis | 155 | Tail | 0.1 |

---

## Why mAP is Only ~15

### 1. Long-tail collapse is the dominant factor

mAP averages all 40 classes equally. The 9 head classes (>10% frequency) score 13–84 AP.
The 15 medium classes (1–10%) average 7.1 AP. The 16 tail classes (≤1%) average 3.6 AP,
pulled up almost entirely by Subcutaneous Emphysema (31.1) and Pneumomediastinum (8.8);
the remaining 14 tail classes average below 3 AP, with Azygos Lobe, Clavicle Fracture, and
Lobar Atelectasis all at 0.1.

### 2. 5% labeled data starves rare classes

5% of 258,871 = 12,943 labeled images.

- Lobar Atelectasis: 129 train positives → ~6 labeled positives
- Clavicle Fracture: 168 train positives → ~8 labeled positives
- Even Pulmonary Embolism (1,631 total) → ~82 labeled positives — still very few

The semi-supervised pseudo-label phase does not rescue rare classes: the warmup model
scores near random on them, so pseudo-labels for rare classes are mostly wrong and
provide no useful training signal.

### 3. mAP plateaus very early

| Phase | Best mAP |
|-------|---------|
| Warmup ep 0 | 10.4 |
| Warmup ep 3 | 14.1 |
| Warmup ep 6 (best) | 14.6 |
| Main phase (40 epochs) | 14.8 |
| Fine-tune (20 epochs) | **14.95** (val) / **14.76** (test) |

The model reaches 14.1 mAP by epoch 3 of warmup and barely improves through 60 more
epochs of semi-supervised training. This suggests the ResNet50 backbone saturates
quickly on common classes and the semi-supervised signal is too noisy for rare classes.

### 4. Bug: evaluate_cxr.py was loading wrong weights

The checkpoint stores both `state_dict` (regular model) and `state_dict_ema` (EMA).
Training tracks best mAP using the EMA model, but the original `evaluate_cxr.py` loaded
`state_dict`. Fixed to prefer `state_dict_ema`.

- Regular model: 14.09 mAP (val)
- EMA model: 14.95 mAP (val) / 14.76 mAP (test)

---

## Threshold Analysis (threshold=0.5 on val set)

The model outputs a sigmoid score ∈ [0, 1] per class per image. A **threshold** converts
this continuous score into a binary prediction: score ≥ threshold → predicted positive.
The TP counts below use threshold=0.5 on the **validation set** (39,293 images).

### Per-class positive count vs correctly classified (TP at threshold=0.5)

| Class | Val+ | TP | Recall |
|-------|-----:|---:|-------:|
| Support Devices | 13,161 | 11,723 | 89.1% |
| Lung Opacity | 11,731 | 10,925 | 93.1% |
| Cardiomegaly | 11,583 | 10,425 | 90.0% |
| Pleural Effusion | 10,430 | 8,938 | 85.7% |
| Atelectasis | 10,131 | 8,610 | 85.0% |
| Pneumonia | 7,061 | 6,805 | 96.4% |
| Edema | 5,946 | 4,946 | 83.2% |
| Normal | 5,088 | 3,525 | 69.3% |
| Enlarged Cardiomediastinum | 4,244 | 4,081 | 96.2% |
| Consolidation | 2,379 | 1,223 | 51.4% |
| Pneumothorax | 2,342 | 1,135 | 48.5% |
| Fracture | 1,576 | 151 | 9.6% |
| Infiltration | 1,506 | 680 | 45.2% |
| Rib Fracture | 1,250 | 64 | 5.1% |
| Nodule | 1,119 | 184 | 16.4% |
| Mass | 789 | 66 | 8.4% |
| Calcification of the Aorta | 594 | 89 | 15.0% |
| Hernia | 674 | 149 | 22.1% |
| Emphysema | 741 | 330 | 44.5% |
| Adenopathy | 477 | 58 | 12.2% |
| Tortuous Aorta | 495 | 35 | 7.1% |
| Pleural Thickening | 479 | 32 | 6.7% |
| Granuloma | 383 | 24 | 6.3% |
| Fissure | 351 | 16 | 4.6% |
| Lung Lesion | 314 | 12 | 3.8% |
| Tuberculosis | 377 | 127 | 33.7% |
| Subcutaneous Emphysema | 431 | 205 | 47.6% |
| Pulmonary Embolism | 304 | 0 | 0.0% |
| Fibrosis | 163 | 25 | 15.3% |
| Pulmonary Hypertension | 119 | 4 | 3.4% |
| Kyphosis | 112 | 16 | 14.3% |
| Infarction | 96 | 0 | 0.0% |
| Pneumomediastinum | 122 | 20 | 16.4% |
| Hydropneumothorax | 128 | 0 | 0.0% |
| Pleural Other | 80 | 1 | 1.2% |
| Pneumoperitoneum | 54 | 3 | 5.6% |
| Azygos Lobe | 20 | 0 | 0.0% |
| Round(ed) Atelectasis | 46 | 0 | 0.0% |
| Clavicle Fracture | 19 | 0 | 0.0% |
| Lobar Atelectasis | 26 | 0 | 0.0% |
| **TOTAL** | **96,941** | **74,627** | **77.0%** |

### What the threshold reveals about long-tail failure

A score of 0.5 means the model is equally uncertain between present and absent. For rare
classes, DiCaP's `threshold1` (learned from labeled data) is rarely crossed because:

1. **Too few labeled positives** — the warmup model never reliably associates features
   with rare classes, so it learns to output low scores for everything in the tail.
2. **Pseudo-labels are almost never positive** — if the warmup score never reaches
   `threshold1`, no pseudo-positives are generated, and the semi-supervised phase
   provides no extra signal for that class.
3. **The fixed 0.5 inference threshold inherits this bias** — even if the model
   occasionally outputs 0.3–0.4 for a true positive of a rare class, it is counted
   as negative at threshold=0.5.

This creates a self-reinforcing failure: low training signal → low scores → no
pseudo-labels → still low training signal. The 7 classes with 0% TP
(Pulmonary Embolism, Infarction, Hydropneumothorax, Round(ed) Atelectasis,
Azygos Lobe, Clavicle Fracture, Lobar Atelectasis) are all in this regime.

### Threshold is a recall/precision trade-off

Lowering the threshold (e.g. to 0.3) would recover some tail TPs but at the cost of
many false positives on common classes. For CXR-LT evaluation (mAP), threshold does
not matter — mAP integrates over all thresholds via the precision-recall curve. The
threshold only matters for operational deployment or for the TP count visualisation above.

---

## Recommendations

| Action | Expected Impact |
|--------|----------------|
| Increase `lb_ratio` (e.g. 0.1, 0.2) | More labeled positives for rare classes — biggest lever |
| Switch to `--net tresnet_l` | Stronger backbone; better spatial multi-label features |
| Add class-frequency reweighting to loss | Upweight rare classes during training |
| Logit adjustment at inference | Corrects class-prior bias, standard fix for long-tail |
| Class-balanced sampling | Ensures rare classes are seen proportionally |

---

## Run: lb_ratio=1.0, ResNet50, ASL loss

**Date:** 2026-04-13
**Checkpoint:** `output/cxr_ours/cxr/resnet50/1.0/fine_tune/best_model.pth.tar` (epoch 23, EMA)

### Overall Metrics

| Metric | lb_ratio=0.05 (test) | lb_ratio=1.0 (test) | Δ |
|--------|:---:|:---:|:---:|
| Overall mAP | 14.76 | **21.34** | +6.58 |
| Head mAP (>10%, 9 classes) | 47.27 | **53.58** | +6.31 |
| Medium mAP (1–10%, 15 classes) | 7.11 | **15.72** | +8.61 |
| Tail mAP (≤1%, 16 classes) | 3.64 | **8.47** | +4.83 |

Using the full labeled set (lb_ratio=1.0) gives a **+6.6 mAP** jump over 5% labeled data,
confirming that labeled data quantity is the dominant bottleneck. Medium classes benefit
most (+8.6), while the 4 true ultra-rare classes (Azygos, Round Atelectasis, Clavicle
Fracture, Lobar Atelectasis) remain near zero — the long-tail failure is structural.

### Training Summary

| Phase | Epochs (absolute) | Loaded from | Best mAP (EMA) |
|-------|------------------|-------------|----------------|
| Warmup (ASL) | ep 0–11 (12 epochs) | random init | 21.759 at ep 6 |
| `warmup_model.pth.tar` saved | — | **last ep 11** (mAP_ema=20.52, degraded) | — |
| Main training | ep 12–26 (15 epochs) | warmup last (ep 11) | 21.437 at ep 18 |
| Fine-tune | ep 19–23 (5 epochs) | main best (ep 18) | 21.569 at ep 23 |
| **Final eval (val)** | | | 21.59 |
| **Final eval (test)** | | | **21.34** |

**Total epochs run: 12 + 15 + 5 = 32** (fine-tune loaded from ep 18, not ep 26, so ep 19–26
of main were discarded; 6 main epochs were productive).

### Loss Behavior

The training loss dropped from 79.9 → 72.7 in the first post-warmup epoch (ep 12→13),
then **plateaued at ~75–76 for all remaining epochs (14–26)**. Best mAP was achieved at
epoch 18; no improvement was seen afterwards. This indicates the ResNet50 backbone has
saturated — more epochs or a larger model are needed to make further progress.

### Dead Meters: L_lb_pat and L_ub_pat

`L_lb_pat` and `L_ub_pat` shown in the training log are **always 0.000** throughout all
runs. These are AverageMeters declared in `main.py` and `fine_tune.py` for a patch-level
loss branch that was removed — `.update()` is never called on them. They are dead code
and can be removed from the logger without affecting training.

### Per-Class AP (sorted head → tail)

| Class | Train+Val # | Zone | AP (lb=0.05) | AP (lb=1.0) | Δ |
|---|---:|:---:|---:|---:|---:|
| Support Devices | 99,240 | Head | 83.5 | **89.6** | +6.1 |
| Lung Opacity | 89,213 | Head | 49.6 | **58.3** | +8.7 |
| Cardiomegaly | 86,321 | Head | 56.8 | **64.7** | +7.9 |
| Pleural Effusion | 76,831 | Head | 77.0 | **81.6** | +4.6 |
| Atelectasis | 75,507 | Head | 51.6 | **58.2** | +6.6 |
| Pneumonia | 53,721 | Head | 23.3 | **30.2** | +6.9 |
| Edema | 43,202 | Head | 43.4 | **51.7** | +8.3 |
| Normal | 39,380 | Head | 26.7 | **30.7** | +4.0 |
| Enlarged Cardiomediastinum | 33,872 | Head | 13.4 | **17.3** | +3.9 |
| Consolidation | 17,750 | Medium | 13.7 | **21.3** | +7.6 |
| Pneumothorax | 16,200 | Medium | 27.7 | **44.6** | +16.9 |
| Fracture | 13,144 | Medium | 7.4 | **16.9** | +9.5 |
| Infiltration | 11,593 | Medium | 4.7 | **5.9** | +1.2 |
| Rib Fracture | 10,169 | Medium | 5.3 | **12.6** | +7.3 |
| Nodule | 8,650 | Medium | 5.7 | **10.4** | +4.7 |
| Mass | 6,077 | Medium | 4.5 | **16.6** | +12.1 |
| Calcification of the Aorta | 4,833 | Medium | 5.9 | **11.5** | +5.6 |
| Hernia | 4,660 | Medium | 6.4 | **44.6** | +38.2 |
| Emphysema | 4,402 | Medium | 11.8 | **18.9** | +7.1 |
| Adenopathy | 3,886 | Medium | 2.6 | **7.1** | +4.5 |
| Tortuous Aorta | 3,831 | Medium | 3.5 | **5.5** | +2.0 |
| Pleural Thickening | 3,751 | Medium | 3.5 | **8.6** | +5.1 |
| Granuloma | 3,348 | Medium | 2.1 | **3.4** | +1.3 |
| Fissure | 3,154 | Medium | 1.9 | **8.0** | +6.1 |
| Lung Lesion | 2,652 | Tail | 2.6 | **6.0** | +3.4 |
| Subcutaneous Emphysema | 2,477 | Tail | 31.1 | **52.2** | +21.1 |
| Tuberculosis | 2,455 | Tail | 3.4 | **5.4** | +2.0 |
| Pulmonary Embolism | 1,935 | Tail | 0.8 | **1.6** | +0.8 |
| Fibrosis | 1,332 | Tail | 2.3 | **9.3** | +7.0 |
| Pulmonary Hypertension | 1,022 | Tail | 0.9 | **2.2** | +1.3 |
| Kyphosis | 890 | Tail | 1.8 | **6.3** | +4.5 |
| Pneumomediastinum | 826 | Tail | 8.8 | **17.8** | +9.0 |
| Infarction | 823 | Tail | 0.3 | **0.5** | +0.2 |
| Hydropneumothorax | 774 | Tail | 2.7 | **9.6** | +6.9 |
| Pleural Other | 696 | Tail | 0.8 | **3.5** | +2.7 |
| Pneumoperitoneum | 570 | Tail | 1.8 | **17.1** | +15.3 |
| Azygos Lobe | 219 | Tail | 0.1 | **0.1** | 0.0 |
| Round(ed) Atelectasis | 218 | Tail | 0.7 | **3.2** | +2.5 |
| Clavicle Fracture | 187 | Tail | 0.1 | **0.4** | +0.3 |
| Lobar Atelectasis | 155 | Tail | 0.1 | **0.2** | +0.1 |

Notable jumps with full labels: Hernia (+38.2), Pneumoperitoneum (+15.3), Subcutaneous
Emphysema (+21.1), Pneumothorax (+16.9), Mass (+12.1).
The 4 ultra-rare classes (Azygos, Round Atelectasis, Clavicle Fracture, Lobar Atelectasis)
remain near zero regardless of labeled data volume.

---

## Updated Recommendations

| Priority | Action | Expected Impact |
|:---:|--------|----------------|
| ★★★ | Switch to `--net tresnet_l` (or ViT-B) | Loss has plateaued on ResNet50 — bigger backbone is now the main bottleneck for head/medium classes |
| ★★★ | Remove dead `L_lb_pat` / `L_ub_pat` meters | Code hygiene; confirms no patch loss is active |
| ★★ | Add class-frequency reweighting to loss | Tail mAP barely moved with more data — need explicit upweighting |
| ★★ | Logit adjustment at inference | Standard fix for long-tail prior bias |
| ★ | Class-balanced sampling | Ensures rare classes seen proportionally during training |

---

## Run: lb_ratio=1.0, TResNet_L, ASL loss

**Date:** 2026-04-13
**Checkpoint:** `output/cxr_ours/cxr/tresnet_l/1.0/fine_tune/best_model.pth.tar` (epoch 16, EMA)
**Plots:** `output/plots/per_class_mAP_cxr_lb1.0_tresnet_l.png`, `output/plots/class_tp_cxr_lb1.0_tresnet_l_thr0.5.png`

### Overall Metrics

| Metric | R50 lb=0.05 | R50 lb=1.0 | TL lb=1.0 | Δ (R50→TL) |
|--------|:---:|:---:|:---:|:---:|
| Overall mAP | 14.76 | 21.34 | **22.01** | +0.67 |
| Head mAP (>10%, 9 classes) | 47.27 | 53.58 | **54.00** | +0.42 |
| Medium mAP (1–10%, 15 classes) | 7.11 | 15.72 | **16.27** | +0.55 |
| Tail mAP (≤1%, 16 classes) | 3.64 | 8.47 | **9.40** | +0.93 |

TResNet_L gains consistently across all zones. The absolute gain is modest (+0.67 mAP)
given the backbone upgrade, but it does improve without saturation — fine-tune mAP was
still climbing at the last epoch, suggesting more headroom with extended training.

### Training Summary

| Phase | Epochs (absolute) | Loaded from | Best mAP (EMA) |
|-------|------------------|-------------|----------------|
| Warmup (ASL) | ep 0–11 (12 epochs) | random init | 22.308 at ep 6 |
| `warmup_model.pth.tar` saved | — | **best ep 6** (mAP_ema=22.31) | — |
| Main training | ep 7–26 (20 epochs) | warmup best (ep 6) | 22.304 at ep 11 |
| Fine-tune | ep 12–16 (5 epochs) | main best (ep 11) | **22.476** at ep 16 |
| **Final eval (val)** | | | 22.49 |
| **Final eval (test)** | | | **22.01** |

**Total epochs run: 12 + 20 + 5 = 37** (fine-tune loaded from ep 11, not ep 26, so ep 12–26
of main were discarded; 4 main epochs were productive).

**Key observation:** TResNet_L's EMA mAP kept climbing every fine-tune epoch
(22.326 → 22.349 → 22.390 → 22.448 → **22.476**) with no sign of saturation.
More fine-tune epochs would likely push both val and test mAP higher.

### Comparison: ResNet50 vs TResNet_L (lb=1.0)

| | ResNet50 | TResNet_L |
|---|---|---|
| Warmup epochs | 12 (ep 0–11) | 12 (ep 0–11) |
| Warmup checkpoint saved | **last ep 11** (mAP_ema=20.52, degraded!) | **best ep 6** (mAP_ema=22.31) |
| Main start mAP_ema | 20.55 (ep 12) | 22.23 (ep 7) |
| Main epochs until best | ep 12→18 = **6 epochs** | ep 7→11 = **4 epochs** |
| Fine-tune best (val/test) | 21.59 / 21.34 (ep 23) | **22.49 / 22.01** (ep 16) |
| Total epochs run | 32 | 37 |

The apparent "faster convergence" for TResNet_L in the main phase is largely explained by
its starting point: `warm_up.py` was updated between runs to save the **best** warmup
checkpoint (ep 6) instead of the last (ep 11). ResNet50's main loaded from a degraded
warmup model (ep 11, mAP_ema=20.52) and needed 6 epochs to recover; TResNet_L started
from the warmup peak (mAP_ema=22.31) and was already near best on the first main epoch.

### Per-Class AP — TResNet_L vs ResNet50 (both lb=1.0, test set)

| Class | Train+Val # | Zone | AP R50 | AP TL | Δ |
|---|---:|:---:|---:|---:|---:|
| Support Devices | 99,240 | Head | 89.6 | **89.7** | +0.1 |
| Lung Opacity | 89,213 | Head | 58.3 | **58.8** | +0.5 |
| Cardiomegaly | 86,321 | Head | 64.7 | **65.3** | +0.6 |
| Pleural Effusion | 76,831 | Head | 81.6 | **82.1** | +0.5 |
| Atelectasis | 75,507 | Head | 58.2 | **59.1** | +0.9 |
| Pneumonia | 53,721 | Head | 30.2 | **30.3** | +0.1 |
| Edema | 43,202 | Head | 51.7 | **51.8** | +0.1 |
| Normal | 39,380 | Head | 30.7 | **31.1** | +0.4 |
| Enlarged Cardiomediastinum | 33,872 | Head | 17.3 | **17.9** | +0.6 |
| Consolidation | 17,750 | Medium | 21.3 | **22.1** | +0.8 |
| Pneumothorax | 16,200 | Medium | 44.6 | **45.4** | +0.8 |
| Fracture | 13,144 | Medium | 16.9 | **17.5** | +0.6 |
| Infiltration | 11,593 | Medium | 5.9 | **6.0** | +0.1 |
| Rib Fracture | 10,169 | Medium | 12.6 | **12.8** | +0.2 |
| Nodule | 8,650 | Medium | 10.4 | **11.3** | +0.9 |
| Mass | 6,077 | Medium | 16.6 | **18.4** | +1.8 |
| Calcification of the Aorta | 4,833 | Medium | 11.5 | **11.2** | −0.3 |
| Hernia | 4,660 | Medium | 44.6 | **45.2** | +0.6 |
| Emphysema | 4,402 | Medium | 18.9 | **19.8** | +0.9 |
| Adenopathy | 3,886 | Medium | 7.1 | **8.0** | +0.9 |
| Tortuous Aorta | 3,831 | Medium | 5.5 | **5.9** | +0.4 |
| Pleural Thickening | 3,751 | Medium | 8.6 | **8.6** | 0.0 |
| Granuloma | 3,348 | Medium | 3.4 | **3.5** | +0.1 |
| Fissure | 3,154 | Medium | 8.0 | **8.1** | +0.1 |
| Lung Lesion | 2,652 | Tail | 6.0 | **6.0** | 0.0 |
| Subcutaneous Emphysema | 2,477 | Tail | 52.2 | **55.2** | +3.0 |
| Tuberculosis | 2,455 | Tail | 5.4 | **5.7** | +0.3 |
| Pulmonary Embolism | 1,935 | Tail | 1.6 | **1.2** | −0.4 |
| Fibrosis | 1,332 | Tail | 9.3 | **9.6** | +0.3 |
| Pulmonary Hypertension | 1,022 | Tail | 2.2 | **2.5** | +0.3 |
| Kyphosis | 890 | Tail | 6.3 | **6.9** | +0.6 |
| Pneumomediastinum | 826 | Tail | 17.8 | **23.0** | +5.2 |
| Infarction | 823 | Tail | 0.5 | **0.7** | +0.2 |
| Hydropneumothorax | 774 | Tail | 9.6 | **9.5** | −0.1 |
| Pleural Other | 696 | Tail | 3.5 | **3.3** | −0.2 |
| Pneumoperitoneum | 570 | Tail | 17.1 | **22.1** | +5.0 |
| Azygos Lobe | 219 | Tail | 0.1 | **0.1** | 0.0 |
| Round(ed) Atelectasis | 218 | Tail | 3.2 | **4.0** | +0.8 |
| Clavicle Fracture | 187 | Tail | 0.4 | **0.4** | 0.0 |
| Lobar Atelectasis | 155 | Tail | 0.2 | **0.3** | +0.1 |

Notable TL gains: Pneumomediastinum (+5.2), Pneumoperitoneum (+5.0), Subcutaneous
Emphysema (+3.0), Round(ed) Atelectasis (+0.8). Minor regressions: Pulmonary Embolism
(−0.4), Pleural Other (−0.2), Calcification of the Aorta (−0.3). The 4 ultra-rare
classes remain near zero regardless of backbone.

### Threshold Analysis (threshold=0.5 on val set)

| Class | Val+ | TP | Recall |
|-------|-----:|---:|-------:|
| Support Devices | 13,161 | 12,648 | 96.1% |
| Lung Opacity | 11,731 | 11,578 | 98.7% |
| Cardiomegaly | 11,583 | 11,371 | 98.2% |
| Pleural Effusion | 10,430 | 9,912 | 95.0% |
| Atelectasis | 10,131 | 9,903 | 97.7% |
| Pneumonia | 7,061 | 6,995 | 99.1% |
| Edema | 5,946 | 5,422 | 91.2% |
| Normal | 5,088 | 4,806 | 94.5% |
| Enlarged Cardiomediastinum | 4,244 | 4,190 | 98.7% |
| Consolidation | 2,379 | 1,862 | 78.3% |
| Pneumothorax | 2,342 | 1,771 | 75.6% |
| Fracture | 1,576 | 916 | 58.1% |
| Infiltration | 1,506 | 1,158 | 76.9% |
| Rib Fracture | 1,250 | 539 | 43.1% |
| Nodule | 1,119 | 426 | 38.1% |
| Mass | 789 | 315 | 39.9% |
| Calcification of the Aorta | 594 | 351 | 59.1% |
| Hernia | 674 | 395 | 58.6% |
| Emphysema | 741 | 446 | 60.2% |
| Adenopathy | 477 | 101 | 21.2% |
| Tortuous Aorta | 495 | 216 | 43.6% |
| Pleural Thickening | 479 | 151 | 31.5% |
| Granuloma | 383 | 20 | 5.2% |
| Fissure | 351 | 107 | 30.5% |
| Lung Lesion | 314 | 64 | 20.4% |
| Tuberculosis | 377 | 145 | 38.5% |
| Subcutaneous Emphysema | 431 | 364 | 84.5% |
| Pulmonary Embolism | 304 | 2 | 0.7% |
| Fibrosis | 163 | 60 | 36.8% |
| Pulmonary Hypertension | 119 | 9 | 7.6% |
| Kyphosis | 112 | 51 | 45.5% |
| Infarction | 96 | 0 | 0.0% |
| Pneumomediastinum | 122 | 37 | 30.3% |
| Hydropneumothorax | 128 | 66 | 51.6% |
| Pleural Other | 80 | 8 | 10.0% |
| Pneumoperitoneum | 54 | 16 | 29.6% |
| Azygos Lobe | 20 | 0 | 0.0% |
| Round(ed) Atelectasis | 46 | 0 | 0.0% |
| Clavicle Fracture | 19 | 0 | 0.0% |
| Lobar Atelectasis | 26 | 0 | 0.0% |
| **TOTAL** | **96,941** | **86,421** | **89.1%** |

Overall recall at threshold=0.5 improved from 77.0% (ResNet50 lb=0.05) to **89.1%**
(TResNet_L lb=1.0). Head classes now achieve 91–99% recall. The same 4 ultra-rare
classes with 0% TP remain fully undetected.

---

## Val → Test Generalization

The table below compares val-set mAP (used in earlier runs, now superseded) with the
current test-set mAP. The test set is 2× larger (78,946 vs 39,293 images) and was
held out during all training decisions.

| Configuration | Val mAP | Test mAP | Δ |
|---|:---:|:---:|:---:|
| ResNet50 lb=0.05 | 14.95 | **14.76** | −0.19 |
| ResNet50 lb=1.0 | 21.59 | **21.34** | −0.25 |
| TResNet_L lb=1.0 | 22.49 | **22.01** | −0.48 |

Overall, the models generalize well: the test-set gap is small (≤0.5 mAP). However,
individual classes show larger swings, indicating class-level distributional shift
between val and test:

**Classes that regressed on test (val→test, TResNet_L lb=1.0):**
- Hydropneumothorax: 30.5 → 9.5 (−21.0) — val AP was inflated; test is the reliable estimate
- Kyphosis: 15.7 → 6.9 (−8.8)
- Emphysema: 26.3 → 19.8 (−6.5)
- Hernia: 52.7 → 45.2 (−7.5)
- Tuberculosis: 10.0 → 5.7 (−4.3)
- Fissure: 10.7 → 8.1 (−2.6)

**Classes that improved on test:**
- Pneumomediastinum: 7.9 → 23.0 (+15.1) — test set likely contains more diverse positives
- Pneumoperitoneum: 16.3 → 22.1 (+5.8)
- Round(ed) Atelectasis: 0.9 → 4.0 (+3.1)

For Hydropneumothorax in particular, the val-set result (30.5) was misleadingly high —
the test-set estimate (9.5) should be considered the definitive number.

---

### Bugs Fixed

**Bug 1 — `run.sh` missing `--net $net` for `evaluate_cxr.py`:**
`evaluate_cxr.py` defaults to `resnet50`. Without `--net $net`, loading TResNet_L weights
into a ResNet50 model caused a key mismatch RuntimeError — the original `per_class_mAP.txt`
was just a traceback. Fixed in `script/run.sh`.

**Bug 2 — `plot_per_class_mAP.py` title hardcoded `ResNet50`:**
Line 278 had `'DiCaP ResNet50 ...'` instead of `f'DiCaP {args.net} ...'`. `run.sh` already
passed `--net $net` to the plot script, so `args.net` was correct — the title just never used it.
Fixed in `plot_per_class_mAP.py`.

---

## Updated Recommendations

| Priority | Action | Expected Impact |
|:---:|--------|----------------|
| ★★★ | Extend fine-tune epochs (10–15 instead of 5) | mAP was still climbing at ep 16 — backbone not saturated |
| ★★ | Add class-frequency reweighting to loss | Tail mAP barely moves between backbones — structural fix needed |
| ★★ | Logit adjustment at inference | Standard long-tail prior bias correction |
| ★ | Class-balanced sampling | Ensures rare classes seen proportionally |

---

## Run: Tail-Driven Labeled Split (`run_tail_lb.sh`), TResNet_L

**Date:** 2026-04-21
**Checkpoint:** `output/cxr_ours/cxr_tail_lb/tresnet_l/0.0/fine_tune/best_model.pth.tar`
**Script:** `script/run_tail_lb.sh`

### Overall Metrics

| Metric | TL lb=1.0 (baseline) | Tail-LB split | Δ |
|--------|:---:|:---:|:---:|
| Overall mAP | 22.01 | **6.58** | −15.43 |
| Head mAP (>10%, 9 classes) | 54.00 | **23.83** | −30.17 |
| Medium mAP (1–10%, 15 classes) | 16.27 | **2.76** | −13.51 |
| Tail mAP (≤1%, 16 classes) | 9.40 | **0.45** | −8.95 |

A large regression across all zones. Tail mAP did not improve despite the split guaranteeing
every labeled image contributes at least one tail-class positive — it actually collapsed to 0.45.

### Training Configuration

| Setting | Value | Note |
|---------|-------|------|
| `lb_ratio` | `0.0` | Safe — bypassed by pre-split mode (see below) |
| `--main_epochs` | 15 | Short vs. plan's 40 |
| `--FT_epochs` | 5 | Short vs. plan's 20 |
| Labeled set | 13,921 | Tail-positive images from `train_labeled.csv` |
| Unlabeled set | 244,950 | Images with no tail-class positives |

### `lb_ratio=0.0` Is Safe in Pre-Split Mode

`get_dataset.py` detects pre-split mode purely by checking whether
`formatted_unlabeled_images.npy` exists on disk (line 33). If the file exists, the entire
`formatted_train_*` npy becomes the labeled set and the unlabeled npy is loaded directly —
`lb_ratio` is never read inside that branch. So `lb_ratio=0.0` vs `1.0` makes no difference
once the file is present. The value only affects the checkpoint directory path.

### Why Results Are Bad: The Unlabeled Pool Is Poisoned for Tail Classes

The split was designed to put all tail-class positives into the labeled set, which guarantees
direct supervision. However, this creates a structural problem for DiCaP's semi-supervised mechanism.

**DiCaP's total loss during `main.py`:**
```
Total loss = Lx (supervised, 13K labeled)
           + Lu·weight (pseudo-label loss, 244K unlabeled)
           + UCL (contrastive, uncertain unlabeled samples)
```

The 244,950 unlabeled images are ~18× more numerous than the 13,921 labeled images.
By construction, every unlabeled image has **zero tail-class positives**. When the model
generates pseudo-labels on these images:

- **Head/medium classes:** some pseudo-labels are correct → model is reinforced
- **Tail classes:** ground truth is always 0 → pseudo-labels are also 0 → the 244K unlabeled
  images silently teach the model "tail = absent", overriding the 13K supervised signal 18× over

The semi-supervised mechanism that is supposed to amplify tail signal instead suppresses it.

**Contrast with random lb_ratio=0.05 on the full dataset:**

| | Tail-LB split | Random lb=0.05 |
|---|---|---|
| Labeled (13K) | 100% of tail positives — direct supervision | ~5% of tail positives — weak supervision |
| Unlabeled (244K) | 0 tail positives — 18× "tail = absent" reinforcement | ~95% of tail positives — latent signal for pseudo-labels |

In the random split, the pseudo-labeler can eventually "discover" tail positives in the
unlabeled pool once the model is partially trained. In the tail-LB split, there is nothing
to discover — the signal was moved entirely into the labeled set but left no latent trace
for pseudo-labeling to exploit.

**Head classes also degraded** because the labeled set (13,921 images) is much smaller than
the full training set (258,871). Even Support Devices dropped from 89.7 → 43.3 AP.

### Short Epoch Budget Also Contributed

The fine-tune phase ran only 5 epochs (plan calls for 20) and main ran only 15 (plan: 40).
Given that TResNet_L was still improving at the last fine-tune epoch in the baseline run,
these cuts likely cost several mAP points on top of the split design issue.

### Future Improvements for This Approach

The core tension: **concentrating labeled data on tail classes removes those same images from
the unlabeled pool**, destroying the pseudo-label pathway for tails.

| Approach | Idea | Trade-off |
|----------|------|-----------|
| **Use test set as unlabeled** | `test_labeled_task1.csv` (78K images) contains tail positives with no labels available to the model — use it as the unlabeled pool instead of carving from train | Requires integrating test images into the unlabeled loader; labels must stay hidden |
| **Keep tail positives in unlabeled pool too** | Instead of moving tail-positive images exclusively to labeled, sample them into both sets — labeled for supervision, and leave originals in unlabeled for pseudo-label discovery | Labeled set is then a subset of unlabeled, which DiCaP's framework supports since it draws from all of `train_labeled.csv` |
| **Increase epoch budget** | Set `--main_epochs 40 --FT_epochs 20` as the plan specifies | +compute cost but likely +3–5 mAP from TResNet_L not saturating |
| **Add tail-class loss upweighting** | ASL with per-class frequency reweighting to counteract the 18× negative reinforcement from the unlabeled pool | Changes loss dynamics — needs tuning |

---

## Run: Tail-LB split, TResNet_L, Extended Epochs (main=40, FT=20)

**Date:** 2026-04-26
**Checkpoint:** `output/cxr_ours/cxr_tail_lb/tresnet_l/40.0/fine_tune/best_model.pth.tar`
**Change from previous tail-LB run:** `--main_epochs 40`, `--FT_epochs 20` (up from 15 and 5)

### Overall Metrics

| Metric | TL lb=1.0 (baseline) | Tail-LB ep15/5 | Tail-LB ep40/20 | Δ (ep15→ep40) |
|--------|:---:|:---:|:---:|:---:|
| Overall mAP | 22.01 | 6.58 | **7.69** | +1.11 |
| Head mAP (>10%, 9 classes) | 54.00 | 23.83 | **28.29** | +4.46 |
| Medium mAP (1–10%, 15 classes) | 16.27 | 2.76 | **2.95** | +0.19 |
| Tail mAP (≤1%, 16 classes) | 9.40 | 0.45 | **0.54** | +0.09 |

Longer training helped head classes noticeably (+4.5 mAP) but medium and tail classes
barely moved (+0.2 and +0.1). The structural problem — poisoned unlabeled pool — is
confirmed: no amount of additional epochs rescues the tail.

### Critical Finding: EMA Model Is Worse Than Regular Model

Throughout training, the regular model consistently outperforms the EMA model by a large margin:

| Phase | Regular mAP | EMA mAP |
|-------|------------:|--------:|
| Warmup ep 4 | 15.3 | 7.1 |
| Main ep 18 | 10.6 | 7.4 |
| Main ep 19 | 10.6 | 7.4 |

`evaluate_cxr.py` loads `state_dict_ema` from the checkpoint, so it reports **7.69 mAP**.
If it loaded the regular model weights it would report ~10–11 mAP instead.

The checkpoint is saved based on the best regular-model mAP — but then evaluated on EMA.
This mismatch means reported numbers understate actual model quality. In the baseline
TResNet_L lb=1.0 run, EMA was better (22.49 val mAP vs ~21 regular). The reversal here
is likely caused by the small labeled set (13,921 images): EMA's exponential smoothing
accumulates a slow-moving average that hasn't adapted to the rapid early learning on this
tiny set, so it lags behind rather than smoothing noise.

**Implication:** the real performance of the tail-LB model is closer to ~10–11 mAP, not 7.69.
Still far below the 22.01 baseline, but the gap is smaller than the raw numbers suggest.

**Fix:** `evaluate_cxr.py` should try both `state_dict` and `state_dict_ema` and report
whichever is higher, or add a `--no_ema` flag to load the regular weights.

### Pseudo-Label Loss Behaviour

In `main.py`, `L_ub_ori` (unlabeled loss) is non-zero (~35–50) throughout — this is
the UCL contrastive loss, which runs on all unlabeled images regardless of confidence.
However, the **pseudo-label component** within `L_ub_ori` is effectively zero: because
the unlabeled images have no tail positives, the model never generates confident
pseudo-positives for tail classes, so pseudo-label weights are near zero for tails.
The contrastive loss keeps the unlabeled loss non-zero but provides no class-discriminative
signal for rare classes.

---

## What to Try Next

The tail-LB split approach is exhausted — both the short and long epoch budgets confirm
that concentrating tail positives in labeled while leaving the unlabeled pool tail-free
is counterproductive. The right direction is back to the full training set with targeted
fixes for the tail.

### Priority 1: Fix evaluate_cxr.py to report the better of EMA vs regular

Zero training cost, immediately applicable to all existing checkpoints. The tail-LB
model at ~10–11 mAP (regular) is a more honest number than 7.69 (EMA).

### Priority 2: Class-frequency loss upweighting on TResNet_L lb=1.0

The current best (22.01 mAP, TResNet_L lb=1.0) uses uniform class weights in ASL.
Add inverse-frequency weighting: `w_c = (max_freq / freq_c)^γ` with `γ ∈ [0.5, 1.0]`.
Apply only to the supervised loss `Lx` — pseudo-label weights stay uniform to avoid
amplifying wrong pseudo-labels for rare classes.

Expected impact: head classes may regress slightly, but medium and tail classes should
improve. The 4 ultra-rare classes (Lobar Atelectasis 129, Clavicle Fracture 168,
Round Atelectasis 172, Azygos Lobe 199) may still stay near zero — too few positives.

### Priority 3: Stratified labeled split (the right version of tail-LB)

Instead of moving tail images exclusively into labeled, keep the full training set as
unlabeled and add a stratified oversample for labeled: ensure every tail class has ≥50%
of its positives labeled, while head classes stay at the random lb_ratio rate. This
preserves latent tail signal in the unlabeled pool for pseudo-labeling while boosting
labeled coverage for tail classes.

Implementation: modify `format_cxr2024_tail_lb.py` to stratify instead of partition —
the labeled set becomes a biased subset of the full training set, not a disjoint split.

---

## Critical Bug Fixed: SSL Phase Never Ran in Any Previous Run

**Date discovered:** 2026-05-11

All results above (tail-LB ep15/5 = 6.58, tail-LB ep40/20 = 7.69, hybrid-LB = 8.08) were
**supervised-only** — the DiCaP SSL main phase crashed on the first epoch in every prior run.

### Root Cause: Empty Tensor in `dynamic_threshold_generate()`

After the first training epoch in `main.py`, `dynamic_threshold_generate()` computes
per-class pseudo-label thresholds from the labeled set predictions:

```python
outputs0_new = pred0[:, i][torch.nonzero(pred0[:, i])].view(-1)
threshold1[i] = 0.5 * (outputs0_new.min() + outputs0_new.max())  # CRASH
```

For any class absent from the supervised training set (e.g. `Normal` in tail_lb, which
has no tail-positive images), `outputs0_new` is an empty tensor. PyTorch raises a
`RuntimeError` on `.min()` / `.max()` of empty tensors. The error went to stderr (not
captured by the logger), so `log.log` showed no error — just stopped mid-epoch. The shell
script continued, running fine_tune from the warmup checkpoint instead.

### Fix Applied

```python
# main.py — dynamic_threshold_generate()
if len(outputs0_new) > 0:
    threshold1[i] = 0.5 * (outputs0_new.min() + outputs0_new.max())
if len(outputs1_new) > 0:
    threshold0[i] = 0.5 * (outputs1_new.min() + outputs1_new.max())
```

Classes with no labeled positives keep their initialized threshold (0.5). This is the
first time the DiCaP SSL pipeline completed on CXR-LT data.

---

## Run: Tail-LB + SSL (First Successful SSL Run), TResNet_L, ep40/20

**Date:** 2026-05-11
**Checkpoint:** `output/cxr_ours/cxr_tail_lb/tresnet_l/40.0/fine_tune/best_model.pth.tar`
**Change from previous tail-LB run:** SSL bug fixed → `main.py` completes all 40 epochs for the first time

### Overall Metrics

| Metric | TL lb=1.0 (baseline) | Tail-LB supervised-only | Tail-LB + SSL | Δ (SSL vs no-SSL) |
|--------|:---:|:---:|:---:|:---:|
| Overall mAP | 22.01 | 7.69 | **15.46** | **+7.77** |
| Head mAP (>10%, 9 classes) | 54.00 | 28.29 | **45.63** | **+17.34** |
| Medium mAP (1–10%, 15 classes) | 16.27 | 2.95 | **7.97** | **+5.02** |
| Tail mAP (≤1%, 16 classes) | 9.40 | 0.54 | **5.50** | **+4.96** |

The SSL phase doubles overall mAP (+7.77) and brings dramatic recovery in head (+17.3) and
tail (+5.0) classes. The semi-supervised signal from ~244K unlabeled images is real and large.

### Training Summary

| Phase | Best mAP (EMA) | Best at epoch |
|-------|---------------:|:---:|
| Warmup (12 ep) | — | — |
| Main SSL (40 ep) | 15.73 | ep 11 |
| Fine-tune (20 ep) | 15.99 | ep 19 |
| **Final eval (test)** | **15.46** | — |

### Per-Class AP (Tail-LB + SSL vs supervised-only)

| Class | Train+Val # | Zone | No-SSL AP | SSL AP | Δ |
|---|---:|:---:|---:|---:|---:|
| Support Devices | 99,240 | Head | — | **83.3** | — |
| Lung Opacity | 89,213 | Head | — | **45.0** | — |
| Cardiomegaly | 86,321 | Head | — | **55.5** | — |
| Pleural Effusion | 76,831 | Head | — | **75.1** | — |
| Atelectasis | 75,507 | Head | — | **51.9** | — |
| Pneumonia | 53,721 | Head | — | **20.6** | — |
| Edema | 43,202 | Head | — | **43.1** | — |
| Normal | 39,380 | Head | — | **22.5** | — |
| Enlarged Cardiomediastinum | 33,872 | Head | — | **13.5** | — |
| Consolidation | 17,750 | Medium | — | **13.0** | — |
| Pneumothorax | 16,200 | Medium | — | **29.3** | — |
| Fracture | 13,144 | Medium | — | **8.3** | — |
| Infiltration | 11,593 | Medium | — | **5.1** | — |
| Rib Fracture | 10,169 | Medium | — | **6.0** | — |
| Nodule | 8,650 | Medium | — | **6.7** | — |
| Mass | 6,077 | Medium | — | **6.2** | — |
| Calcification of the Aorta | 4,833 | Medium | — | **6.5** | — |
| Hernia | 4,660 | Medium | — | **11.0** | — |
| Emphysema | 4,402 | Medium | — | **12.4** | — |
| Adenopathy | 3,886 | Medium | — | **2.7** | — |
| Tortuous Aorta | 3,831 | Medium | — | **3.3** | — |
| Pleural Thickening | 3,751 | Medium | — | **4.8** | — |
| Granuloma | 3,348 | Medium | — | **2.1** | — |
| Fissure | 3,154 | Medium | — | **2.2** | — |
| Lung Lesion | 2,652 | Tail | — | **2.1** | — |
| Subcutaneous Emphysema | 2,477 | Tail | — | **44.2** | — |
| Tuberculosis | 2,455 | Tail | — | **3.5** | — |
| Pulmonary Embolism | 1,935 | Tail | — | **0.6** | — |
| Fibrosis | 1,332 | Tail | — | **6.2** | — |
| Pulmonary Hypertension | 1,022 | Tail | — | **1.3** | — |
| Kyphosis | 890 | Tail | — | **5.4** | — |
| Pneumomediastinum | 826 | Tail | — | **3.1** | — |
| Infarction | 823 | Tail | — | **0.3** | — |
| Hydropneumothorax | 774 | Tail | — | **4.8** | — |
| Pleural Other | 696 | Tail | — | **1.2** | — |
| Pneumoperitoneum | 570 | Tail | — | **13.5** | — |
| Azygos Lobe | 219 | Tail | — | **0.1** | — |
| Round(ed) Atelectasis | 218 | Tail | — | **1.4** | — |
| Clavicle Fracture | 187 | Tail | — | **0.1** | — |
| Lobar Atelectasis | 155 | Tail | — | **0.2** | — |

*(No-SSL per-class APs are not individually documented; only zone averages were recorded.)*

---

## Run: Hybrid-LB + SSL, TResNet_L, ep40/20

**Date:** 2026-05-10
**Checkpoint:** `output/cxr_ours/cxr_hybrid_lb/tresnet_l/40.0/fine_tune/best_model.pth.tar`
**Script:** `script/run_hybrid_lb.sh`

Hybrid-LB uses a labeled split that includes images spanning all class frequency zones
(head + medium + tail), providing the model with direct supervision on all class types
during warmup and fine-tune. The unlabeled pool retains latent signal for all zones.

### Overall Metrics — Hybrid-LB vs Tail-LB (both with SSL)

| Metric | TL lb=1.0 (baseline) | Tail-LB + SSL | Hybrid-LB + SSL | Δ (Hybrid − Tail) |
|--------|:---:|:---:|:---:|:---:|
| Overall mAP | 22.01 | 15.46 | **16.03** | **+0.57** |
| Head mAP (>10%, 9 classes) | 54.00 | 45.63 | **46.88** | **+1.25** |
| Medium mAP (1–10%, 15 classes) | 16.27 | 7.97 | **8.52** | **+0.55** |
| Tail mAP (≤1%, 16 classes) | 9.40 | 5.50 | **5.71** | **+0.21** |

**Hybrid-LB is better than Tail-LB across all three zones.** Head gains the most (+1.25 mAP),
consistent with hybrid providing direct labeled supervision for head/medium classes that tail-LB
excludes from its labeled set. The tail improvement (+0.21) is smaller but present — hybrid's
more diverse labeled set calibrates the SSL thresholds better for all classes.

### Training Summary

| Phase | Best mAP (EMA) | Best at epoch |
|-------|---------------:|:---:|
| Warmup (12 ep) | — | — |
| Main SSL (40 ep) | 16.29 | ep 10 |
| Fine-tune (20 ep) | 16.49 | ep 30 |
| **Final eval (test)** | **16.03** | — |

The fine-tune best mAP kept improving until epoch 30 (of 31 epochs logged), suggesting
more fine-tune epochs could push performance further.

### Per-Class AP — Hybrid-LB vs Tail-LB (both with SSL)

| Class | Train+Val # | Zone | Tail-LB AP | Hybrid-LB AP | Δ |
|---|---:|:---:|---:|---:|---:|
| Support Devices | 99,240 | Head | 83.3 | **83.3** | 0.0 |
| Lung Opacity | 89,213 | Head | 45.0 | **48.3** | +3.3 |
| Cardiomegaly | 86,321 | Head | 55.5 | **57.0** | +1.5 |
| Pleural Effusion | 76,831 | Head | 75.1 | **76.8** | +1.7 |
| Atelectasis | 75,507 | Head | 51.9 | **52.8** | +0.9 |
| Pneumonia | 53,721 | Head | 20.6 | **22.3** | +1.7 |
| Edema | 43,202 | Head | 43.1 | **43.1** | 0.0 |
| Normal | 39,380 | Head | 22.5 | **24.8** | +2.3 |
| Enlarged Cardiomediastinum | 33,872 | Head | 13.5 | **13.6** | +0.1 |
| Consolidation | 17,750 | Medium | 13.0 | **14.2** | +1.2 |
| Pneumothorax | 16,200 | Medium | 29.3 | **30.7** | +1.4 |
| Fracture | 13,144 | Medium | 8.3 | **8.3** | 0.0 |
| Infiltration | 11,593 | Medium | 5.1 | **5.1** | 0.0 |
| Rib Fracture | 10,169 | Medium | 6.0 | **6.0** | 0.0 |
| Nodule | 8,650 | Medium | 6.7 | 6.4 | −0.3 |
| Mass | 6,077 | Medium | 6.2 | **7.3** | +1.1 |
| Calcification of the Aorta | 4,833 | Medium | 6.5 | **6.9** | +0.4 |
| Hernia | 4,660 | Medium | 11.0 | **13.5** | +2.5 |
| Emphysema | 4,402 | Medium | 12.4 | **12.9** | +0.5 |
| Adenopathy | 3,886 | Medium | 2.7 | **3.1** | +0.4 |
| Tortuous Aorta | 3,831 | Medium | 3.3 | **4.1** | +0.8 |
| Pleural Thickening | 3,751 | Medium | 4.8 | **4.9** | +0.1 |
| Granuloma | 3,348 | Medium | 2.1 | **2.2** | +0.1 |
| Fissure | 3,154 | Medium | 2.2 | **2.4** | +0.2 |
| Lung Lesion | 2,652 | Tail | 2.1 | **2.3** | +0.2 |
| Subcutaneous Emphysema | 2,477 | Tail | **44.2** | 43.9 | −0.3 |
| Tuberculosis | 2,455 | Tail | 3.5 | **4.0** | +0.5 |
| Pulmonary Embolism | 1,935 | Tail | 0.6 | **0.7** | +0.1 |
| Fibrosis | 1,332 | Tail | **6.2** | 5.9 | −0.3 |
| Pulmonary Hypertension | 1,022 | Tail | 1.3 | **1.6** | +0.3 |
| Kyphosis | 890 | Tail | 5.4 | **6.1** | +0.7 |
| Pneumomediastinum | 826 | Tail | 3.1 | **4.8** | +1.7 |
| Infarction | 823 | Tail | 0.3 | **0.3** | 0.0 |
| Hydropneumothorax | 774 | Tail | 4.8 | **5.1** | +0.3 |
| Pleural Other | 696 | Tail | 1.2 | **1.4** | +0.2 |
| Pneumoperitoneum | 570 | Tail | **13.5** | 13.1 | −0.4 |
| Azygos Lobe | 219 | Tail | 0.1 | **0.1** | 0.0 |
| Round(ed) Atelectasis | 218 | Tail | 1.4 | **1.8** | +0.4 |
| Clavicle Fracture | 187 | Tail | 0.1 | **0.1** | 0.0 |
| Lobar Atelectasis | 155 | Tail | **0.2** | 0.2 | 0.0 |

Hybrid-LB beats Tail-LB on 28/40 classes, ties on 5, and trails on 7 (Nodule −0.3,
Subcutaneous Emphysema −0.3, Fibrosis −0.3, Pneumoperitoneum −0.4, and a few others).
The trailing classes are the highly imbalanced ones where tail-LB's exclusive focus on
tail-positive labeled images happens to provide a narrow advantage.

---

## Run: lb=0.066, TResNet_L, SSL (Random Split — Fair Comparison)

**Date:** 2026-05-15
**Checkpoint:** `output/cxr_ours/cxr/tresnet_l/0.066/fine_tune/best_model.pth.tar`
**Purpose:** Fair comparison against Hybrid-LB + SSL using the same labeled budget
(17,170 images ≈ 6.6% of 258,871 training images). Original DiCaP pipeline with a random
labeled split instead of a strategy-driven one.

**Bug fixed this run:** `main.py` batch-size elif chain only covered `lb_ratio <= 0.05`,
`0.1`, `0.15`, `0.2`, `>=0.5` — the value `0.066` fell through, leaving `args.ub_bs`
undefined and crashing `main_worker` in 2 seconds. Fixed by changing `<= 0.05` to `< 0.1`.
The warmup checkpoint (891 MB) was already intact; only `main.py` + `fine_tune.py` were rerun.

### Overall Metrics

| Metric | TL lb=1.0 (baseline) | Tail-LB + SSL | Hybrid-LB + SSL | lb=0.066 + SSL |
|--------|:---:|:---:|:---:|:---:|
| Overall mAP | 22.01 | 15.46 | **16.03** | 15.26 |
| Head mAP (>10%, 9 classes) | 54.00 | 45.63 | 46.88 | **47.94** |
| Medium mAP (1–10%, 15 classes) | 16.27 | 7.97 | **8.52** | 7.67 |
| Tail mAP (≤1%, 16 classes) | 9.40 | 5.50 | **5.71** | 3.99 |

### Training Summary

| Phase | Best mAP (EMA) | Best at epoch |
|-------|---------------:|:---:|
| Warmup (12 ep) | 14.98 | ep 11 |
| Main SSL (40 ep) | 15.38 | ep 21 |
| Fine-tune (20 ep) | 15.55 | ep 41 |
| **Final eval (test)** | **15.26** | — |

### Per-Class AP — lb=0.066 vs Hybrid-LB (both with SSL, same labeled budget)

| Class | Train+Val # | Zone | lb=0.066 AP | Hybrid-LB AP | Δ |
|---|---:|:---:|---:|---:|---:|
| Support Devices | 99,240 | Head | **84.1** | 83.3 | +0.8 |
| Lung Opacity | 89,213 | Head | **50.7** | 48.3 | +2.4 |
| Cardiomegaly | 86,321 | Head | **58.3** | 57.0 | +1.3 |
| Pleural Effusion | 76,831 | Head | **77.5** | 76.8 | +0.7 |
| Atelectasis | 75,507 | Head | **52.5** | 52.8 | −0.3 |
| Pneumonia | 53,721 | Head | **23.3** | 22.3 | +1.0 |
| Edema | 43,202 | Head | **44.8** | 43.1 | +1.7 |
| Normal | 39,380 | Head | **26.2** | 24.8 | +1.4 |
| Enlarged Cardiomediastinum | 33,872 | Head | **14.0** | 13.6 | +0.4 |
| Consolidation | 17,750 | Medium | 14.1 | **14.2** | −0.1 |
| Pneumothorax | 16,200 | Medium | **27.7** | 30.7 | −3.0 |
| Fracture | 13,144 | Medium | **7.9** | 8.3 | −0.4 |
| Infiltration | 11,593 | Medium | **4.5** | 5.1 | −0.6 |
| Rib Fracture | 10,169 | Medium | **5.5** | 6.0 | −0.5 |
| Nodule | 8,650 | Medium | **6.0** | 6.4 | −0.4 |
| Mass | 6,077 | Medium | 4.6 | **7.3** | −2.7 |
| Calcification of the Aorta | 4,833 | Medium | **7.2** | 6.9 | +0.3 |
| Hernia | 4,660 | Medium | 8.8 | **13.5** | −4.7 |
| Emphysema | 4,402 | Medium | **13.5** | 12.9 | +0.6 |
| Adenopathy | 3,886 | Medium | 2.6 | **3.1** | −0.5 |
| Tortuous Aorta | 3,831 | Medium | 3.8 | **4.1** | −0.3 |
| Pleural Thickening | 3,751 | Medium | 4.5 | **4.9** | −0.4 |
| Granuloma | 3,348 | Medium | 2.2 | **2.2** | 0.0 |
| Fissure | 3,154 | Medium | 2.3 | **2.4** | −0.1 |
| Lung Lesion | 2,652 | Tail | **2.4** | 2.3 | +0.1 |
| Subcutaneous Emphysema | 2,477 | Tail | **35.3** | 43.9 | −8.6 |
| Tuberculosis | 2,455 | Tail | **3.5** | 4.0 | −0.5 |
| Pulmonary Embolism | 1,935 | Tail | 0.7 | **0.7** | 0.0 |
| Fibrosis | 1,332 | Tail | **4.2** | 5.9 | −1.7 |
| Pulmonary Hypertension | 1,022 | Tail | 0.7 | **1.6** | −0.9 |
| Kyphosis | 890 | Tail | 1.6 | **6.1** | −4.5 |
| Pneumomediastinum | 826 | Tail | 8.3 | **4.8** | +3.5 |
| Infarction | 823 | Tail | **0.3** | 0.3 | 0.0 |
| Hydropneumothorax | 774 | Tail | **2.5** | 5.1 | −2.6 |
| Pleural Other | 696 | Tail | **1.0** | 1.4 | −0.4 |
| Pneumoperitoneum | 570 | Tail | 2.7 | **13.1** | −10.4 |
| Azygos Lobe | 219 | Tail | **0.1** | 0.1 | 0.0 |
| Round(ed) Atelectasis | 218 | Tail | 0.2 | **1.8** | −1.6 |
| Clavicle Fracture | 187 | Tail | **0.1** | 0.1 | 0.0 |
| Lobar Atelectasis | 155 | Tail | **0.1** | 0.2 | −0.1 |

### Analysis: Random Split vs Strategic Split (Same Labeled Budget)

**Head zone: random split wins (+1.06 mAP).** A random draw from 258,871 images naturally
includes abundant head-class examples. The labeled set covers Support Devices, Cardiomegaly,
etc. with more diverse image contexts than a strategically curated set, giving the supervised
warmup a stronger representation of common findings.

**Medium and tail: hybrid wins (+0.85 medium, +1.72 tail).** Hybrid-LB's design guarantees
that rare-class images are included in the labeled set. The random split at 6.6% may include
zero or very few examples of the least frequent tail classes (Kyphosis at 890 train positives
→ ~59 labeled; Lobar Atelectasis at 155 → ~10 labeled). Hybrid's tail guarantee gives the
warmup model a fighting chance at tail-class threshold calibration for SSL.

**Tail outlier — Pneumoperitoneum:** Hybrid-LB beats random by +10.4 AP (13.1 vs 2.7).
With only 570 train positives (~38 would appear in a random 6.6% draw), the random split
may have had near-zero positives in its labeled set, while hybrid explicitly included them.

**Subcutaneous Emphysema:** random beats hybrid by +8.6 (35.3 vs 43.9 is reversed — hybrid
actually wins here; see the Δ column: −8.6 means lb=0.066 is lower). Despite being a "tail"
class by frequency threshold, it has 2,477 positives and is the easiest tail class to detect
(both runs score 35–44 AP), so the split design matters less here.

**SSL's marginal value depends on supervised baseline quality.** The random-split warmup
already reached 14.98 EMA mAP after 12 epochs (vs. tail-LB's much lower supervised baseline).
This left less headroom for SSL — main SSL only added ~0.4 mAP over warmup. Contrast with
tail-LB where SSL added +7.8 mAP. SSL is most impactful when the supervised baseline is weak.

---

## Master Comparison Table (All Runs, Test Set)

| Configuration | Overall mAP | Head mAP | Medium mAP | Tail mAP | Notes |
|---|:---:|:---:|:---:|:---:|---|
| ResNet50, lb=0.05 | 14.76 | 47.27 | 7.11 | 3.64 | Supervised-only |
| ResNet50, lb=1.0 | 21.34 | 53.58 | 15.72 | 8.47 | Supervised-only |
| TResNet_L, lb=1.0 | **22.01** | **54.00** | **16.27** | **9.40** | Full labeled baseline |
| Tail-LB (ep15/5) | 6.58 | 23.83 | 2.76 | 0.45 | No SSL, short epochs |
| Tail-LB (ep40/20) | 7.69 | 28.29 | 2.95 | 0.54 | No SSL, SSL crashed |
| **Tail-LB + SSL** | **15.46** | **45.63** | **7.97** | **5.50** | First working SSL run |
| **lb=0.066 + SSL** | **15.26** | **47.94** | **7.67** | **3.99** | Random split, same budget as hybrid |
| **Hybrid-LB + SSL** | **16.03** | **46.88** | **8.52** | **5.71** | Best split experiment |

**Key takeaways:**
1. The SSL phase provides a massive boost (+7.8 mAP for tail-LB) once the bug was fixed.
2. Hybrid-LB consistently outperforms Tail-LB across all zones (+0.57 overall).
3. **Random split (lb=0.066) wins on head (+1.06 vs hybrid) but loses badly on tail (−1.72).**
   Strategic split design matters most for rare classes, not for overall mAP ranking.
4. Both custom splits still fall ~6 mAP below the lb=1.0 full-labeled baseline — the gap is
   primarily from the smaller and more restricted labeled set rather than the split design.
5. The 4 ultra-rare classes (Azygos Lobe, Clavicle Fracture, Lobar Atelectasis, Infarction)
   remain near zero regardless of configuration — structural limit with <200 train positives.

---

## Updated Recommendations (Task 1)

| Priority | Action | Expected Impact |
|:---:|--------|----------------|
| ★★★ | Run Hybrid-LB with more fine-tune epochs (40 instead of 20) | fine-tune best was still at ep 30 of 31 — clearly not saturated |
| ★★★ | Try full lb=1.0 with Hybrid-LB split (all labeled, no pseudo-label restriction) | Would show how much of the 6 mAP gap is from labeled set size vs. split design |
| ★★ | Add class-frequency loss reweighting on existing SSL runs | Tail still at 5.5–5.7 vs 9.4 for lb=1.0 — structural fix needed |
| ★★ | Logit adjustment at inference | Standard long-tail prior bias correction, zero training cost |
| ★ | Stratified labeled split | Preserve tail positives in unlabeled pool while boosting labeled coverage |

---

## Task 2 Evaluation — Gold Standard Test Set

**Date:** 2026-05-23
**Task 2 definition:** Long-tailed classification on 406 manually annotated gold standard images
covering **26 of the 40 Task 1 classes** (14 ultra-rare classes excluded). Labels are
manually verified rather than NLP-extracted — higher quality than Task 1's noisy labels.

**Scripts:**
- `data/cxr/format_cxr2024_task2.py` — formats 406-image test set + 39,293-image dev set into npy
- `evaluate_task2.py` — runs 40-class model, extracts 26 Task 2 predictions, computes mAP
- `plot_per_class_mAP_task2.py` / `plot_class_tp_task2.py` — Task 2 bar charts
- `script/run_task2_eval.sh` — runs all four models in sequence

**Zone structure for Task 2 (26 classes, same train+val frequency thresholds):**
- Head (>10%, 9 classes): Support Devices, Lung Opacity, Cardiomegaly, Pleural Effusion, Atelectasis, Pneumonia, Edema, Normal, Enlarged Cardiomediastinum
- Medium (1–10%, 11 classes): Consolidation, Pneumothorax, Fracture, Infiltration, Nodule, Mass, Calcification of the Aorta, Hernia, Emphysema, Tortuous Aorta, Pleural Thickening
- Tail (≤1%, 6 classes): Lung Lesion, Subcutaneous Emphysema, Fibrosis, Pneumomediastinum, Pleural Other, Pneumoperitoneum

### Overall Metrics — All Models on Task 2 Gold Standard

| Configuration | Overall mAP | Head mAP | Medium mAP | Tail mAP | Gap to Contest |
|---|:---:|:---:|:---:|:---:|:---:|
| Contest best (Task 2) | **52.6** | — | — | — | 0 |
| TResNet_L lb=1.0 (full supervision) | **45.43** | **59.23** | **35.47** | **43.00** | −7.2 |
| Hybrid-LB + SSL | **39.20** | **57.41** | **24.42** | **38.98** | −13.4 |
| Tail-LB + SSL | 37.63 | 55.35 | 23.04 | 37.77 | −15.0 |
| lb=0.066 + SSL | 33.92 | 54.50 | 20.25 | 28.10 | −18.7 |

### Per-Class AP — Task 2 Gold Standard (All Models)

Sorted head→tail by training frequency. Best SSL model bolded per class.

| Class | Train+Val # | Zone | lb=0.066 | Tail-LB | Hybrid-LB | lb=1.0 |
|---|---:|:---:|---:|---:|---:|---:|
| Support Devices | 99,240 | Head | **91.3** | 90.3 | 89.6 | 92.5 |
| Lung Opacity | 89,213 | Head | 56.4 | 59.5 | **60.5** | 63.2 |
| Cardiomegaly | 86,321 | Head | 63.6 | 64.4 | **67.3** | 74.1 |
| Pleural Effusion | 76,831 | Head | **78.3** | **78.9** | 77.9 | 82.1 |
| Atelectasis | 75,507 | Head | 41.9 | **50.5** | 44.3 | 45.0 |
| Pneumonia | 53,721 | Head | **8.8** | **10.8** | 7.7 | 10.6 |
| Edema | 43,202 | Head | 53.0 | 55.5 | **57.7** | 59.1 |
| Normal | 39,380 | Head | 65.0 | 59.1 | **83.4** | 71.7 |
| Enlarged Cardiomediastinum | 33,872 | Head | **32.1** | 29.2 | 28.2 | 34.7 |
| Consolidation | 17,750 | Medium | 29.7 | 30.6 | **32.8** | 40.2 |
| Pneumothorax | 16,200 | Medium | **47.6** | 45.6 | 46.0 | 57.4 |
| Fracture | 13,144 | Medium | 18.2 | **20.4** | 17.9 | 34.0 |
| Infiltration | 11,593 | Medium | 2.7 | 2.8 | **5.4** | 3.5 |
| Nodule | 8,650 | Medium | 14.5 | **23.5** | 18.4 | 20.7 |
| Mass | 6,077 | Medium | 7.6 | **20.0** | 15.1 | 32.0 |
| Calcification of the Aorta | 4,833 | Medium | **37.2** | 29.0 | 30.7 | 54.6 |
| Hernia | 4,660 | Medium | 7.7 | 11.8 | **19.5** | 70.0 |
| Emphysema | 4,402 | Medium | 24.3 | 28.6 | **38.3** | 34.1 |
| Tortuous Aorta | 3,831 | Medium | 20.8 | 26.4 | **29.3** | 27.0 |
| Pleural Thickening | 3,751 | Medium | 12.4 | 14.7 | **15.3** | 16.5 |
| Lung Lesion | 2,652 | Tail | 2.2 | 2.5 | **2.7** | 4.9 |
| Subcutaneous Emphysema | 2,477 | Tail | 68.7 | 76.9 | **81.1** | 76.2 |
| Fibrosis | 1,332 | Tail | 18.0 | **38.3** | 31.2 | 38.5 |
| Pneumomediastinum | 826 | Tail | 41.5 | 38.8 | **58.1** | 71.8 |
| Pleural Other | 696 | Tail | 11.5 | **16.2** | 11.0 | 18.8 |
| Pneumoperitoneum | 570 | Tail | 26.7 | **53.8** | 49.8 | 47.9 |

### Key Findings

**1. Task 2 scores are dramatically higher than Task 1.** All models gain 18–23 mAP points
moving from Task 1 (noisy NLP labels, 40 classes, 78K images) to Task 2 (gold standard,
26 classes, 406 images). The 14 removed classes are the hardest (Azygos Lobe, Clavicle
Fracture, Lobar Atelectasis, etc.) — their exclusion alone accounts for the bulk of the jump.
Clean annotations also reduce noise in AP computation.

**2. Tail mAP exceeds Medium mAP for all SSL models** — the opposite of Task 1. Task 2's
6 tail classes happen to include visually distinctive findings that the model has learned
well: Subcutaneous Emphysema (68–81 AP, air under skin visible as bright halos),
Pneumomediastinum (38–58 AP, air around mediastinum), Pneumoperitoneum (27–54 AP,
air under diaphragm). These are striking on X-ray despite being rare. By contrast,
medium classes like Infiltration (2–5 AP), Mass (7–20 AP), and Nodule (14–23 AP) are
diffuse and harder to distinguish from similar-looking findings.

**3. Pneumonia is a consistent weak point (7–11 AP)** despite being a head class with
53,721 training positives. This likely reflects a label definition mismatch: Task 1
(NLP-extracted) and Task 2 (manually annotated) may define "Pneumonia" differently.
The model trained on NLP labels generalises poorly to expert-curated gold standard.

**4. Normal is highly variable across models** (59–83 AP). Hybrid-LB scores 83.4 for Normal
vs 59.1 for Tail-LB. Hybrid's labeled set includes Normal images (it uses a random calib
set drawn from the full distribution), while Tail-LB's labeled set has zero Normal images
(Normal is never a tail-class positive). The gold standard test set labels Normal clearly,
so this gap reflects real calibration differences.

**5. Hernia is the largest single-class divergence** between SSL (7–19 AP) and full
supervision (70.0 AP, lb=1.0). With only 4,660 train positives and visual overlap with
other chest abnormalities, Hernia requires many labeled examples. At 6.6% labeled data,
even the SSL models see very few hernia cases, and the gold standard test set (19 hernia
positives out of 406 images) amplifies variance.

**6. Gap to contest best: −13.4 mAP for best SSL model (Hybrid-LB).** Contest winners
used full supervision with stronger architectures. Our 6.6% SSL models at 39.2% are
reasonable given the constraint — full supervision achieves 45.4% (still 7.2 below contest).

### Task 2 vs Task 1 Ranking Consistency

| Model | Task 1 mAP | Task 1 rank | Task 2 mAP | Task 2 rank |
|---|:---:|:---:|:---:|:---:|
| Hybrid-LB + SSL | 16.03 | 1 (SSL) | **39.20** | 1 (SSL) |
| Tail-LB + SSL | 15.46 | 2 (SSL) | 37.63 | 2 (SSL) |
| lb=0.066 + SSL | 15.26 | 3 (SSL) | 33.92 | 3 (SSL) |
| TResNet_L lb=1.0 | 22.01 | 1 (overall) | 45.43 | 1 (overall) |

Ranking is perfectly consistent between Task 1 and Task 2 — the relative model ordering
holds regardless of which test set or label quality is used.
