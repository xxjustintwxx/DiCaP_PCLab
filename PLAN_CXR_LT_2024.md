# Plan: Adapting DiCaP to CXR-LT 2024

## Goal
Run DiCaP semi-supervised multi-label classification on the CXR-LT 2024 dataset
(chest X-ray, 40 disease classes, long-tailed distribution).

---

## Dataset Overview

| Property | Value |
|----------|-------|
| Source | `/home/share/cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays-2.0.0/cxr-lt-2024/` |
| Images | `/home/share/mimic-cxr-jpg-2.0.0/` |
| Train split | `train_labeled.csv` — 258,871 images |
| Val split | `development_labeled_task1.csv` — 39,293 images (early stopping / model selection during training) |
| Test split | `test_labeled_task1.csv` — 78,946 images (final evaluation — matches competition test phase) |
| Classes | 40 disease labels (columns 7–46 in CSV) |
| Image format | Grayscale JPEG (already handled by `.convert('RGB')` in handlers) |
| Downsampled | `files-512/` subfolder available (recommended for speed) |

### Class Distribution (Positive Label Counts)

Total images: **298,164** (Train: 258,871 | Val: 39,293)

Ratio = positive count / total images. Multi-label — one image can have multiple positive labels.

| Rank | Class | Train+ | Val+ | Total+ | Ratio |
|------|-------|-------:|-----:|-------:|------:|
| 1 | Support Devices | 86,079 | 13,161 | 99,240 | 33.28% |
| 2 | Lung Opacity | 77,482 | 11,731 | 89,213 | 29.92% |
| 3 | Cardiomegaly | 74,738 | 11,583 | 86,321 | 28.95% |
| 4 | Pleural Effusion | 66,401 | 10,430 | 76,831 | 25.77% |
| 5 | Atelectasis | 65,376 | 10,131 | 75,507 | 25.32% |
| 6 | Pneumonia | 46,660 | 7,061 | 53,721 | 18.02% |
| 7 | Edema | 37,256 | 5,946 | 43,202 | 14.49% |
| 8 | Normal | 34,292 | 5,088 | 39,380 | 13.21% |
| 9 | Enlarged Cardiomediastinum | 29,628 | 4,244 | 33,872 | 11.36% |
| 10 | Consolidation | 15,371 | 2,379 | 17,750 | 5.95% |
| 11 | Pneumothorax | 13,858 | 2,342 | 16,200 | 5.43% |
| 12 | Fracture | 11,568 | 1,576 | 13,144 | 4.41% |
| 13 | Infiltration | 10,087 | 1,506 | 11,593 | 3.89% |
| 14 | Rib Fracture | 8,919 | 1,250 | 10,169 | 3.41% |
| 15 | Nodule | 7,531 | 1,119 | 8,650 | 2.90% |
| 16 | Mass | 5,288 | 789 | 6,077 | 2.04% |
| 17 | Calcification of the Aorta | 4,239 | 594 | 4,833 | 1.62% |
| 18 | Hernia | 3,986 | 674 | 4,660 | 1.56% |
| 19 | Emphysema | 3,661 | 741 | 4,402 | 1.48% |
| 20 | Adenopathy | 3,409 | 477 | 3,886 | 1.30% |
| 21 | Tortuous Aorta | 3,336 | 495 | 3,831 | 1.28% |
| 22 | Pleural Thickening | 3,272 | 479 | 3,751 | 1.26% |
| 23 | Granuloma | 2,965 | 383 | 3,348 | 1.12% |
| 24 | Fissure | 2,803 | 351 | 3,154 | 1.06% |
| 25 | Lung Lesion | 2,338 | 314 | 2,652 | 0.89% |
| 26 | Subcutaneous Emphysema | 2,046 | 431 | 2,477 | 0.83% |
| 27 | Tuberculosis | 2,078 | 377 | 2,455 | 0.82% |
| 28 | Pulmonary Embolism | 1,631 | 304 | 1,935 | 0.65% |
| 29 | Fibrosis | 1,169 | 163 | 1,332 | 0.45% |
| 30 | Pulmonary Hypertension | 903 | 119 | 1,022 | 0.34% |
| 31 | Kyphosis | 778 | 112 | 890 | 0.30% |
| 32 | Pneumomediastinum | 704 | 122 | 826 | 0.28% |
| 33 | Infarction | 727 | 96 | 823 | 0.28% |
| 34 | Hydropneumothorax | 646 | 128 | 774 | 0.26% |
| 35 | Pleural Other | 616 | 80 | 696 | 0.23% |
| 36 | Pneumoperitoneum | 516 | 54 | 570 | 0.19% |
| 37 | Azygos Lobe | 199 | 20 | 219 | 0.07% |
| 38 | Round(ed) Atelectasis | 172 | 46 | 218 | 0.07% |
| 39 | Clavicle Fracture | 168 | 19 | 187 | 0.06% |
| 40 | Lobar Atelectasis | 129 | 26 | 155 | 0.05% |

**Long-tail gap:** Support Devices (33.28%) vs Lobar Atelectasis (0.05%) — a **640× imbalance** between the most and least frequent class.

---

## Step-by-Step Plan

---

### Step 1 — Write the data formatting script

**File to create:** `data/cxr/format_cxr2024.py`

This mirrors `data/voc/format_voc2012.py`. It reads the CXR-LT 2024 CSV files
and produces the four `.npy` files that DiCaP's `load_data()` expects.

```
formatted_train_images.npy   shape: (258871,)   dtype: str  — absolute image paths
formatted_train_labels.npy   shape: (258871, 40) dtype: float32
formatted_val_images.npy     shape: (~39293,)    dtype: str
formatted_val_labels.npy     shape: (~39293, 40) dtype: float32
```

**Key design decision — absolute paths in npy:**
Each path stored in `formatted_*_images.npy` will be the **full absolute path**
to the image (e.g. `/home/share/mimic-cxr-jpg-2.0.0/files/p10/.../xxx.jpg`).
Because Python's `os.path.join(base, '/absolute/path')` returns `/absolute/path`,
the handler will resolve correctly regardless of what `--dataset_dir` is set to.
This avoids any need to change the data loading pipeline.

**To use the 512px downsampled images** (recommended — ~10× faster data loading,
images are resized to 224×224 anyway), replace `files/` with `files-512/` in
the path construction inside the script.

**Script logic (no code — implement similarly to `data/voc/format_voc2012.py`):**
- Read `train_labeled.csv` and `development_labeled_task1.csv` with pandas
- Extract the `fpath` column, prepend `/home/share/mimic-cxr-jpg-2.0.0/` to make absolute paths → save as `formatted_train_images.npy` / `formatted_val_images.npy`
- Extract the 40 disease label columns (everything after `fpath`) as float32 → save as `formatted_train_labels.npy` / `formatted_val_labels.npy`
- To use 512px images, replace `files/` with `files-512/` in the paths

**Run once to generate npy files:**
```bash
mkdir -p data/cxr
conda run -n dicap python data/cxr/format_cxr2024.py
```

---

### Step 2 — Add CXR handler to `lib/dataset/handlers.py`

Append a new `CXR_handler` class at the bottom of the file.
It is identical to `COCO2014_handler` — open image with PIL, call `.convert('RGB')`
(this already handles grayscale JPEG correctly), apply transform, return `(image, label, index)`.

---

### Step 3 — Register CXR in `lib/dataset/get_dataset.py`

Add `CXR_handler` to the import line at the top, and add `'cxr': CXR_handler` to
`HANDLER_DICT`. No other changes needed — `dataset_dir = args.dataset_dir` (already
fixed earlier) passes through correctly, and absolute paths in the npy handle the rest.

---

### Step 4 — Register CXR class count in `warm_up.py`, `main.py`, `fine_tune.py`

Each script has a `NUM_CLASS` dict near the top. Add `'cxr': 40` to all three files.
This tells the model how many output nodes and class-query embeddings to create.

---

### Step 5 — Update `script/run.sh`

Change `dataset_dir` to `'./data/cxr'` and `dataset_name` to `'cxr'`.
Set `--output cxr_ours` so results don't mix with VOC outputs.
Keep `lb_ratio` values you want to test (e.g. `0.05 0.1`).
All other flags (net, loss, epochs, lr) stay the same.

---

### Step 6 — Write a per-class mAP evaluation script

**File to create:** `evaluate_cxr.py`

`function_mAP` in `lib/utils/helper.py` already computes per-class AP internally but
discards the list and only returns the mean. For CXR-LT, per-class AP is the critical
metric — overall mAP can look fine while the model completely fails on rare classes
(Tuberculosis, Pulmonary Embolism, Hydropneumothorax).

The script should:
1. Load a saved checkpoint (`best_model.pth.tar`) from a specified output directory
2. Run inference on the val set (`development_labeled_task1.csv`)
3. Compute AP per class using sklearn's `average_precision_score`
4. Print a table of all 40 classes sorted by their training frequency (so the long-tail
   behaviour is immediately visible — common classes at top, rare at bottom)
5. Also report mean AP (head classes only), mean AP (tail classes only), and overall mAP
   to quantify the long-tail gap

This script is run once after training, not during training. It takes `--checkpoint`,
`--dataset_dir`, and optionally a `--tail_threshold` (minimum training samples to be
considered a "head" class).

---

### Step 7 — Backbone choice

**For a pipeline check:** use `--net resnet50` (default, fast to run)

**For real experiments:** switch to `--net tresnet_l`

TResNet-L is already supported in DiCaP (no code changes needed) and is the backbone
used in the original ML-Decoder paper that DiCaP's architecture is built on. It uses
in-place attention mechanisms better suited to multi-label spatial features than a plain
ResNet, which matters for CXR where different diseases occupy different spatial regions.

DenseNet121 (the traditional CXR/CheXNet backbone) is not in DiCaP's supported list.
Adding it is possible via timm but adds complexity — TResNet-L is the pragmatic choice.

Backbone comparison (all already supported, just change `--net`):

| Backbone | Flag | Recommendation |
|----------|------|----------------|
| ResNet50 | `resnet50` | Pipeline verification only |
| ResNet101 | `resnet101` | Easy improvement, same architecture |
| TResNet-L | `tresnet_l` | **Use this for all real experiments** |
| TResNet-XL | `tresnet_xl` | If GPU memory allows (32GB on RTX 5090: yes) |

---

## Three-Phase Training — How They Differ

### Why warm_up only trains on 80% of labeled data

The labeled set is split 80/20 at the start and **this split is fixed across all three phases**:

- **80%** → supervised training (`lb_train_dataset_train`)
- **20%** → held-out estimation set (`lb_train_dataset_valid`)

The 20% holdout is intentionally kept out of the supervised loss so that in `main.py` it can serve as an **unbiased calibration probe**. DiCaP estimates per-class confidence distributions (`global_pos_ratio`) by running the warmup model over this holdout and measuring how often each confidence bin actually contains a true positive. If warmup had trained on those labels, the model would be overfit to them and its scores on that 20% would be inflated — making the calibration unreliable. Keeping it unseen during supervised training ensures the confidence estimates are honest.

---

### Phase comparison

|   | `warm_up.py` | `main.py` | `fine_tune.py` |
|---|--------------|-----------|----------------|
| **Starts from** | Random init | warm_up checkpoint | main best checkpoint |
| **Backbone**    | Trainable | Trainable | **Frozen** |
| **Supervised loss** | 80% labeled (`lb_train_loader`) | 80% labeled (`lb_train_loader`) | **20% holdout** (`lb_valid_loader`) |
| **Calibration**     | — | 20% holdout → `global_pos_ratio` bins | — |
| **Pseudo-labels**   | No | Yes — dual thresholds + weight matrix | **No** |
| **Contrastive loss (UCL)** | Yes — on 20% holdout + unlabeled | Yes — on uncertain unlabeled only | **No** |
| **Total loss**       | `Lx + 0.1·UCL` | `Lx + Lu·weight + UCL` | `Lx` only |
| **Epochs (default)** | 12 | 40 | 20 |
| **Purpose**          | Learn backbone representations | Semi-supervised refinement with DiCaP | Polish decoder with held-out labels |

### How the 20% holdout is used across phases

```
warm_up  →  20% holdout used for contrastive loss only (no supervised gradient)
              ↓  model has never fitted these labels → confidence scores are honest
main     →  20% holdout used to compute calibration bins (global_pos_ratio)
              ↓  also goes into combined_unlabeled_dataset → UCL contrastive loss
fine_tune → 20% holdout used as the supervised training set for the frozen decoder
              (calibration is done; these labels are now "spent" for supervised polish)
```

### What fine_tune actually does

`fine_tune.py` freezes the backbone and trains **only the ML-Decoder** with a pure supervised loss on the 20% holdout. The learning rate scheduler is paced to `len(lb_valid_loader)` steps. There is no pseudo-labeling, no contrastive loss, and the 80% labeled training set is not touched. This phase polishes the class-query embeddings and decoder attention weights using labels the backbone has never been directly supervised on.

---

## Summary of All Files Changed

| File | Change |
|------|--------|
| `data/cxr/format_cxr2024.py` | **Created** — reads CSVs, saves 4 npy files with absolute 512px image paths |
| `lib/dataset/handlers.py` | **Added** `CXR_handler` class at bottom |
| `lib/dataset/get_dataset.py` | **Added** `CXR_handler` to import and `'cxr': CXR_handler` to HANDLER_DICT |
| `warm_up.py` | **Added** `'cxr': 40` to `NUM_CLASS` and `'cxr'` to argparse choices |
| `main.py` | **Added** `'cxr': 40` to `NUM_CLASS` and `'cxr'` to argparse choices |
| `fine_tune.py` | **Added** `'cxr': 40` to `NUM_CLASS` and `'cxr'` to argparse choices |
| `script/run.sh` | **Updated** `dataset_name='cxr'`, `dataset_dir='./data'`, `--output cxr_ours` |
| `evaluate_cxr.py` | **Created** — per-class AP evaluation script |

---

## Commands

### Step 1 — Generate npy files (one-time)
```bash
python data/cxr/format_cxr2024.py
```

### Step 2 — Run full training
```bash
bash script/run.sh
```
Runs warm_up (12 epochs) → main (40 epochs) → fine_tune (20 epochs).
Checkpoints saved to `output/cxr_ours/cxr/0.05/`.

### Step 3 — Per-class mAP evaluation (after training)
```bash
python evaluate_cxr.py \
  --checkpoint output/cxr_ours/cxr/0.05/fine_tune/best_model.pth.tar \
  --dataset_dir ./data
```
Prints AP per class sorted head→tail, plus overall / head-only / tail-only mAP.

---

---

## Experiment: Tail-Class-Driven Labeled Split (test_labeled_task1.csv)

### Motivation

The current setting uses `lb_ratio` to randomly sample a fraction of `train_labeled.csv` as labeled data. This means rare tail classes may get very few (or zero) labeled examples at low ratios, making semi-supervised training on tail classes unreliable.

This new setting replaces random ratio-based labeling with a **semantically-driven split** derived from the test set: any test image that contains at least one positive label among the 16 rarest classes is treated as labeled; all other test images become unlabeled. This guarantees every labeled sample contributes signal for at least one hard tail class, while the much larger pool of head-class-only images forms the unlabeled set for pseudo-labeling.

---

### Labeled Set Coverage (tail_lb split, 13,921 images)

Coverage = positives in labeled set / total train positives.

| Class | LB+ | Train Total+ | Coverage | Zone |
|-------|----:|-------------:|---------:|:----:|
| Support Devices | 4,998 | 86,079 | 5.8% | Head |
| Lung Opacity | 5,275 | 77,482 | 6.8% | Head |
| Cardiomegaly | 3,763 | 74,738 | 5.0% | Head |
| Pleural Effusion | 4,462 | 66,401 | 6.7% | Head |
| Atelectasis | 4,091 | 65,376 | 6.3% | Head |
| Pneumonia | 2,125 | 46,660 | 4.6% | Head |
| Edema | 1,707 | 37,256 | 4.6% | Head |
| **Normal** | **0** | **34,292** | **0.0%** | Head |
| Enlarged Cardiomediastinum | 2,348 | 29,628 | 7.9% | Head |
| Consolidation | 1,009 | 15,371 | 6.6% | Medium |
| Pneumothorax | 2,266 | 13,858 | 16.4% | Medium |
| Fracture | 903 | 11,568 | 7.8% | Medium |
| Infiltration | 532 | 10,087 | 5.3% | Medium |
| Rib Fracture | 555 | 8,919 | 6.2% | Medium |
| Nodule | 672 | 7,531 | 8.9% | Medium |
| Mass | 619 | 5,288 | 11.7% | Medium |
| Calcification of the Aorta | 181 | 4,239 | 4.3% | Medium |
| Hernia | 303 | 3,986 | 7.6% | Medium |
| Emphysema | 474 | 3,661 | 12.9% | Medium |
| Adenopathy | 394 | 3,409 | 11.6% | Medium |
| Tortuous Aorta | 165 | 3,336 | 4.9% | Medium |
| Pleural Thickening | 354 | 3,272 | 10.8% | Medium |
| Granuloma | 283 | 2,965 | 9.5% | Medium |
| Fissure | 267 | 2,803 | 9.5% | Medium |
| Lung Lesion | 2,338 | 2,338 | **100%** | Tail |
| Tuberculosis | 2,078 | 2,078 | **100%** | Tail |
| Subcutaneous Emphysema | 2,046 | 2,046 | **100%** | Tail |
| Pulmonary Embolism | 1,631 | 1,631 | **100%** | Tail |
| Fibrosis | 1,169 | 1,169 | **100%** | Tail |
| Pulmonary Hypertension | 903 | 903 | **100%** | Tail |
| Kyphosis | 778 | 778 | **100%** | Tail |
| Infarction | 727 | 727 | **100%** | Tail |
| Pneumomediastinum | 704 | 704 | **100%** | Tail |
| Hydropneumothorax | 646 | 646 | **100%** | Tail |
| Pleural Other | 616 | 616 | **100%** | Tail |
| Pneumoperitoneum | 516 | 516 | **100%** | Tail |
| Azygos Lobe | 199 | 199 | **100%** | Tail |
| Round(ed) Atelectasis | 172 | 172 | **100%** | Tail |
| Clavicle Fracture | 168 | 168 | **100%** | Tail |
| Lobar Atelectasis | 129 | 129 | **100%** | Tail |

**Key observations:**
- All 16 tail classes: 100% coverage — every positive is in the labeled set, none left in unlabeled pool
- **Normal: 0 positives** — labeled set has zero Normal images; model has no direct supervision for this class
- Head/medium classes: only 4–17% coverage — 83–96% of their positives sit in the 244K unlabeled pool with labels hidden, producing only negative pseudo-label reinforcement for those classes

---

### Split Statistics (train_labeled.csv — 258,871 images)

| Subset | Condition | Count | % |
|--------|-----------|------:|--:|
| **Labeled** | Has ≥ 1 tail-class positive | **13,921** | **5.38%** |
| **Unlabeled** | No tail-class positive | 244,950 | 94.62% |
| Total | — | 258,871 | 100% |

Val (early stopping during training): `development_labeled_task1.csv` — 39,293 images (unchanged).
Test (final evaluation): `test_labeled_task1.csv` — 78,946 images.

---

### Implementation Steps

#### Step A — New format script: `data/cxr/format_cxr2024_tail_lb.py`

Create a new script (separate from `format_cxr2024.py` — do not modify the existing one) that:

1. Reads `train_labeled.csv` (258,871 images) and `development_labeled_task1.csv` (39,293 images)
2. Defines `TAIL_CLASSES` — the 16 classes listed above
3. For the train set, computes a boolean mask: `has_tail = (df[TAIL_CLASSES] == 1).any(axis=1)`
4. Splits into `lb_df` (has_tail=True, 13,921 rows) and `ub_df` (has_tail=False, 244,950 rows)
5. Saves six `.npy` files into `data/cxr_tail_lb/`:

```
formatted_train_images.npy       shape: (13921,)    dtype: str    — labeled (tail-positive) paths
formatted_train_labels.npy       shape: (13921, 40) dtype: float32
formatted_unlabeled_images.npy   shape: (244950,)   dtype: str    — unlabeled (no tail) paths
formatted_unlabeled_labels.npy   shape: (244950, 40) dtype: float32
formatted_val_images.npy         shape: (39293,)    dtype: str    — dev set, for early stopping
formatted_val_labels.npy         shape: (39293, 40) dtype: float32
formatted_test_images.npy        shape: (78946,)    dtype: str    — test set, for final eval
formatted_test_labels.npy        shape: (78946, 40) dtype: float32
formatted_full_train_labels.npy  shape: (258871, 40) dtype: float32 — full train for class-freq sort
```

All paths use `files-512/` (same `USE_512 = True` logic as `format_cxr2024.py`).

Run once:
```bash
mkdir -p data/cxr_tail_lb
conda run -n dicap python data/cxr/format_cxr2024_tail_lb.py
```

---

#### Step B — Extend `lib/dataset/get_dataset.py` to support pre-split unlabeled npy

Add a branch in `get_datasets()`: if `formatted_unlabeled_images.npy` exists in the dataset directory, load it directly as the unlabeled set instead of splitting by `lb_ratio`. Set `lb_ratio=0.0` in this mode so the entire `formatted_train_*` becomes the labeled set without any further splitting.

Minimal change — add after `source_data = load_data(args.dataset_dir)`:

```python
unlabeled_path = os.path.join(args.dataset_dir, 'formatted_unlabeled_images.npy')
if os.path.exists(unlabeled_path):
    # Pre-split mode: train npy = labeled set, unlabeled npy = unlabeled set
    lb_train_imgs   = source_data['train']['images']
    lb_train_labels = source_data['train']['labels']
    ub_train_imgs   = np.load(unlabeled_path, allow_pickle=True)
    ub_train_labels = np.load(os.path.join(args.dataset_dir,
                              'formatted_unlabeled_labels.npy'))
else:
    # Original lb_ratio split (random)
    n_train = len(source_data['train']['labels'])
    n_lb = int(args.lb_ratio * n_train)
    indices = torch.randperm(n_train).tolist()
    lb_idxs, ub_idxs = indices[:n_lb], indices[n_lb:]
    lb_train_imgs   = source_data['train']['images'][lb_idxs]
    lb_train_labels = source_data['train']['labels'][lb_idxs]
    ub_train_imgs   = source_data['train']['images'][ub_idxs]
    ub_train_labels = source_data['train']['labels'][ub_idxs]
```

The rest of `get_datasets()` (building handlers, returning datasets) is unchanged.

---

#### Step C — New run script: `script/run_tail_lb.sh`

Copy `script/run.sh` and change three lines:

```bash
dataset_dir='./data/cxr_tail_lb'
output='cxr_tail_lb'

for lb_ratio in 1.0   # lb_ratio is ignored in pre-split mode; kept for checkpoint path
```

The `evaluate_cxr.py` and `plot_*.py` calls at the end update their `--checkpoint` and `--dataset_dir` paths accordingly (use `${dataset_dir}` variable — already parameterized in run.sh).

---

#### Step D — (Optional) Log labeled/unlabeled counts at startup

In `warm_up.py` and `main.py`, after `get_datasets()` returns, add a print:

```python
print(f'[data] labeled={len(lb_train_dataset)}, unlabeled={len(ub_train_dataset)}, val={len(val_dataset)}')
```

This confirms the pre-split mode is active (expected: 13921 / 244950 / 39293).

---

### Summary of New Files / Changes

| File | Action |
|------|--------|
| `data/cxr/format_cxr2024_tail_lb.py` | **Create** — splits test set by tail-class presence, writes 6 npy files to `data/cxr_tail_lb/` |
| `lib/dataset/get_dataset.py` | **Edit** — add pre-split unlabeled npy branch (≈15 lines) |
| `script/run_tail_lb.sh` | **Create** — copy of run.sh with `dataset_dir` and `output` updated |

No changes to `warm_up.py`, `main.py`, `fine_tune.py`, or the handler/num_class registrations — the dataset is still `'cxr'` with 40 classes.

---

## Notes

- **Use `files-512/`** in the format script if disk I/O is slow — it cuts image load
  time by ~10× with no accuracy loss (transforms resize to 224×224 anyway).
- **lb_ratio meaning with 258K images:** 5% = ~12,943 labeled, 95% = ~245,928 unlabeled.
  Even 1% gives ~2,588 labeled samples — large relative to VOC's 285.
- **Long-tailed classes** like Tuberculosis, Pulmonary Embolism will have very few
  positives at low lb_ratio — mAP on rare classes will be the key metric to watch.
- **Val vs Test split:** `development_labeled_task1.csv` (39K) is used as the **val set**
  during training (early stopping, model selection). `test_labeled_task1.csv` (78K) is
  the **final test set** used for all reported evaluation numbers — this matches the
  competition's test phase. The two sets are always kept separate to avoid test leakage.
  Tasks 2 and 3 use different label subsets — check the CXR-LT 2024 paper for details.

---

## Experiment: Hybrid Split — Calibration-Corrected Tail-Class Training

### Motivation

The `tail_lb` experiment revealed two compounding problems:

1. **Poisoned unlabeled pool**: 244K unlabeled images have zero tail-class positives → pseudo-labels
   flood the model with "tail = absent" signal, overwhelming the supervised tail signal.
2. **Miscalibrated DiCaP machinery**: DiCaP's two internal mechanisms — the confidence weight
   histogram (`update_weight`) and per-class dynamic thresholds (`dynamic_threshold_generate`) —
   are both estimated from the 20% holdout of the labeled set. In `tail_lb`, that holdout is
   also tail-biased (same pool), so the calibration is wrong for the actual unlabeled pool.

This experiment keeps the tail-positive supervised training set intact but **replaces the 20%
calibration holdout** with images drawn from the full natural long-tail distribution (all 258K).
This corrects DiCaP's internal threshold and weight machinery without changing the supervised
training signal.

---

### Proposed Split Design

#### Sampling strategy

1. **Sample calibration set** from the full 258K training images (random, reflects true distribution).
   This ensures all 40 classes appear with their real prevalence — including Normal (13.2%),
   head classes, medium classes, and rare tail classes at <1%.
2. **Supervised training set** = all tail-positive images that were **not drawn** into the
   calibration set (to avoid the same images appearing in both sets).
3. **Unlabeled pool** = all remaining images (neither calibration nor supervised training).

#### Computing the 20/80 ratio

Target: `N_calib / (N_calib + N_train) = 0.2`, equivalently `N_calib = 0.25 × N_train`.

- `N_tail_total` = 13,921 tail-positive images
- Expected overlap when sampling N_calib from 258K: `overlap ≈ N_calib × (13921 / 258871) ≈ N_calib × 0.054`
- `N_train ≈ N_tail_total − overlap ≈ 13,921 − 0.054 × N_calib`
- Solving `N_calib = 0.25 × (13,921 − 0.054 × N_calib)` → **N_calib ≈ 3,434**

#### Actual allocation (seed=42, `format_cxr2024_hybrid_lb.py`)

```
Sample 3,434 from full 258K (natural distribution) → calibration set
  ├── 185 of those are tail-positive (5.4% — matches natural tail prevalence 5.38%)
  └── 3,249 non-tail images

Supervised training set:  13,736  (tail-positive, not in calibration)
Unlabeled pool:          241,701

Total labeled = 3,434 + 13,736 = 17,170
Calib ratio  = 3,434 / 17,170 = 20.0%  ✓
```

| Subset | Count | Notes |
|--------|------:|-------|
| Supervised training (80%) | **13,736** | Tail-positive, not in calib; rarest=Normal (0), most common=Lung Opacity (5,215) |
| Calibration set (20%) | **3,434** | Natural distribution; incl. 185 tail-positive; rarest=Round(ed) Atelectasis (0), most common=Support Devices (1,112) |
| Unlabeled pool | **241,701** | Non-tail, non-calib; rarest=Azygos Lobe (0), most common=Support Devices (80,031) |
| Val set | 39,293 | `development_labeled_task1.csv` — unchanged |
| Test set | 78,946 | `test_labeled_task1.csv` — unchanged |

**Notes on zero counts:**
- Normal = 0 in supervised training — expected; Normal images cannot be tail-positive
- Azygos Lobe = 0 in unlabeled pool — expected; all Azygos Lobe positives are in the tail-positive labeled set (it is a TAIL_CLASS)
- Round(ed) Atelectasis = 0 in calib — sampling variance; expected count ≈ 2.3 in 3,434 images

---

### What this fixes in DiCaP's machinery

| DiCaP mechanism | `tail_lb` problem | After this fix |
|---|---|---|
| `update_weight()` — calibration histogram | Computed from tail-biased 20% → `global_pos_ratio[low bins]` artificially high for tail | Computed from full-distribution 20% → `global_pos_ratio` reflects true `P(positive \| score)` |
| `dynamic_threshold_generate()` — per-class thresholds | Derived from tail-positive images only → thresholds for all classes biased | Derived from natural-distribution calibration → thresholds meaningful for unlabeled pool |
| `fine_tune.py` — supervised loss on 20% holdout | 20% holdout is tail-biased → decoder fine-tuned on skewed distribution | 20% holdout has natural distribution → decoder polished on representative labels |

---

### Known remaining problem: the poisoned unlabeled pool

Correcting the calibration does **not** fix the fundamental structural problem. With correct
calibration, `global_pos_ratio[bin_for_0.0-0.1]` ≈ 0.01 for tail classes (correctly reflecting
that most unlabeled images truly don't have tail positives). This means:

- Unlabeled image scores 0.05 on a tail class → pseudo-label = -1
- `sample_weights[pseudo==-1] = 1 − 0.01 = 0.99` — very high-confidence negative
- **Correct calibration makes negative pseudo-label weight stronger, not weaker**

The update ratio per epoch remains heavily skewed against tail classes:

```
Supervised positive updates:  ~13,736 / lb_bs  ≈   215 batches  (tail class signal)
Unlabeled negative updates:   ~241K   / ub_bs  ≈ 7,543 batches  (weight ≈ 0.99 for tail)
```

**We run this experiment anyway** to isolate which failure mode dominates. If head/medium mAP
improves but tail mAP stays near zero, it confirms the poisoned pool is the irreducible
bottleneck and not the calibration mismatch.

If tail mAP remains poor, the next mitigation is to mask tail classes out of `Lu` entirely —
since all tail pseudo-labels in the unlabeled pool are -1 and only hurt, removing them from `Lu`
lets the supervised loss be the sole driver for tail classes.

---

### Implementation

#### Step A — New format script: `data/cxr/format_cxr2024_hybrid_lb.py`

Separate from existing format scripts. Logic:

1. Read `train_labeled.csv` (258,871 rows)
2. Compute `has_tail` mask: any of the 16 tail-class columns == 1
3. Random sample 3,390 rows from the full 258,871 → `calib_df`
4. `tail_df` = rows where `has_tail=True` AND not in `calib_df` → supervised training
5. Remaining rows (not in calib, not in tail) → unlabeled pool

Save to `data/cxr_hybrid_lb/`:

```
formatted_train_images.npy      (~13737,)    — tail-positive, supervised training
formatted_train_labels.npy      (~13737, 40)
formatted_calib_images.npy      (3390,)      — natural distribution, calibration only
formatted_calib_labels.npy      (3390, 40)
formatted_unlabeled_images.npy  (~241744,)   — remaining images, unlabeled pool
formatted_unlabeled_labels.npy  (~241744, 40)
formatted_val_images.npy        (39293,)     — dev set (unchanged)
formatted_val_labels.npy        (39293, 40)
formatted_test_images.npy       (78946,)     — test set (unchanged)
formatted_test_labels.npy       (78946, 40)
formatted_full_train_labels.npy (258871, 40) — full train, for class-frequency zone sorting
```

Run once:
```bash
mkdir -p data/cxr_hybrid_lb
conda run -n dicap python data/cxr/format_cxr2024_hybrid_lb.py
```

---

#### Step B — Modify `main.py` to load calibration set from npy if present

The existing random 80/20 `random_split` of `lb_train_dataset` (lines 206–216 in `main.py`)
is replaced by a conditional branch. If `formatted_calib_images.npy` exists, skip `random_split`
and load the pre-defined calibration set directly.

```python
calib_path = os.path.join(args.dataset_dir, 'formatted_calib_images.npy')
if os.path.exists(calib_path):
    # Hybrid split: pre-defined calibration set from natural distribution
    calib_imgs   = np.load(calib_path, allow_pickle=True)
    calib_labels = np.load(os.path.join(args.dataset_dir, 'formatted_calib_labels.npy'))
    lb_train_dataset_train = lb_train_dataset   # full tail-positive set → supervised
    lb_train_dataset_valid = data_handler(calib_imgs, calib_labels, dataset_dir,
                                          transform=train_transform)
    print(f'[hybrid-split] supervised={len(lb_train_dataset_train)}, calib={len(lb_train_dataset_valid)}')
else:
    # Original random 80/20 split
    train_len = int(len(lb_train_dataset) * args.percent)
    valid_len  = len(lb_train_dataset) - train_len
    lb_train_dataset_train, lb_train_dataset_valid = random_split(
        lb_train_dataset, [train_len, valid_len],
        generator=torch.Generator().manual_seed(args.split_seed))
```

`data_handler` and `dataset_dir` are already available in `main_worker()` — pull them from
`args.dataset_name` and `args.dataset_dir` using the existing `HANDLER_DICT`. The rest of
`main.py` is unchanged: `lb_train_dataset_valid` flows into `update_weight()`,
`dynamic_threshold_generate()`, and `combined_unlabeled_dataset` exactly as before.

---

#### Step C — New run script: `script/run_hybrid_lb.sh`

Copy `script/run_tail_lb.sh`, change:
```bash
dataset_dir='./data/cxr_hybrid_lb'
output='cxr_hybrid_lb'
```

---

### Summary of new files / changes

| File | Action |
|------|--------|
| `data/cxr/format_cxr2024_hybrid_lb.py` | **Create** — samples 3,434-image natural-distribution calibration set from full 258K (seed=42), writes 10 npy files to `data/cxr_hybrid_lb/` |
| `main.py` | **Edit** — add calib npy branch before `random_split` (≈15 lines, backward compatible — no effect on existing runs) |
| `script/run_hybrid_lb.sh` | **Create** — copy of `run_tail_lb.sh` with `dataset_dir` and `output` updated |

---

## Future Improvements for Tail-Class Performance

### Why the tail_lb split failed (and what not to try)

The `run_tail_lb.sh` experiment (overall mAP 6.58 vs baseline 22.01) demonstrated that
concentrating all tail positives in the labeled set is counterproductive. By construction,
the 244,950 unlabeled images had zero tail-class positives — making DiCaP's pseudo-labeling
produce 18× more "tail = absent" signal than the 13,921 labeled images produced "tail = present".

**Using the test set as the unlabeled pool is not viable** — it breaks benchmark comparability
and we must evaluate on `test_labeled_task1.csv` to match the competition metric.

The fundamental constraint: **any improvement must keep `train_labeled.csv` as the full
training pool and `test_labeled_task1.csv` as the held-out evaluation set.**

### Viable approaches (ranked by implementation cost)

#### 1. Tail-class loss upweighting in ASL (low cost)

Weight each class's loss contribution by inverse frequency during the supervised loss `Lx`.
The 4 ultra-rare classes (Lobar Atelectasis 129, Clavicle Fracture 168, etc.) would receive
~670× more gradient than Support Devices. Apply only to labeled loss — pseudo-label loss
weights should remain uniform or use a softer schedule to avoid amplifying wrong pseudo-labels.

ASL already accepts per-class weights; add a `--class_weights freq` flag that computes
`w_c = (max_freq / freq_c) ^ gamma` with `gamma ∈ [0.5, 1.0]`.

#### 2. Class-balanced sampling of the labeled loader (low cost)

Oversample labeled images that contain tail positives so each epoch sees tail classes
proportionally. Keeps the same data — just changes sampling frequency. Can be combined
with upweighting. DiCaP's `lb_train_loader` uses a standard `DataLoader`; replace with a
`WeightedRandomSampler` where each image's weight is the sum of its inverse-frequency
class weights.

#### 3. Logit adjustment at inference (zero training cost)

Subtract `log p(y=1 | class)` (the training prior) from each class's logit before sigmoid
at inference time. This corrects the model's learned bias toward predicting common classes.
Can be applied to existing checkpoints — no retraining. Formula:

```
adjusted_logit_c = logit_c - τ · log(freq_c / max_freq)
```

where `τ` is a temperature (try 1.0). Implement as a flag in `evaluate_cxr.py`.

#### 4. Extend epoch budget for existing TResNet_L checkpoint (zero code cost)

The TResNet_L lb=1.0 fine-tune mAP was still climbing at the last epoch (ep 16).
Resume from `output/cxr_ours/cxr/tresnet_l/1.0/fine_tune/best_model.pth.tar` and run
10–15 more fine-tune epochs. Expected gain: +0.5–1.5 mAP at minimal GPU cost.

#### 5. Irreducible floor for ultra-rare classes

The 4 classes with ≤219 total train positives (Azygos Lobe, Round Atelectasis, Clavicle
Fracture, Lobar Atelectasis) scored near 0 AP even at lb_ratio=1.0 with TResNet_L.
With 50–200 positives across 258K images (~0.06%), no standard training strategy is
expected to reliably detect these. Data augmentation specific to these classes or
external pre-training data would be needed to make progress here.

---

## Task 2 — Gold Standard Evaluation

### Dataset Overview

CXR-LT 2024 has two evaluation tracks. Task 1 evaluates on the full NLP-labeled test set
(78,946 images, 40 classes). Task 2 evaluates on a **manually annotated gold standard subset**
designed to remove label noise and enable reliable per-class measurement.

| Property | Value |
|----------|-------|
| Source CSV (test) | `test_labeled_task2.csv` — **406 images**, 26 classes |
| Source CSV (dev) | `development_labeled_task2.csv` — 39,293 images, 26 classes |
| Classes | 26 of the 40 Task 1 classes (14 ultra-rare Task 1 classes excluded) |
| Image paths | Same MIMIC-CXR pool as Task 1 |
| Annotation | Manual — higher precision than NLP-derived Task 1 labels |
| Contest best mAP | **52.6%** (overall; per-zone breakdown not publicly available) |

The 26 Task 2 classes (in CSV column order):
```
Atelectasis, Calcification of the Aorta, Cardiomegaly, Consolidation,
Edema, Emphysema, Enlarged Cardiomediastinum, Fibrosis, Fracture,
Hernia, Infiltration, Lung Lesion, Lung Opacity, Mass, Normal,
Nodule, Pleural Effusion, Pleural Other, Pleural Thickening,
Pneumomediastinum, Pneumonia, Pneumoperitoneum, Pneumothorax,
Subcutaneous Emphysema, Support Devices, Tortuous Aorta
```

**Important ordering note:** Task 2 CSV has Normal (col 14) before Nodule (col 15), while the
model's alphabetical `LABEL_COLS` has Nodule (idx 23) before Normal (idx 24). The index
mapping handles this swap explicitly.

---

### Zone Structure (Task 2, 26 classes)

Thresholds based on train+val combined frequency: Head >10% (>29,816), Tail <1% (<2,982).

| Zone | Count | Classes |
|------|------:|---------|
| Head | 9 | Support Devices, Lung Opacity, Cardiomegaly, Pleural Effusion, Atelectasis, Pneumonia, Edema, Normal, Enlarged Cardiomediastinum |
| Medium | 11 | Consolidation, Pneumothorax, Fracture, Infiltration, Nodule, Mass, Calcification of the Aorta, Hernia, Emphysema, Tortuous Aorta, Pleural Thickening |
| Tail | 6 | Lung Lesion, Subcutaneous Emphysema, Fibrosis, Pleural Other, Pneumomediastinum, Pneumoperitoneum |

**Tail anomaly:** Task 2's 6 tail classes include visually distinctive air/gas pattern findings
(Subcutaneous Emphysema, Pneumomediastinum, Pneumoperitoneum) that the model detects well despite
rarity, pulling tail mAP **above** medium mAP in all four models evaluated — the opposite of the
Task 1 zone ordering.

---

### Model Index Mapping

The model outputs 40 classes. Task 2 predictions are extracted by slicing
`outputs[:, MODEL_TASK2_INDICES]` where:

```python
MODEL_TASK2_INDICES = [
    1,   # Atelectasis
    3,   # Calcification of the Aorta
    4,   # Cardiomegaly
    6,   # Consolidation
    7,   # Edema
    8,   # Emphysema
    9,   # Enlarged Cardiomediastinum
    10,  # Fibrosis
    12,  # Fracture
    14,  # Hernia
    17,  # Infiltration
    20,  # Lung Lesion
    21,  # Lung Opacity
    22,  # Mass
    24,  # Normal  ← model idx 24, Task 2 col 14 (swap with Nodule)
    23,  # Nodule  ← model idx 23, Task 2 col 15 (swap with Normal)
    25,  # Pleural Effusion
    26,  # Pleural Other
    27,  # Pleural Thickening
    28,  # Pneumomediastinum
    29,  # Pneumonia
    30,  # Pneumoperitoneum
    31,  # Pneumothorax
    36,  # Subcutaneous Emphysema
    37,  # Support Devices
    38,  # Tortuous Aorta
]
```

---

### Scripts Created

| File | Purpose |
|------|---------|
| `data/cxr/format_cxr2024_task2.py` | One-time: formats `test_labeled_task2.csv` and `development_labeled_task2.csv` into 4 `.npy` files in `data/cxr/` |
| `evaluate_task2.py` | Runs model inference on 406-image test set, prints per-class AP table (head→tail), overall/zone mAP |
| `plot_per_class_mAP_task2.py` | Horizontal bar chart: per-class AP, zone shading, contest best (52.6%) and model overall mAP reference lines |
| `plot_class_tp_task2.py` | Paired bar chart: total positives vs TP at threshold=0.5, recall % annotations, zone shading |
| `script/run_task2_eval.sh` | Runs all three scripts for each of the 4 trained models; saves `.txt` and `.png` outputs to each model's output directory |

**One-time setup:**
```bash
conda run -n dicap python data/cxr/format_cxr2024_task2.py
```
Produces `formatted_task2_test_images.npy`, `formatted_task2_test_labels.npy`,
`formatted_task2_dev_images.npy`, `formatted_task2_dev_labels.npy` in `data/cxr/`.

**Run all Task 2 evaluations:**
```bash
bash script/run_task2_eval.sh
```

---

### Results Summary

| Model | Overall mAP | Head mAP | Medium mAP | Tail mAP | Gap to Contest Best |
|-------|------------:|----------:|-----------:|---------:|--------------------:|
| TResNet_L lb=1.0 (fully supervised) | **45.43%** | 59.23% | 35.47% | 43.00% | −7.2 pp |
| Hybrid-LB + SSL | 39.20% | 57.41% | 24.42% | 38.98% | −13.4 pp |
| Tail-LB + SSL | 37.63% | 55.35% | 23.04% | 37.77% | −15.0 pp |
| lb=0.066 + SSL | 33.92% | 54.50% | 20.25% | 28.10% | −18.7 pp |
| Contest best | 52.6% | — | — | — | — |

**Key findings:**
- Fully supervised lb=1.0 is the strongest SSL model by 6.2 pp — more labeled data matters more
  than SSL when the supervised baseline is already strong.
- SSL models (Hybrid-LB, Tail-LB) close the tail gap: Tail-LB tail mAP 37.77 vs lb=0.066 tail 28.10 (+9.7 pp), despite comparable overall mAP on Task 1.
- Pneumonia underperforms across all models (Task 2 label noise hypothesis: Task 2 gold standard
  uses stricter radiologist criteria than NLP-derived Task 1 labels → model learned NLP-definition,
  not radiologist-definition).
- Hernia has substantial variance between models (Task 1 mAP 28–42, Task 2 varies similarly) —
  rare enough that small labeled count differences matter.
- All 4 models achieve tail mAP > medium mAP on Task 2, confirming the air/gas class advantage
  is consistent across training strategies.

**Output files** (saved per model):
```
output/cxr_ours/{key}/task2_per_class_mAP.txt
output/cxr_ours/{key}/task2_per_class_mAP_{dataset_name}_lb{lb_ratio}_{net}.png
output/cxr_ours/{key}/task2_class_tp_{dataset_name}_lb{lb_ratio}_{net}_thr0.5.png
```

---

## Backbone Swap: ConvNeXt-Base

### Motivation

CXR-LT 2024 paper analysis: many top teams used ConvNeXt or EfficientNetV2 backbones pretrained on
MIMIC-CXR, whereas our current best model (TResNet_L) was pretrained only on ImageNet-1K. Swapping
to ConvNeXt-Base with ImageNet-22K pretraining gives richer initialization (14M images, 22K classes)
and matches the architecture choices of higher-scoring teams.

ConvNeXt-Base chosen over:
- ConvNeXt-Large: 197M params vs 89M — doubles training time for marginal gain
- EfficientNetV2: messier internal block structure, harder to extract clean spatial feature maps
- DenseNet121 (torchxrayvision): would require dropping the ML-Decoder head entirely

The ML-Decoder head (`ClasswiseEncoder`) is kept unchanged — only the backbone is swapped, so any
mAP difference is attributable to the backbone alone.

---

### Architecture Compatibility Verification

Verified 2026-05-26 with `timm==1.0.19`, `convnext_base.fb_in22k_ft_in1k`, input 512×512:

```
named_children: ['stem', 'stages', 'norm_pre', 'head']

after stem:     [1, 128, 128, 128]   (4× downsample)
after stages:   [1, 1024, 16, 16]    (32× total, 1024-D features)
after norm_pre: [1, 1024, 16, 16]    (LayerNorm, same shape)
```

`norm_pre` is a top-level named child → `IntermediateLayerExtracter` cuts cleanly at `norm_pre`,
forwarding `stem → stages → norm_pre` and returning normalized `[B, 1024, H//32, W//32]`.

`ClasswiseEncoder` then does:
```
[B, 1024, 16, 16] → flatten → [B, 256, 1024] → feature_projector → [B, 256, 512]
```
The `feature_projector` (`nn.Linear(dim_feature, dim_embed)`) re-initializes automatically when
`dim_feature` changes from 2432 (TResNet_L) to 1024 — no other head changes needed.

---

### Code Changes Required

#### 1. `lib/ML_decoder/backbone/cnn.py`

Add to `_MODELS` dict:
```python
"convnext_base":  lambda pretrained: timm.create_model(
    "convnext_base.fb_in22k_ft_in1k", pretrained=pretrained),
"convnext_large": lambda pretrained: timm.create_model(
    "convnext_large.fb_in22k_ft_in1k", pretrained=pretrained),
```

Add to `_MODELS_INFO` dict:
```python
"convnext_base":  ModelInfo("norm_pre", "head.fc", 1024, 1024, 1000, 32),
"convnext_large": ModelInfo("norm_pre", "head.fc", 1536, 1536, 1000, 32),
```

Update `create_cnn_backbone` — add `elif "convnext" in name:` branch:
```python
elif "convnext" in name:
    model = _MODELS[name](pretrained=pretrained)
```

The existing `IntermediateLayerExtracter` and `create_featuremap_backbone` need no changes.

#### 2. `script/run.sh` / `warm_up.py` / `fine_tune.py`

No structural changes. Pass `--net convnext_base` at the CLI. Warmup and fine-tune scripts already
accept `args.net` as a string — they work unchanged.

#### 3. `format_cxr2024.py` (and variants)

No changes needed — data pipeline is backbone-agnostic.

---

### Experiment Plan

Run warmup only first to get a cheap signal before committing to full SSL training.

| Step | Command | Purpose |
|------|---------|---------|
| 1. Warmup baseline (already done) | `--net tresnet_l --lb_ratio 1.0` | Reference: 40-class warmup mAP |
| 2. ConvNeXt warmup | `--net convnext_base --lb_ratio 1.0` | Comparable warmup mAP |
| 3. If ≥ TResNet_L: full SSL | `--net convnext_base --lb_ratio 0.066` | Tail-boost SSL run |
| 4. Task 1 + Task 2 eval | `bash script/run_task2_eval.sh` | Compare both tasks |

Decision rule: if ConvNeXt warmup mAP ≥ TResNet_L warmup mAP, proceed to full SSL pipeline.
If < TResNet_L, investigate whether longer warmup epochs or lower LR closes the gap before giving up.

**Estimated compute per warmup run:** similar to current TResNet_L warmup (ConvNeXt-Base ~89M params vs TResNet_L ~55M — roughly 1.5–1.6× more FLOPs per forward pass, offset partially by ConvNeXt's higher memory efficiency).

---

### Hyperparameter Notes for ConvNeXt

- **Learning rate**: ConvNeXt is generally more sensitive to LR than TResNet. If current LR causes
  instability, try 0.5× the TResNet_L LR as a starting point.
- **Weight decay**: ConvNeXt authors recommend higher weight decay (0.05–0.1) vs ResNet defaults.
  Current `args.weight_decay` may need tuning.
- **Image size**: Current 512×512 gives 16×16 feature maps. Could try 448×448 (14×14) to reduce
  memory if batch size is constrained. Don't go below 384×384 (12×12 feature maps get sparse).
- **`dim_embed`**: Currently 512. With 1024-D backbone features the projection ratio is 2:1 (same
  direction as TResNet_L's 2432→512). No change needed but 768 or 1024 could be explored later.

### Results (2026-05-28)

| Configuration | Overall | Head | Medium | Tail |
|---|:---:|:---:|:---:|:---:|
| TResNet_L lb=1.0 (reference) | 22.01 | 54.00 | 16.27 | 9.40 |
| ConvNeXt-Base lb=1.0 | 21.99 | 54.14 | 16.65 | 8.92 |
| TResNet_L Hybrid-LB | 16.03 | 46.88 | 8.52 | 5.71 |
| ConvNeXt-Base Hybrid-LB | 16.05 | 45.82 | 8.79 | 6.10 |

**Conclusion:** ConvNeXt-Base with ImageNet-22K pretraining is statistically equivalent to
TResNet_L in both lb=1.0 and hybrid-LB settings. The bottleneck is the SSL pipeline and the
ImageNet domain gap, not the architecture. Domain-specific CXR pretraining is the next lever.

---

## Next Step: CXR Domain Pretraining

### What the Top CXR-LT 2024 Teams Actually Did

Full analysis of the 9 top solutions from the challenge paper (arxiv 2506.07984):

| Rank | Team | Backbone | Pretraining | Task 1 | Task 2 |
|------|------|----------|-------------|--------|--------|
| T1-1st | zguo | ConvNeXt-S/B/T, ConvNeXt V2-B | ImageNet only | **0.281** | 0.511 |
| T1-2nd | tianjie_dai | EfficientNetV2-L | ImageNet → NIH, CheXpert, VinDr, BRAX | 0.279 | **0.519** |
| T1-3rd / T2-1st | XYPB | ConvNeXt-S, EfficientNetV2-S | ImageNet → MIMIC-CXR (CLIP) | 0.277 | **0.526** |
| T1-4th | dongkyunk | ConvNeXt-S | ImageNet → CheXpert, NIH, VinDr | 0.277 | — |
| T2-2nd | yangz16 | ViT-L (DINOv2) | Self-supervised on 710K CXR images | — | 0.511 |
| T2-4th | YYama | ConvNeXt V2-S, MaxViT-T | ImageNet → NIH | — | 0.509 |

**Key finding: The 1st place team used ImageNet-only pretraining!** Their secret was a
12-model ensemble + diffusion-model synthetic data generation (~100 images per tail class
with carefully crafted comorbidity prompts). Domain-specific CXR pretraining was used by
2nd–4th place teams, but it was not the decisive factor.

**Most relevant team for us: dongkyunk (4th place, Task 1)**
- Same ML-Decoder classification head as DiCaP
- Supervised multi-stage: ImageNet → CheXpert + NIH + VinDr-CXR
- Resolution 1024 (vs our 224)
- Added Noisy Student self-training (analogous to DiCaP's SSL phase)
- CheXFusion multi-view aggregation

---

### Plan: Multi-Stage Supervised CXR Pretraining

Follow the dongkyunk / tianjie_dai approach: supervised pretraining on publicly available
CXR datasets before fine-tuning on CXR-LT 2024.

#### Stage 1 — Pretrain on NIH Chest X-Ray14

**Dataset:** NIH Chest X-Ray14 (112,120 images, 14 disease labels)
- Publicly available, no DUA required
- Download: `https://nihcc.app.box.com/v/ChestXray-NIHCC`
- 14 classes: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema,
  Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax
- 8 of these 14 classes overlap directly with CXR-LT 2024's 40 classes

**Pretraining setup:**
- Model: `convnext_base.fb_in22k_ft_in1k` (keep ImageNet-22K init, fine-tune on NIH)
- Loss: ASL (same as our main pipeline)
- Image resolution: 224×224 (can step up to 384 if memory allows)
- Epochs: 20–30 (until val AP plateaus)
- Save best checkpoint as `convnext_base_nih_pretrained.pth.tar`

**Data format:** NIH provides a CSV (`Data_Entry_2017_v2020.csv`) with image paths and
pipe-separated labels. Write a format script similar to `format_cxr2024.py`.

---

#### Stage 2 — Fine-tune on CXR-LT 2024 (full pipeline)

Load `convnext_base_nih_pretrained.pth.tar` as the starting point instead of ImageNet
weights. Run the full DiCaP pipeline: warm_up → main (SSL) → fine_tune.

Add a new backbone variant `"convnext_base_nih"` in `cnn.py` that loads the local checkpoint:

```python
def _load_convnext_base_nih(pretrained):
    model = timm.create_model("convnext_base", pretrained=False)
    if pretrained:
        ckpt = torch.load("./pretrained/convnext_base_nih.pth.tar", map_location="cpu")
        key = "state_dict_ema" if "state_dict_ema" in ckpt else "state_dict"
        sd = {k.replace("module.", ""): v for k, v in ckpt[key].items()}
        model.load_state_dict(sd, strict=False)
    return model

_MODELS["convnext_base_nih"] = _load_convnext_base_nih
_MODELS_INFO["convnext_base_nih"] = ModelInfo("norm_pre", "head.fc", 1024, 1024, 14, 32)
```

Note `dim_out=14` reflects NIH's 14 classes. The head (`head.fc`) will be re-initialized
when DiCaP replaces it with the 40-class ClasswiseEncoder — only the backbone weights matter.

---

#### Alternative: CLIP-Based MIMIC-CXR Pretraining (like XYPB, 1st Task 2)

XYPB's approach used contrastive image–text pretraining on MIMIC-CXR image–report pairs
with BioMedLM as the text encoder. This is the most complex option but achieved the highest
Task 2 mAP (0.526).

MIMIC-CXR paired reports are at `/home/share/mimic-cxr/`. The approach requires:
1. BioMedLM or BioGPT as the text encoder
2. A CLIP-style symmetric contrastive loss over (image, report) pairs
3. Significant GPU time (days at 1024×1024)

**Recommendation:** Start with NIH supervised pretraining (Stage 1 above). If that gives
≥1 pp improvement, proceed to CLIP-based MIMIC-CXR pretraining as a second iteration.

---

### Decision Checkpoints

| Decision | Criterion |
|----------|-----------|
| NIH pretraining successful | Warmup mAP ≥ 23% (> ImageNet-22K baseline 22.15%) |
| Proceed to CLIP pretraining | NIH warmup gains ≥ 1 pp but still below 25% |
| Consider resolution increase | After pretraining gives gains; step up to 384×384 |
| Try synthetic tail data (1st place strategy) | After all pretraining options explored |

---

### Files to Create

| File | Purpose |
|------|---------|
| `data/nih/format_nih.py` | Parse NIH CSV, write 4 npy files to `data/nih/` |
| `pretrain_nih.py` | Supervised pretraining script (can reuse warm_up.py logic) |
| `pretrained/` | Directory for storing domain-pretrained checkpoints |
| `script/pretrain_nih.sh` | Shell script for NIH pretraining run |
