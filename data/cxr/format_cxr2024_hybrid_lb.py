"""
format_cxr2024_hybrid_lb.py — Hybrid calibration-corrected tail-class split.

Supervised training set (80%):  tail-positive images NOT drawn into the calibration set
Calibration set         (20%):  random sample from the full 258K (natural long-tail dist.)
Unlabeled pool:                 remaining images (non-calib, non-tail-positive)

The calibration set is sampled from the full 258,871 training images so that all 40 classes
appear at their real prevalence.  This gives DiCaP's update_weight() and
dynamic_threshold_generate() an unbiased view of the data distribution, unlike the tail_lb
split where the 20% holdout was also tail-biased.

N_calib is chosen so that N_calib / (N_calib + N_train) ≈ 20%, accounting for the small
expected overlap between the calib sample and the tail-positive pool.

Output dir: data/cxr_hybrid_lb/
  formatted_train_images.npy      (~13737,)    str    — tail-positive, supervised training
  formatted_train_labels.npy      (~13737, 40) float32
  formatted_calib_images.npy      (~3434,)     str    — natural distribution, calibration
  formatted_calib_labels.npy      (~3434, 40)  float32
  formatted_unlabeled_images.npy  (~241700,)   str    — remaining images, unlabeled pool
  formatted_unlabeled_labels.npy  (~241700, 40) float32
  formatted_val_images.npy        (39293,)     str
  formatted_val_labels.npy        (39293, 40)  float32
  formatted_test_images.npy       (78946,)     str
  formatted_test_labels.npy       (78946, 40)  float32
  formatted_full_train_labels.npy (258871, 40) float32 — full train for class-freq sorting
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42

CXR_DIR  = '/home/share/mimic-cxr-jpg-2.0.0'
CSV_DIR  = '/home/share/cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays-2.0.0/cxr-lt-2024'
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cxr_hybrid_lb')

USE_512 = True  # Replace files/ with files-512/ for ~10x faster loading

LABEL_COLS = [
    'Adenopathy', 'Atelectasis', 'Azygos Lobe', 'Calcification of the Aorta',
    'Cardiomegaly', 'Clavicle Fracture', 'Consolidation', 'Edema', 'Emphysema',
    'Enlarged Cardiomediastinum', 'Fibrosis', 'Fissure', 'Fracture', 'Granuloma',
    'Hernia', 'Hydropneumothorax', 'Infarction', 'Infiltration', 'Kyphosis',
    'Lobar Atelectasis', 'Lung Lesion', 'Lung Opacity', 'Mass', 'Nodule', 'Normal',
    'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', 'Pneumomediastinum',
    'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 'Pulmonary Embolism',
    'Pulmonary Hypertension', 'Rib Fracture', 'Round(ed) Atelectasis',
    'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta', 'Tuberculosis'
]

TAIL_CLASSES = [
    'Lobar Atelectasis',       # 129  train+
    'Clavicle Fracture',       # 168
    'Round(ed) Atelectasis',   # 172
    'Azygos Lobe',             # 199
    'Pneumoperitoneum',        # 516
    'Pleural Other',           # 616
    'Hydropneumothorax',       # 646
    'Pneumomediastinum',       # 704
    'Infarction',              # 727
    'Kyphosis',                # 778
    'Pulmonary Hypertension',  # 903
    'Fibrosis',                # 1169
    'Pulmonary Embolism',      # 1631
    'Subcutaneous Emphysema',  # 2046
    'Tuberculosis',            # 2078
    'Lung Lesion',             # 2338
]


def make_path(fpath):
    if USE_512:
        fpath = fpath.replace('files/', 'files-512/', 1)
    return os.path.join(CXR_DIR, fpath)


os.makedirs(SAVE_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

# ── Train set ─────────────────────────────────────────────────────────────────
train_df = pd.read_csv(os.path.join(CSV_DIR, 'train_labeled.csv'))
n_total  = len(train_df)

has_tail = (train_df[TAIL_CLASSES] == 1).any(axis=1).values  # bool array (n_total,)
n_tail   = int(has_tail.sum())

# Compute n_calib so that n_calib / (n_calib + n_train) ≈ 0.20
# where n_train = n_tail - expected_overlap = n_tail - n_calib*(n_tail/n_total)
# Solving: n_calib = 0.2 * n_tail / (0.8 + 0.2 * n_tail / n_total)
f       = n_tail / n_total
n_calib = round(0.2 * n_tail / (0.8 + 0.2 * f))

print(f'Train total  : {n_total}')
print(f'Tail-positive: {n_tail} ({100*n_tail/n_total:.2f}%)')
print(f'Calib target : {n_calib} (expected ratio ≈ 20%)')

# Sample calibration set from the full 258K (natural distribution)
calib_indices = rng.choice(n_total, size=n_calib, replace=False)
calib_mask    = np.zeros(n_total, dtype=bool)
calib_mask[calib_indices] = True

# Supervised training: tail-positive AND not in calib
train_mask = has_tail & ~calib_mask

# Unlabeled pool: not tail-positive AND not in calib
ub_mask = ~has_tail & ~calib_mask

calib_df = train_df[calib_mask].reset_index(drop=True)
train_supervised_df = train_df[train_mask].reset_index(drop=True)
ub_df    = train_df[ub_mask].reset_index(drop=True)

# Actual overlap (tail-positive images that landed in calib)
actual_overlap = int((calib_mask & has_tail).sum())
actual_ratio   = len(calib_df) / (len(calib_df) + len(train_supervised_df))

print(f'\nActual split:')
print(f'  Calib set (20% target): {len(calib_df)} images '
      f'(incl. {actual_overlap} tail-positive — {100*actual_overlap/len(calib_df):.1f}%)')
print(f'  Supervised training   : {len(train_supervised_df)} images (tail-positive, not in calib)')
print(f'  Unlabeled pool        : {len(ub_df)} images')
print(f'  Calib ratio           : {100*actual_ratio:.1f}%  (target 20%)')
print(f'  Total labeled         : {len(calib_df) + len(train_supervised_df)}')

# ── Save supervised training set ──────────────────────────────────────────────
for tag, df in [('train', train_supervised_df), ('calib', calib_df), ('unlabeled', ub_df)]:
    images = np.array([make_path(p) for p in df['fpath']])
    labels = df[LABEL_COLS].values.astype(np.float32)
    np.save(os.path.join(SAVE_DIR, f'formatted_{tag}_images.npy'), images)
    np.save(os.path.join(SAVE_DIR, f'formatted_{tag}_labels.npy'), labels)
    pos_per_class = labels.sum(axis=0)
    print(f'\n[{tag}] {len(images)} images')
    print(f'  total positives : {pos_per_class.sum():.0f}')
    print(f'  rarest class    : {LABEL_COLS[int(pos_per_class.argmin())]} ({pos_per_class.min():.0f})')
    print(f'  most common     : {LABEL_COLS[int(pos_per_class.argmax())]} ({pos_per_class.max():.0f})')

# ── Dev set → val ─────────────────────────────────────────────────────────────
dev_df    = pd.read_csv(os.path.join(CSV_DIR, 'development_labeled_task1.csv'))
val_images = np.array([make_path(p) for p in dev_df['fpath']])
val_labels = dev_df[LABEL_COLS].values.astype(np.float32)
np.save(os.path.join(SAVE_DIR, 'formatted_val_images.npy'), val_images)
np.save(os.path.join(SAVE_DIR, 'formatted_val_labels.npy'), val_labels)
print(f'\nVal set      : {len(val_images)} images')

# ── Test set ──────────────────────────────────────────────────────────────────
test_df    = pd.read_csv(os.path.join(CSV_DIR, 'test_labeled_task1.csv'))
test_images = np.array([make_path(p) for p in test_df['fpath']])
test_labels = test_df[LABEL_COLS].values.astype(np.float32)
np.save(os.path.join(SAVE_DIR, 'formatted_test_images.npy'), test_images)
np.save(os.path.join(SAVE_DIR, 'formatted_test_labels.npy'), test_labels)
print(f'Test set     : {len(test_images)} images')

# ── Full train labels (all 258,871 rows) for class-frequency zone sorting ─────
full_train_labels = train_df[LABEL_COLS].values.astype(np.float32)
np.save(os.path.join(SAVE_DIR, 'formatted_full_train_labels.npy'), full_train_labels)
print(f'Full train labels: {len(full_train_labels)} rows')

print(f'\nDone. Files saved to {os.path.abspath(SAVE_DIR)}')
