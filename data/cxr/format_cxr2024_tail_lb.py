"""
format_cxr2024_tail_lb.py — Tail-class-driven labeled/unlabeled split.

Source: train_labeled.csv  (258,871 images)
  → Labeled   (13,921):  has >= 1 positive label among the 16 tail classes
  → Unlabeled (244,950): no positive label in any tail class

Val:   development_labeled_task1.csv (39,293 images) — for evaluation only

Output dir: data/cxr_tail_lb/
  formatted_train_images.npy       (13921,)     str    — labeled paths
  formatted_train_labels.npy       (13921, 40)  float32
  formatted_unlabeled_images.npy   (244950,)    str    — unlabeled paths
  formatted_unlabeled_labels.npy   (244950, 40) float32
  formatted_val_images.npy         (39293,)     str
  formatted_val_labels.npy         (39293, 40)  float32
"""

import os
import numpy as np
import pandas as pd

CXR_DIR  = '/home/share/mimic-cxr-jpg-2.0.0'
CSV_DIR  = '/home/share/cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays-2.0.0/cxr-lt-2024'
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cxr_tail_lb')

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

# 16 tail classes — bottom 16 by training frequency in train_labeled.csv
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

# ── Train set → labeled + unlabeled ─────────────────────────────────────────
train_df = pd.read_csv(os.path.join(CSV_DIR, 'train_labeled.csv'))

has_tail = (train_df[TAIL_CLASSES] == 1).any(axis=1)
lb_df = train_df[has_tail].reset_index(drop=True)
ub_df = train_df[~has_tail].reset_index(drop=True)

print(f'Train total  : {len(train_df)}')
print(f'  Labeled    : {len(lb_df)} ({100*len(lb_df)/len(train_df):.2f}%)')
print(f'  Unlabeled  : {len(ub_df)} ({100*len(ub_df)/len(train_df):.2f}%)')

for tag, df in [('train', lb_df), ('unlabeled', ub_df)]:
    images = np.array([make_path(p) for p in df['fpath']])
    labels = df[LABEL_COLS].values.astype(np.float32)
    np.save(os.path.join(SAVE_DIR, f'formatted_{tag}_images.npy'), images)
    np.save(os.path.join(SAVE_DIR, f'formatted_{tag}_labels.npy'), labels)
    pos_per_class = labels.sum(axis=0)
    print(f'  [{tag}] {len(images)} images, {pos_per_class.sum():.0f} total positives, '
          f'avg {pos_per_class.mean():.1f} pos/class')
    print(f'    rarest  : {LABEL_COLS[pos_per_class.argmin()]} ({pos_per_class.min():.0f})')
    print(f'    most common: {LABEL_COLS[pos_per_class.argmax()]} ({pos_per_class.max():.0f})')

# ── Dev set → val (for early stopping during training) ───────────────────────
dev_df = pd.read_csv(os.path.join(CSV_DIR, 'development_labeled_task1.csv'))
val_images = np.array([make_path(p) for p in dev_df['fpath']])
val_labels = dev_df[LABEL_COLS].values.astype(np.float32)
np.save(os.path.join(SAVE_DIR, 'formatted_val_images.npy'), val_images)
np.save(os.path.join(SAVE_DIR, 'formatted_val_labels.npy'), val_labels)
print(f'Val set      : {len(val_images)} images')

# ── Test set → test (for final evaluation) ───────────────────────────────────
test_df = pd.read_csv(os.path.join(CSV_DIR, 'test_labeled_task1.csv'))
test_images = np.array([make_path(p) for p in test_df['fpath']])
test_labels = test_df[LABEL_COLS].values.astype(np.float32)
np.save(os.path.join(SAVE_DIR, 'formatted_test_images.npy'), test_images)
np.save(os.path.join(SAVE_DIR, 'formatted_test_labels.npy'), test_labels)
print(f'Test set     : {len(test_images)} images')

# ── Full train labels (all 258,871 rows) for correct class-frequency sorting ─
# formatted_train_labels.npy only has the 13,921 labeled subset; eval scripts
# need the full training distribution to sort classes head→tail correctly.
full_train_labels = train_df[LABEL_COLS].values.astype(np.float32)
np.save(os.path.join(SAVE_DIR, 'formatted_full_train_labels.npy'), full_train_labels)
print(f'Full train labels: {len(full_train_labels)} rows saved')

print(f'\nDone. Files saved to {os.path.abspath(SAVE_DIR)}')
