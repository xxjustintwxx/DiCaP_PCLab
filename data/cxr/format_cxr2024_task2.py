"""Format CXR-LT 2024 Task 2 data into npy arrays.

Task 2 uses a manually annotated gold standard subset (406 test images)
covering 26 of the 40 Task 1 disease classes.

Run once from the DiCaP repo root or directly:
    python data/cxr/format_cxr2024_task2.py
"""

import os
import numpy as np
import pandas as pd

CXR_DIR = '/home/share/mimic-cxr-jpg-2.0.0'
CSV_DIR = '/home/share/cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays-2.0.0/cxr-lt-2024'
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
USE_512  = True

# 26 Task 2 classes in CSV column order (Normal before Nodule here)
TASK2_LABEL_COLS = [
    'Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly', 'Consolidation',
    'Edema', 'Emphysema', 'Enlarged Cardiomediastinum', 'Fibrosis', 'Fracture',
    'Hernia', 'Infiltration', 'Lung Lesion', 'Lung Opacity', 'Mass', 'Normal',
    'Nodule', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening',
    'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax',
    'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta',
]


def make_path(fpath):
    if USE_512:
        fpath = fpath.replace('files/', 'files-512/', 1)
    return os.path.join(CXR_DIR, fpath)


for phase, csv_file in [
    ('task2_test', 'test_labeled_task2.csv'),
    ('task2_dev',  'development_labeled_task2.csv'),
]:
    df = pd.read_csv(os.path.join(CSV_DIR, csv_file))
    images = np.array([make_path(p) for p in df['fpath']])
    labels = df[TASK2_LABEL_COLS].values.astype(np.float32)

    np.save(os.path.join(SAVE_DIR, f'formatted_{phase}_images.npy'), images)
    np.save(os.path.join(SAVE_DIR, f'formatted_{phase}_labels.npy'), labels)

    pos = labels.sum(axis=0)
    print(f'\n{phase}: {len(images)} images, {labels.shape[1]} classes')
    print(f'{"Class":<35} {"Positives":>10}')
    print('-' * 47)
    for c, p in zip(TASK2_LABEL_COLS, pos):
        print(f'{c:<35} {int(p):>10}')
    print(f'Total positives: {int(pos.sum())}')

print('\nDone. Files saved to', SAVE_DIR)
