import os
import numpy as np
import pandas as pd

CXR_DIR  = '/home/share/mimic-cxr-jpg-2.0.0'
CSV_DIR  = '/home/share/cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays-2.0.0/cxr-lt-2024'
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use files-512 for faster loading (images are resized to 224x224 anyway)
USE_512 = True

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

for phase, csv_file in [('train', 'train_labeled.csv'),
                         ('val',   'development_labeled_task1.csv'),
                         ('test',  'test_labeled_task1.csv')]:
    df = pd.read_csv(os.path.join(CSV_DIR, csv_file))

    # Build absolute image paths
    def make_path(fpath):
        if USE_512:
            fpath = fpath.replace('files/', 'files-512/', 1)
        return os.path.join(CXR_DIR, fpath)

    images = np.array([make_path(p) for p in df['fpath']])
    labels = df[LABEL_COLS].values.astype(np.float32)

    np.save(os.path.join(SAVE_DIR, f'formatted_{phase}_images.npy'), images)
    np.save(os.path.join(SAVE_DIR, f'formatted_{phase}_labels.npy'), labels)

    pos_per_class = labels.sum(axis=0)
    print(f'{phase}: {len(images)} images, {labels.shape[1]} classes, '
          f'{pos_per_class.sum():.0f} total positives, '
          f'avg {pos_per_class.mean():.1f} pos/class')
    print(f'  rarest:  {LABEL_COLS[pos_per_class.argmin()]} ({pos_per_class.min():.0f})')
    print(f'  most common: {LABEL_COLS[pos_per_class.argmax()]} ({pos_per_class.max():.0f})')

print('Done. Files saved to', SAVE_DIR)
