"""
Per-class mAP evaluation for CXR-LT 2024 Task 2 (gold standard test set).

Task 2 evaluates on 406 manually annotated images covering 26 of the 40 Task 1
disease classes. The model outputs 40 classes; we extract the 26 Task 2 classes
using MODEL_TASK2_INDICES.

Contest best mAP on Task 2: 52.6%

Usage:
    python evaluate_task2.py \\
        --checkpoint output/cxr_ours/cxr_hybrid_lb/tresnet_l/40.0/fine_tune/best_model.pth.tar \\
        --dataset_dir data/cxr \\
        --net tresnet_l \\
        --dataset_name cxr_hybrid_lb --lb_ratio 40.0
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.utils.data

import _init_paths
from lib.dataset.get_dataset import TransformWithPatches_Val
from lib.dataset.handlers import CXR_handler
from lib.ML_decoder.backbone.ML_decoder import ClasswiseModel
from lib.utils.helper import clean_state_dict
from sklearn.metrics import average_precision_score

# Task 2 classes in CSV column order (26 of the 40 Task 1 classes)
TASK2_LABEL_COLS = [
    'Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly', 'Consolidation',
    'Edema', 'Emphysema', 'Enlarged Cardiomediastinum', 'Fibrosis', 'Fracture',
    'Hernia', 'Infiltration', 'Lung Lesion', 'Lung Opacity', 'Mass', 'Normal',
    'Nodule', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening',
    'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax',
    'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta',
]

# Model's 40-class output indices corresponding to each Task 2 class (in TASK2_LABEL_COLS order)
# Model LABEL_COLS is sorted alphabetically; Normal(24) comes after Nodule(23) there,
# but Task 2 CSV has Normal before Nodule — hence the swap at positions 14/15.
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
    24,  # Normal  (note: model idx 24, Task2 col 14)
    23,  # Nodule  (note: model idx 23, Task2 col 15)
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--dataset_dir', default='./data/cxr', type=str)
    parser.add_argument('--dataset_name', default='cxr', type=str)
    parser.add_argument('--lb_ratio', default=0.05, type=float)
    parser.add_argument('--net', default='resnet50', type=str)
    parser.add_argument('--dim_embed', default=512, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--grid_perside', default=2, type=int)
    parser.add_argument('--cutout', default=0.5, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--split_seed', default=1, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    args.n_classes = 40

    # Load Task 2 gold standard test set (406 images, 26 classes)
    # formatted_task2_test_*.npy live in data/cxr/ (run format_cxr2024_task2.py first)
    task2_dir = './data/cxr'
    test_images = np.load(os.path.join(task2_dir, 'formatted_task2_test_images.npy'),
                          allow_pickle=True)
    test_labels = np.load(os.path.join(task2_dir, 'formatted_task2_test_labels.npy'))

    print(f'Task 2 test set: {len(test_images)} images, {test_labels.shape[1]} classes')

    eval_dataset = CXR_handler(test_images, test_labels, task2_dir,
                               transform=TransformWithPatches_Val(args))
    loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Load model (40-class)
    model = ClasswiseModel(args.net, args.n_classes, args.dim_embed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    ema_key = 'state_dict_ema' if 'state_dict_ema' in checkpoint else 'state_dict'
    state = clean_state_dict(checkpoint.get(ema_key, checkpoint))
    model.load_state_dict(state, strict=True)
    model = model.cuda()
    model.eval()

    print(f'Using weights from checkpoint key: {ema_key}')
    print(f'Loaded checkpoint: {args.checkpoint}')
    print(f'Evaluating on Task 2 gold standard test set — {len(test_images)} images...')

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs_list, targets, _ in loader:
            inputs = inputs_list[0].cuda()
            outputs = torch.sigmoid(model(inputs))
            # Extract only the 26 Task 2 class predictions
            task2_preds = outputs[:, MODEL_TASK2_INDICES]
            all_preds.append(task2_preds.cpu().numpy())
            all_targets.append(targets.numpy())

    preds   = np.vstack(all_preds)    # [406, 26]
    targets = np.vstack(all_targets)  # [406, 26]

    # Training frequency for zone classification (train+val combined, same as Task 1)
    train_lbl = np.load('./data/cxr/formatted_train_labels.npy')
    val_lbl   = np.load('./data/cxr/formatted_val_labels.npy')
    combined  = np.vstack([train_lbl, val_lbl])
    n_combined = len(combined)
    # Extract frequencies for Task 2 classes only (in TASK2_LABEL_COLS order)
    train_freq = combined[:, MODEL_TASK2_INDICES].sum(axis=0)

    head_thresh = n_combined * 0.10
    tail_thresh = n_combined * 0.01

    # Per-class AP
    ap_per_class = []
    for i in range(len(TASK2_LABEL_COLS)):
        if targets[:, i].sum() > 0:
            ap = average_precision_score(targets[:, i], preds[:, i]) * 100
        else:
            ap = float('nan')
        ap_per_class.append(ap)
    ap_per_class = np.array(ap_per_class)

    # Print table sorted head → tail by training frequency
    sort_idx = np.argsort(train_freq)[::-1]
    print(f'\n{"Class":<35} {"Train+Val #":>11}  {"AP":>6}')
    print('-' * 58)
    for i in sort_idx:
        f = train_freq[i]
        if f <= tail_thresh:
            marker = ' <-- tail'
        elif f <= head_thresh:
            marker = ' <-- medium'
        else:
            marker = ''
        ap_str = f'{ap_per_class[i]:.1f}' if not np.isnan(ap_per_class[i]) else '  N/A'
        print(f'{TASK2_LABEL_COLS[i]:<35} {f:>11.0f}  {ap_str:>6}{marker}')

    head_mask = train_freq > head_thresh
    mid_mask  = (train_freq > tail_thresh) & (train_freq <= head_thresh)
    tail_mask = train_freq <= tail_thresh
    valid_aps = ap_per_class[~np.isnan(ap_per_class)]

    print('-' * 58)
    print(f'Overall mAP:      {valid_aps.mean():.2f}')
    print(f'Head mAP  (>10%): {np.nanmean(ap_per_class[head_mask]):.2f}  ({head_mask.sum()} classes)')
    print(f'Medium mAP (1-10%): {np.nanmean(ap_per_class[mid_mask]):.2f}  ({mid_mask.sum()} classes)')
    print(f'Tail mAP  (≤1%):  {np.nanmean(ap_per_class[tail_mask]):.2f}  ({tail_mask.sum()} classes)')
    print(f'\nContest best (Task 2): 52.6% mAP')
    print(f'Note: 406 gold-standard test images; classes with few positives (Lung Lesion=6,')
    print(f'      Infiltration=12) have high AP variance.')


if __name__ == '__main__':
    main()
