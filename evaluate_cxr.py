"""
Per-class mAP evaluation for CXR-LT 2024.

Usage:
    python evaluate_cxr.py --checkpoint output/cxr_ours/cxr/0.05/fine_tune/best_model.pth.tar \
                           --dataset_dir data/cxr --lb_ratio 0.05
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--dataset_dir', default='./data', type=str)
    parser.add_argument('--dataset_name', default='cxr', type=str)
    parser.add_argument('--lb_ratio', default=0.05, type=float)
    parser.add_argument('--split', default='test', choices=['val', 'test'],
                        help='Which npy split to evaluate on (val=dev set, test=test set)')
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


def run_inference(loader, model):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs_list, targets, _ in loader:
            inputs = inputs_list[0].cuda()
            outputs = torch.sigmoid(model(inputs))
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
    return np.vstack(all_preds), np.vstack(all_targets)


def main():
    args = get_args()
    args.n_classes = 40

    # Load eval dataset directly from npy (val=dev set for tuning, test=final eval)
    eval_images = np.load(os.path.join(args.dataset_dir, f'formatted_{args.split}_images.npy'),
                          allow_pickle=True)
    eval_labels = np.load(os.path.join(args.dataset_dir, f'formatted_{args.split}_labels.npy'))
    eval_dataset = CXR_handler(eval_images, eval_labels, args.dataset_dir,
                               transform=TransformWithPatches_Val(args))
    val_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Load model
    model = ClasswiseModel(args.net, args.n_classes, args.dim_embed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # Prefer EMA weights — training tracks best mAP using the EMA model
    ema_key = 'state_dict_ema' if 'state_dict_ema' in checkpoint else 'state_dict'
    state = clean_state_dict(checkpoint.get(ema_key, checkpoint))
    print(f'Using weights from checkpoint key: {ema_key}')
    model.load_state_dict(state, strict=True)
    model = model.cuda()

    print(f'Loaded checkpoint: {args.checkpoint}')
    print(f'Evaluating on [{args.split}] split — {len(eval_dataset)} images...\n')

    preds, targets = run_inference(val_loader, model)

    # Zone classification: use train+val combined positive counts with 10%/1% thresholds
    # → 9 head, 15 medium, 16 tail classes (matches competition paper)
    # Use full_train_labels if present (cxr_tail_lb only has labeled subset in train npy)
    full_train_path = os.path.join(args.dataset_dir, 'formatted_full_train_labels.npy')
    train_freq_path = full_train_path if os.path.exists(full_train_path) else \
                      os.path.join(args.dataset_dir, 'formatted_train_labels.npy')
    train_labels = np.load(train_freq_path)
    val_labels   = np.load(os.path.join(args.dataset_dir, 'formatted_val_labels.npy'))
    combined_labels = np.vstack([train_labels, val_labels])
    n_combined  = len(combined_labels)
    train_freq  = combined_labels.sum(axis=0)   # total+ per class across train+val
    HEAD_THRESH = n_combined * 0.10             # 29,816
    TAIL_THRESH = n_combined * 0.01             # 2,982

    ap_per_class = []
    for i in range(args.n_classes):
        if targets[:, i].sum() > 0:
            ap = average_precision_score(targets[:, i], preds[:, i]) * 100
        else:
            ap = float('nan')
        ap_per_class.append(ap)

    ap_per_class = np.array(ap_per_class)
    sort_idx = np.argsort(train_freq)[::-1]  # sort head → tail

    print(f'{"Class":<35} {"Train+Val #":>11}  {"AP":>6}')
    print('-' * 58)
    for i in sort_idx:
        if train_freq[i] <= TAIL_THRESH:
            marker = ' <-- tail'
        elif train_freq[i] <= HEAD_THRESH:
            marker = ' <-- medium'
        else:
            marker = ''
        ap_str = f'{ap_per_class[i]:.1f}' if not np.isnan(ap_per_class[i]) else '  N/A'
        print(f'{LABEL_COLS[i]:<35} {train_freq[i]:>11.0f}  {ap_str:>6}{marker}')

    head_mask = train_freq > HEAD_THRESH
    mid_mask  = (train_freq > TAIL_THRESH) & (train_freq <= HEAD_THRESH)
    tail_mask = train_freq <= TAIL_THRESH

    valid_aps = ap_per_class[~np.isnan(ap_per_class)]
    print('-' * 55)
    print(f'Overall mAP:      {valid_aps.mean():.2f}')
    print(f'Head mAP  (>10%): {np.nanmean(ap_per_class[head_mask]):.2f}  '
          f'({head_mask.sum()} classes)')
    print(f'Medium mAP (1-10%): {np.nanmean(ap_per_class[mid_mask]):.2f}  '
          f'({mid_mask.sum()} classes)')
    print(f'Tail mAP  (≤1%):  {np.nanmean(ap_per_class[tail_mask]):.2f}  '
          f'({tail_mask.sum()} classes)')


if __name__ == '__main__':
    main()
