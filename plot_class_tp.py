"""
Per-class positive count vs true-positive count bar chart for CXR-LT 2024.

For each class (ranked left→right by val positive count, high to low) plots:
  - Total positives in the val set  (blue)
  - Correctly classified positives  (green, TP at threshold=0.5)

Usage:
    python plot_class_tp.py \
        --checkpoint output/cxr_ours/cxr/0.05/fine_tune/best_model.pth.tar \
        --dataset_dir ./data/cxr \
        --lb_ratio 0.05 \
        --out_dir ./output/plots
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import _init_paths
import torch
import torch.utils.data
from lib.dataset.get_dataset import TransformWithPatches_Val
from lib.dataset.handlers import CXR_handler
from lib.ML_decoder.backbone.ML_decoder import ClasswiseModel
from lib.utils.helper import clean_state_dict

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

# Zone thresholds computed dynamically from train+val combined counts
# (10%/1% of 298,164 → HEAD=29,816 ; TAIL=2,982 → 9/15/16 split)
# Placeholders — overwritten in main() after loading labels.
HEAD_THRESH = 29816
TAIL_THRESH = 2982


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        default='output/cxr_ours/cxr/0.05/fine_tune/best_model.pth.tar')
    parser.add_argument('--dataset_dir', default='./data/cxr')
    parser.add_argument('--dataset_name', default='cxr')
    parser.add_argument('--lb_ratio', default=0.05, type=float)
    parser.add_argument('--split', default='test', choices=['val', 'test'],
                        help='Which npy split to evaluate on (val=dev set, test=test set)')
    parser.add_argument('--net', default='resnet50')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='Sigmoid threshold for positive prediction')
    parser.add_argument('--dim_embed', default=512, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--grid_perside', default=2, type=int)
    parser.add_argument('--cutout', default=0.5, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--split_seed', default=1, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('--out_dir', default='./output/plots')
    return parser.parse_args()


def run_inference(args):
    args.n_classes = 40
    eval_images = np.load(os.path.join(args.dataset_dir, f'formatted_{args.split}_images.npy'),
                          allow_pickle=True)
    eval_labels = np.load(os.path.join(args.dataset_dir, f'formatted_{args.split}_labels.npy'))
    eval_dataset = CXR_handler(eval_images, eval_labels, args.dataset_dir,
                               transform=TransformWithPatches_Val(args))
    val_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = ClasswiseModel(args.net, args.n_classes, args.dim_embed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    ema_key = 'state_dict_ema' if 'state_dict_ema' in checkpoint else 'state_dict'
    state = clean_state_dict(checkpoint.get(ema_key, checkpoint))
    model.load_state_dict(state, strict=True)
    model = model.cuda()
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs_list, targets, _ in val_loader:
            inputs = inputs_list[0].cuda()
            outputs = torch.sigmoid(model(inputs))
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    preds   = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    return preds, targets


def main():
    args = get_args()

    print(f'Loading checkpoint: {args.checkpoint}')
    preds, targets = run_inference(args)

    # Binarise predictions
    pred_bin = (preds >= args.threshold).astype(np.float32)

    # Per-class counts
    val_pos = targets.sum(axis=0).astype(int)           # total positives in val
    true_pos = (pred_bin * targets).sum(axis=0).astype(int)  # TP per class
    recall = np.where(val_pos > 0, true_pos / val_pos, 0.0)

    # Training frequency for zone colouring
    # Train+val combined for zone classification (10%/1% → 9/15/16 split)
    full_train_path = os.path.join(args.dataset_dir, 'formatted_full_train_labels.npy')
    train_freq_path = full_train_path if os.path.exists(full_train_path) else \
                      os.path.join(args.dataset_dir, 'formatted_train_labels.npy')
    train_lbl  = np.load(train_freq_path)
    val_lbl    = np.load(os.path.join(args.dataset_dir, 'formatted_val_labels.npy'))
    combined   = np.vstack([train_lbl, val_lbl])
    n_combined = len(combined)
    train_freq = combined.sum(axis=0).astype(int)

    global HEAD_THRESH, TAIL_THRESH
    HEAD_THRESH = int(n_combined * 0.10)
    TAIL_THRESH = int(n_combined * 0.01)

    # Sort left→right: highest train frequency to lowest (ensures color zones are contiguous)
    sort_idx = np.argsort(train_freq)[::-1]

    labels_sorted    = [LABEL_COLS[i]  for i in sort_idx]
    val_pos_sorted   = val_pos[sort_idx]
    true_pos_sorted  = true_pos[sort_idx]
    recall_sorted    = recall[sort_idx]
    train_freq_sorted = train_freq[sort_idx]

    # Zone colour per class (based on train frequency)
    zone_colors = []
    for f in train_freq_sorted:
        if f >= HEAD_THRESH:
            zone_colors.append('#2196F3')   # blue  — head
        elif f >= TAIL_THRESH:
            zone_colors.append('#FF9800')   # orange — medium
        else:
            zone_colors.append('#F44336')   # red   — tail

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    n = len(LABEL_COLS)
    fig_width = max(22, n * 0.62)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    x = np.arange(n)
    bar_w = 0.38

    # Total positives (lighter shade)
    bars_pos = ax.bar(x - bar_w / 2, val_pos_sorted,
                      width=bar_w, label='Total positives (val)',
                      color=[c + '55' for c in zone_colors],   # ~33% alpha hex
                      edgecolor=zone_colors, linewidth=0.8)

    # True positives (solid)
    bars_tp = ax.bar(x + bar_w / 2, true_pos_sorted,
                     width=bar_w, label=f'Correctly classified TP (thr={args.threshold})',
                     color=zone_colors, alpha=0.88)

    # Recall % label above each TP bar
    for xi, (tp, vp, rc) in enumerate(zip(true_pos_sorted, val_pos_sorted, recall_sorted)):
        if vp > 0:
            ax.text(xi + bar_w / 2, tp + max(val_pos_sorted) * 0.005,
                    f'{rc*100:.0f}%', ha='center', va='bottom',
                    fontsize=5.5, color='#333333')

    # Zone background shading
    head_end   = sum(1 for f in train_freq_sorted if f >= HEAD_THRESH)
    medium_end = sum(1 for f in train_freq_sorted if f >= TAIL_THRESH)

    ax.axvspan(-0.5, head_end - 0.5,            facecolor='#E3F2FD', alpha=0.3, zorder=0)
    ax.axvspan(head_end - 0.5, medium_end - 0.5, facecolor='#FFF3E0', alpha=0.3, zorder=0)
    ax.axvspan(medium_end - 0.5, n - 0.5,        facecolor='#FFEBEE', alpha=0.3, zorder=0)

    # Zone labels at top
    ymax = ax.get_ylim()[1]
    for label, x0, x1, color in [
        ('HEAD',   -0.5, head_end - 0.5,   '#1565C0'),
        ('MEDIUM', head_end - 0.5, medium_end - 0.5, '#E65100'),
        ('TAIL',   medium_end - 0.5, n - 0.5, '#B71C1C'),
    ]:
        if x1 > x0:
            ax.text((x0 + x1) / 2, ax.get_ylim()[1] * 0.97,
                    label, ha='center', va='top',
                    fontsize=9, color=color, fontweight='bold', alpha=0.7)

    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'{lbl}\n(train {tf:,})' for lbl, tf in zip(labels_sorted, train_freq_sorted)],
        rotation=45, ha='right', fontsize=6.5)

    ax.set_ylabel('Number of samples', fontsize=11)
    ax.set_xlim(-0.6, n - 0.4)
    split_label = 'Test Set' if args.split == 'test' else 'Dev Set (Val)'
    ax.set_title(
        f'Per-Class Positive Count vs Correctly Classified (TP) — CXR-LT 2024 {split_label}\n'
        f'DiCaP {args.net}  lb_ratio={args.lb_ratio}  threshold={args.threshold}  '
        f'(sorted high→low by train frequency)',
        fontsize=11, pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='#BBBBBB', edgecolor='#555555', label='Total positives (val)'),
        mpatches.Patch(facecolor='#555555', alpha=0.88, label=f'True positives TP (thr={args.threshold})'),
        mpatches.Patch(facecolor='#2196F3', alpha=0.85, label='Head classes (>10 % train prevalence)'),
        mpatches.Patch(facecolor='#FF9800', alpha=0.85, label='Medium classes (1–10 %)'),
        mpatches.Patch(facecolor='#F44336', alpha=0.85, label='Tail classes (<1 %)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    fname = (f'class_tp'
             f'_{args.dataset_name}'
             f'_lb{args.lb_ratio}'
             f'_{args.net}'
             f'_thr{args.threshold}.png')
    fpath = os.path.join(args.out_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved → {fpath}')

    # Print summary table
    overall_recall = true_pos.sum() / val_pos.sum() if val_pos.sum() > 0 else 0
    print(f'\n{"Class":<35} {"Val+":>7} {"TP":>7} {"Recall":>8}')
    print('-' * 62)
    for i in sort_idx:
        rc = true_pos[i] / val_pos[i] * 100 if val_pos[i] > 0 else 0
        print(f'{LABEL_COLS[i]:<35} {val_pos[i]:>7} {true_pos[i]:>7} {rc:>7.1f}%')
    print('-' * 62)
    print(f'{"TOTAL":<35} {val_pos.sum():>7} {true_pos.sum():>7} '
          f'{overall_recall*100:>7.1f}%')


if __name__ == '__main__':
    main()
