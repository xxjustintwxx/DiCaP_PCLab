"""
Per-class TP count bar chart for CXR-LT 2024 Task 2 (gold standard, 26 classes).

Shows total positives vs correctly classified (TP at threshold=0.5) on the
406-image gold standard test set, sorted high→low by training frequency.

Usage:
    python plot_class_tp_task2.py \\
        --checkpoint output/cxr_ours/cxr_hybrid_lb/tresnet_l/40.0/fine_tune/best_model.pth.tar \\
        --dataset_name cxr_hybrid_lb --lb_ratio 40.0 --net tresnet_l \\
        --out_dir output/cxr_ours/cxr_hybrid_lb/tresnet_l/40.0
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

TASK2_LABEL_COLS = [
    'Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly', 'Consolidation',
    'Edema', 'Emphysema', 'Enlarged Cardiomediastinum', 'Fibrosis', 'Fracture',
    'Hernia', 'Infiltration', 'Lung Lesion', 'Lung Opacity', 'Mass', 'Normal',
    'Nodule', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening',
    'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax',
    'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta',
]

MODEL_TASK2_INDICES = [
    1, 3, 4, 6, 7, 8, 9, 10, 12, 14, 17, 20, 21, 22, 24, 23,
    25, 26, 27, 28, 29, 30, 31, 36, 37, 38,
]

HEAD_THRESH = 29816
TAIL_THRESH = 2982


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset_dir', default='./data/cxr')
    parser.add_argument('--dataset_name', default='cxr')
    parser.add_argument('--lb_ratio', default=0.05, type=float)
    parser.add_argument('--net', default='resnet50')
    parser.add_argument('--threshold', default=0.5, type=float)
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


def main():
    args = get_args()
    args.n_classes = 40

    task2_dir = './data/cxr'
    test_images = np.load(os.path.join(task2_dir, 'formatted_task2_test_images.npy'),
                          allow_pickle=True)
    test_labels = np.load(os.path.join(task2_dir, 'formatted_task2_test_labels.npy'))

    eval_dataset = CXR_handler(test_images, test_labels, task2_dir,
                               transform=TransformWithPatches_Val(args))
    loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = ClasswiseModel(args.net, args.n_classes, args.dim_embed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    ema_key = 'state_dict_ema' if 'state_dict_ema' in checkpoint else 'state_dict'
    state = clean_state_dict(checkpoint.get(ema_key, checkpoint))
    model.load_state_dict(state, strict=True)
    model = model.cuda()
    model.eval()

    print(f'Loading checkpoint: {args.checkpoint}')

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs_list, targets, _ in loader:
            inputs = inputs_list[0].cuda()
            outputs = torch.sigmoid(model(inputs))
            all_preds.append(outputs[:, MODEL_TASK2_INDICES].cpu().numpy())
            all_targets.append(targets.numpy())

    preds   = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    pred_bin = (preds >= args.threshold).astype(np.float32)

    val_pos  = targets.sum(axis=0).astype(int)
    true_pos = (pred_bin * targets).sum(axis=0).astype(int)
    recall   = np.where(val_pos > 0, true_pos / val_pos, 0.0)

    train_lbl = np.load('./data/cxr/formatted_train_labels.npy')
    val_lbl   = np.load('./data/cxr/formatted_val_labels.npy')
    combined  = np.vstack([train_lbl, val_lbl])
    train_freq = combined[:, MODEL_TASK2_INDICES].sum(axis=0).astype(int)

    global HEAD_THRESH, TAIL_THRESH
    HEAD_THRESH = int(len(combined) * 0.10)
    TAIL_THRESH = int(len(combined) * 0.01)

    sort_idx = np.argsort(train_freq)[::-1]
    labels_sorted     = [TASK2_LABEL_COLS[i] for i in sort_idx]
    val_pos_sorted    = val_pos[sort_idx]
    true_pos_sorted   = true_pos[sort_idx]
    recall_sorted     = recall[sort_idx]
    train_freq_sorted = train_freq[sort_idx]

    zone_colors = []
    for f in train_freq_sorted:
        if f >= HEAD_THRESH:
            zone_colors.append('#2196F3')
        elif f >= TAIL_THRESH:
            zone_colors.append('#FF9800')
        else:
            zone_colors.append('#F44336')

    n = len(TASK2_LABEL_COLS)
    fig_width = max(18, n * 0.72)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    x = np.arange(n)
    bar_w = 0.38

    ax.bar(x - bar_w / 2, val_pos_sorted,
           width=bar_w, label='Total positives (test)',
           color=[c + '55' for c in zone_colors],
           edgecolor=zone_colors, linewidth=0.8)
    ax.bar(x + bar_w / 2, true_pos_sorted,
           width=bar_w, label=f'True positives TP (thr={args.threshold})',
           color=zone_colors, alpha=0.88)

    for xi, (tp, vp, rc) in enumerate(zip(true_pos_sorted, val_pos_sorted, recall_sorted)):
        if vp > 0:
            ax.text(xi + bar_w / 2, tp + max(val_pos_sorted) * 0.005,
                    f'{rc*100:.0f}%', ha='center', va='bottom',
                    fontsize=6, color='#333333')

    head_end   = sum(1 for f in train_freq_sorted if f >= HEAD_THRESH)
    medium_end = sum(1 for f in train_freq_sorted if f >= TAIL_THRESH)

    ax.axvspan(-0.5, head_end - 0.5,            facecolor='#E3F2FD', alpha=0.3, zorder=0)
    ax.axvspan(head_end - 0.5, medium_end - 0.5, facecolor='#FFF3E0', alpha=0.3, zorder=0)
    ax.axvspan(medium_end - 0.5, n - 0.5,        facecolor='#FFEBEE', alpha=0.3, zorder=0)

    for label, x0, x1, color in [
        ('HEAD',   -0.5, head_end - 0.5,   '#1565C0'),
        ('MEDIUM', head_end - 0.5, medium_end - 0.5, '#E65100'),
        ('TAIL',   medium_end - 0.5, n - 0.5, '#B71C1C'),
    ]:
        if x1 > x0:
            ax.text((x0 + x1) / 2, ax.get_ylim()[1] * 0.97,
                    label, ha='center', va='top',
                    fontsize=9, color=color, fontweight='bold', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'{lbl}\n(train {tf:,})' for lbl, tf in zip(labels_sorted, train_freq_sorted)],
        rotation=45, ha='right', fontsize=6.5)

    ax.set_ylabel('Number of samples', fontsize=11)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_title(
        f'Per-Class Positive Count vs TP — CXR-LT 2024 Task 2 (Gold Standard, 406 images)\n'
        f'DiCaP {args.net}  {args.dataset_name}  lb={args.lb_ratio}  threshold={args.threshold}',
        fontsize=11, pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor='#BBBBBB', edgecolor='#555555', label='Total positives (test)'),
        mpatches.Patch(facecolor='#555555', alpha=0.88, label=f'True positives (thr={args.threshold})'),
        mpatches.Patch(facecolor='#2196F3', alpha=0.85, label='Head (>10% train prevalence)'),
        mpatches.Patch(facecolor='#FF9800', alpha=0.85, label='Medium (1–10%)'),
        mpatches.Patch(facecolor='#F44336', alpha=0.85, label='Tail (<1%)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    fname = f'task2_class_tp_{args.dataset_name}_lb{args.lb_ratio}_{args.net}_thr{args.threshold}.png'
    fpath = os.path.join(args.out_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved → {fpath}')

    # Summary table
    overall_recall = true_pos.sum() / val_pos.sum() if val_pos.sum() > 0 else 0
    print(f'\n{"Class":<35} {"Test+":>6} {"TP":>6} {"Recall":>8}')
    print('-' * 60)
    for i in sort_idx:
        rc = true_pos[i] / val_pos[i] * 100 if val_pos[i] > 0 else 0
        print(f'{TASK2_LABEL_COLS[i]:<35} {val_pos[i]:>6} {true_pos[i]:>6} {rc:>7.1f}%')
    print('-' * 60)
    print(f'{"TOTAL":<35} {val_pos.sum():>6} {true_pos.sum():>6} '
          f'{overall_recall*100:>7.1f}%')


if __name__ == '__main__':
    main()
