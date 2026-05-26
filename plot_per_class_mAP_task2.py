"""
Per-class mAP bar chart for CXR-LT 2024 Task 2 (gold standard, 26 classes).

Contest best mAP on Task 2: 52.6% (overall; per-zone breakdown not published).

Usage:
    python plot_per_class_mAP_task2.py \\
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

# Contest best (Task 2, overall mAP only — zone breakdown not publicly available)
BEST_OVERALL_MAP = 52.6

# 26 Task 2 classes in CSV column order
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
    parser.add_argument('--out_dir', default='./output/plots', type=str)
    return parser.parse_args()


def run_inference_and_get_ap(args):
    import torch
    import torch.utils.data
    from sklearn.metrics import average_precision_score
    import _init_paths
    from lib.dataset.get_dataset import TransformWithPatches_Val
    from lib.dataset.handlers import CXR_handler
    from lib.ML_decoder.backbone.ML_decoder import ClasswiseModel
    from lib.utils.helper import clean_state_dict

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

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs_list, targets, _ in loader:
            inputs = inputs_list[0].cuda()
            outputs = torch.sigmoid(model(inputs))
            all_preds.append(outputs[:, MODEL_TASK2_INDICES].cpu().numpy())
            all_targets.append(targets.numpy())

    preds   = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    train_lbl = np.load('./data/cxr/formatted_train_labels.npy')
    val_lbl   = np.load('./data/cxr/formatted_val_labels.npy')
    combined  = np.vstack([train_lbl, val_lbl])
    train_freq = combined[:, MODEL_TASK2_INDICES].sum(axis=0)

    global HEAD_THRESH, TAIL_THRESH
    HEAD_THRESH = len(combined) * 0.10
    TAIL_THRESH = len(combined) * 0.01

    ap_per_class = []
    for i in range(len(TASK2_LABEL_COLS)):
        if targets[:, i].sum() > 0:
            ap = average_precision_score(targets[:, i], preds[:, i]) * 100
        else:
            ap = float('nan')
        ap_per_class.append(ap)

    return np.array(ap_per_class), train_freq, TASK2_LABEL_COLS


def plot(ap_per_class, train_freq, class_names, args, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    n = len(class_names)
    sort_idx = np.argsort(train_freq)[::-1]

    labels_sorted = [class_names[i] for i in sort_idx]
    ap_sorted     = ap_per_class[sort_idx]
    freq_sorted   = train_freq[sort_idx]

    bar_colors = []
    for f in freq_sorted:
        if f >= HEAD_THRESH:
            bar_colors.append('#2196F3')
        elif f >= TAIL_THRESH:
            bar_colors.append('#FF9800')
        else:
            bar_colors.append('#F44336')

    fig_height = max(10, n * 0.42)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    y_pos = np.arange(n)

    bars = ax.barh(y_pos, ap_sorted, color=bar_colors, alpha=0.85, height=0.6)

    # Contest best mAP overall reference line
    ax.axvline(BEST_OVERALL_MAP, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.text(BEST_OVERALL_MAP, -0.85,
            f'Contest best\nmAP {BEST_OVERALL_MAP:.1f}%\n(Task 2)',
            color='black', fontsize=7.5, ha='center', va='bottom', alpha=0.85,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black',
                      alpha=0.6, linewidth=0.7))

    head_end   = sum(1 for f in freq_sorted if f >= HEAD_THRESH)
    medium_end = sum(1 for f in freq_sorted if f >= TAIL_THRESH)

    your_head    = float(np.nanmean(ap_sorted[:head_end]))
    your_medium  = float(np.nanmean(ap_sorted[head_end:medium_end]))
    your_tail    = float(np.nanmean(ap_sorted[medium_end:]))
    your_overall = float(np.nanmean(ap_sorted[~np.isnan(ap_sorted)]))

    def _hline(y0, y1, val, color, label, ls='-.'):
        ax.plot([val, val], [y0 - 0.5, y1 - 0.5], color=color,
                linestyle=ls, linewidth=1.4, alpha=0.75)
        ax.text(val + 0.4, (y0 + y1) / 2 - 0.5, label,
                color=color, fontsize=7, va='center', alpha=0.9)

    _hline(0,          head_end,   your_head,   '#2196F3', f'Head {your_head:.1f}')
    _hline(head_end,   medium_end, your_medium, '#FF9800', f'Med {your_medium:.1f}')
    _hline(medium_end, n,          your_tail,   '#F44336', f'Tail {your_tail:.1f}')

    ax.axvline(your_overall, color='steelblue', linestyle='-.', linewidth=1.5, alpha=0.6)
    ax.text(your_overall, -0.85,
            f'Yours\nmAP {your_overall:.1f}%',
            color='steelblue', fontsize=7.5, ha='center', va='bottom', alpha=0.95,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='steelblue',
                      alpha=0.6, linewidth=0.7))

    for bar, val in zip(bars, ap_sorted):
        if not np.isnan(val) and val > 0:
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', ha='left', fontsize=7.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f'{lbl}  ({int(f):,})' for lbl, f in zip(labels_sorted, freq_sorted)],
        fontsize=8.5)
    ax.invert_yaxis()

    ax.axhspan(-0.5, head_end - 0.5,   facecolor='#E3F2FD', alpha=0.25)
    ax.axhspan(head_end - 0.5, medium_end - 0.5, facecolor='#FFF3E0', alpha=0.25)
    ax.axhspan(medium_end - 0.5, n - 0.5,        facecolor='#FFEBEE', alpha=0.25)

    ax.text(1.002, (0 + head_end) / 2 / n, 'HEAD',
            transform=ax.transAxes, fontsize=8, color='#1565C0',
            va='center', ha='left', rotation=90, alpha=0.7)
    ax.text(1.002, (head_end + medium_end) / 2 / n, 'MEDIUM',
            transform=ax.transAxes, fontsize=8, color='#E65100',
            va='center', ha='left', rotation=90, alpha=0.7)
    ax.text(1.002, (medium_end + n) / 2 / n, 'TAIL',
            transform=ax.transAxes, fontsize=8, color='#B71C1C',
            va='center', ha='left', rotation=90, alpha=0.7)

    legend_patches = [
        mpatches.Patch(color='#2196F3', alpha=0.85, label='Head (>10% train prevalence)'),
        mpatches.Patch(color='#FF9800', alpha=0.85, label='Medium (1–10%)'),
        mpatches.Patch(color='#F44336', alpha=0.85, label='Tail (<1%)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8)

    ax.set_xlabel('Average Precision (%)', fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_title(
        f'Per-Class AP — CXR-LT 2024 Task 2 (Gold Standard, 406 images, 26 classes)\n'
        f'DiCaP {args.net}  {args.dataset_name}  lb={args.lb_ratio}'
        f'  (overall mAP {your_overall:.1f}%)  vs  Contest Best ({BEST_OVERALL_MAP:.1f}%)',
        fontsize=10, pad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    fname = f'task2_per_class_mAP_{args.dataset_name}_lb{args.lb_ratio}_{args.net}.png'
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved → {fpath}')
    return fpath


def main():
    args = get_args()
    print('Running Task 2 inference...')
    ap_per_class, train_freq, class_names = run_inference_and_get_ap(args)

    sort_idx = np.argsort(train_freq)[::-1]
    print(f'\n{"Class":<35} {"Train+Val #":>11}  {"AP":>6}')
    print('-' * 58)
    for i in sort_idx:
        f = train_freq[i]
        marker = ' <-- tail' if f <= TAIL_THRESH else (' <-- medium' if f <= HEAD_THRESH else '')
        ap_str = f'{ap_per_class[i]:.1f}' if not np.isnan(ap_per_class[i]) else '  N/A'
        print(f'{class_names[i]:<35} {f:>11.0f}  {ap_str:>6}{marker}')

    head_mask = train_freq > HEAD_THRESH
    mid_mask  = (train_freq > TAIL_THRESH) & ~head_mask
    tail_mask = train_freq <= TAIL_THRESH
    print('-' * 58)
    print(f'Overall mAP:      {np.nanmean(ap_per_class):.2f}')
    print(f'Head mAP  (>10%): {np.nanmean(ap_per_class[head_mask]):.2f}  ({head_mask.sum()} classes)')
    print(f'Medium mAP (1-10%): {np.nanmean(ap_per_class[mid_mask]):.2f}  ({mid_mask.sum()} classes)')
    print(f'Tail mAP  (≤1%):  {np.nanmean(ap_per_class[tail_mask]):.2f}  ({tail_mask.sum()} classes)')

    plot(ap_per_class, train_freq, class_names, args, args.out_dir)


if __name__ == '__main__':
    main()
