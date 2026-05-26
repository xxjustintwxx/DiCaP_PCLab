"""
Per-class mAP bar chart for CXR-LT 2024.

Generates a horizontal bar chart comparing your model's per-class AP against
the CXR-LT 2024 contest best model.

NOTE on best-model data:
  The per-class breakdown for the contest winner is not publicly released
  (it is only in the paper's supplementary materials).  What IS public is:
    - Overall mAP : 28.1 %
    - Head classes (>10 % prevalence)   : 56.7 %
    - Medium classes (1–10 %)           : 26.3 %
    - Tail classes (<1 %)               : 13.6 %
  These are shown as horizontal reference lines.
  To add real per-class best-model values, populate BEST_MODEL_AP below.

Usage (standalone):
    python plot_per_class_mAP.py \\
        --checkpoint output/cxr_ours/cxr/0.05/fine_tune/best_model.pth.tar \\
        --dataset_dir ./data/cxr \\
        --lb_ratio 0.05

Called automatically from evaluate_cxr.py — no args needed in that case.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Contest best-model per-class AP (Task 1, full test set).
# Populate with actual values if/when the supplementary data is released.
# Leave empty dict {} to show only aggregate reference lines.
# ---------------------------------------------------------------------------
BEST_MODEL_AP = {
    # Example format (fill in when data is available):
    # 'Adenopathy': 12.3,
    # 'Atelectasis': 55.1,
    # ...
}

# Aggregate reference lines from the published paper (Team A, Task 1)
BEST_OVERALL_MAP   = 28.1   # overall mAP
BEST_HEAD_MAP      = 56.7   # head classes  (prevalence > 10 %)
BEST_MEDIUM_MAP    = 26.3   # medium classes (prevalence 1–10 %)
BEST_TAIL_MAP      = 13.6   # tail classes  (prevalence < 1 %)

# Zone thresholds are computed dynamically from train+val combined counts
# (10% and 1% of 298,164 total images → HEAD=29,816 ; TAIL=2,982 → 9/15/16 split)
# These module-level placeholders are overwritten in run_inference_and_get_ap().
HEAD_THRESH   = 29816
TAIL_THRESH   = 2982

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
    parser.add_argument('--dataset_dir', default='./data/cxr', type=str)
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
    parser.add_argument('--out_dir', default='./output/plots', type=str,
                        help='directory to save the figure')
    return parser.parse_args()


def run_inference_and_get_ap(args):
    """Load model, run inference, return (ap_per_class, train_freq, class_names)."""
    import torch
    import torch.utils.data
    from sklearn.metrics import average_precision_score

    import _init_paths
    from lib.dataset.get_dataset import TransformWithPatches_Val
    from lib.dataset.handlers import CXR_handler
    from lib.ML_decoder.backbone.ML_decoder import ClasswiseModel
    from lib.utils.helper import clean_state_dict

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

    # Train+val combined for zone classification (10%/1% → 9/15/16 split)
    full_train_path = os.path.join(args.dataset_dir, 'formatted_full_train_labels.npy')
    train_freq_path = full_train_path if os.path.exists(full_train_path) else \
                      os.path.join(args.dataset_dir, 'formatted_train_labels.npy')
    train_lbl  = np.load(train_freq_path)
    val_lbl    = np.load(os.path.join(args.dataset_dir, 'formatted_val_labels.npy'))
    combined   = np.vstack([train_lbl, val_lbl])
    n_combined = len(combined)
    train_freq = combined.sum(axis=0)

    # Update module-level thresholds so plot() uses the same values
    global HEAD_THRESH, TAIL_THRESH
    HEAD_THRESH = n_combined * 0.10
    TAIL_THRESH = n_combined * 0.01

    ap_per_class = []
    for i in range(40):
        if targets[:, i].sum() > 0:
            ap = average_precision_score(targets[:, i], preds[:, i]) * 100
        else:
            ap = float('nan')
        ap_per_class.append(ap)

    return np.array(ap_per_class), train_freq, LABEL_COLS


def plot(ap_per_class, train_freq, class_names, args, out_dir):
    """Draw and save the per-class mAP bar chart."""
    os.makedirs(out_dir, exist_ok=True)

    n = len(class_names)
    sort_idx = np.argsort(train_freq)[::-1]          # head → tail

    labels_sorted = [class_names[i] for i in sort_idx]
    ap_sorted     = ap_per_class[sort_idx]
    freq_sorted   = train_freq[sort_idx]

    # Zone colours for each bar
    bar_colors = []
    for f in freq_sorted:
        if f >= HEAD_THRESH:
            bar_colors.append('#2196F3')   # blue  — head
        elif f >= TAIL_THRESH:
            bar_colors.append('#FF9800')   # orange — medium
        else:
            bar_colors.append('#F44336')   # red  — tail

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig_height = max(12, n * 0.38)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    y_pos = np.arange(n)

    # Your model bars
    bars = ax.barh(y_pos, ap_sorted, color=bar_colors, alpha=0.85,
                   height=0.6, label='DiCaP (yours)')

    # Best-model per-class dots (if data available)
    if BEST_MODEL_AP:
        best_ap_sorted = [BEST_MODEL_AP.get(class_names[i], np.nan) for i in sort_idx]
        valid = [(y, v) for y, v in zip(y_pos, best_ap_sorted) if not np.isnan(v)]
        if valid:
            ys, vs = zip(*valid)
            ax.scatter(vs, ys, color='black', zorder=5, s=30,
                       label='Contest best (per-class)')

    # ------------------------------------------------------------------
    # Best-model aggregate reference lines (zone averages)
    # ------------------------------------------------------------------
    head_end   = sum(1 for f in freq_sorted if f >= HEAD_THRESH)
    medium_end = sum(1 for f in freq_sorted if f >= TAIL_THRESH)

    def _hline(y0, y1, val, color, label, ls='--'):
        ax.plot([val, val], [y0 - 0.5, y1 - 0.5], color=color,
                linestyle=ls, linewidth=1.4, alpha=0.75)
        ax.text(val + 0.4, (y0 + y1) / 2 - 0.5, label,
                color=color, fontsize=7, va='center', alpha=0.9)

    # Overall best-model mAP (full-width dashed line) — labelled at top
    ax.axvline(BEST_OVERALL_MAP, color='black', linestyle=':', linewidth=1.5,
               alpha=0.6, label=f'Contest best overall mAP ({BEST_OVERALL_MAP:.1f})')
    ax.text(BEST_OVERALL_MAP, -0.85,
            f'Contest best\nmAP {BEST_OVERALL_MAP:.1f}%',
            color='black', fontsize=7.5, ha='center', va='bottom',
            alpha=0.85,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black',
                      alpha=0.6, linewidth=0.7))

    # Zone-level averages
    _hline(0,          head_end,   BEST_HEAD_MAP,   '#1565C0',
           f'Best head {BEST_HEAD_MAP:.1f}')
    _hline(head_end,   medium_end, BEST_MEDIUM_MAP, '#E65100',
           f'Best med {BEST_MEDIUM_MAP:.1f}')
    _hline(medium_end, n,          BEST_TAIL_MAP,   '#B71C1C',
           f'Best tail {BEST_TAIL_MAP:.1f}')

    # ------------------------------------------------------------------
    # Annotate your model's mAP per zone as vertical tick lines
    # ------------------------------------------------------------------
    your_head   = float(np.nanmean(ap_sorted[:head_end]))
    your_medium = float(np.nanmean(ap_sorted[head_end:medium_end]))
    your_tail   = float(np.nanmean(ap_sorted[medium_end:]))
    your_overall= float(np.nanmean(ap_sorted))

    _hline(0,          head_end,   your_head,   '#2196F3',
           f'Yours head {your_head:.1f}', ls='-.')
    _hline(head_end,   medium_end, your_medium, '#FF9800',
           f'Yours med {your_medium:.1f}', ls='-.')
    _hline(medium_end, n,          your_tail,   '#F44336',
           f'Yours tail {your_tail:.1f}', ls='-.')

    ax.axvline(your_overall, color='steelblue', linestyle='-.', linewidth=1.5,
               alpha=0.6, label=f'Your overall mAP ({your_overall:.1f})')
    ax.text(your_overall, -0.85,
            f'Yours\nmAP {your_overall:.1f}%',
            color='steelblue', fontsize=7.5, ha='center', va='bottom',
            alpha=0.95,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='steelblue',
                      alpha=0.6, linewidth=0.7))

    # Value labels on bars
    for bar, val in zip(bars, ap_sorted):
        if not np.isnan(val) and val > 0:
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', ha='left', fontsize=7.5)

    # Y-axis labels with training counts
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f'{lbl}  ({int(f):,})' for lbl, f in zip(labels_sorted, freq_sorted)],
        fontsize=8.5)
    ax.invert_yaxis()

    # Zone background shading
    ax.axhspan(-0.5, head_end - 0.5,   facecolor='#E3F2FD', alpha=0.25)
    ax.axhspan(head_end - 0.5, medium_end - 0.5, facecolor='#FFF3E0', alpha=0.25)
    ax.axhspan(medium_end - 0.5, n - 0.5,        facecolor='#FFEBEE', alpha=0.25)

    # Zone labels on the right margin
    ax.text(1.002, (0 + head_end) / 2 / n, 'HEAD',
            transform=ax.transAxes, fontsize=8, color='#1565C0',
            va='center', ha='left', rotation=90, alpha=0.7)
    ax.text(1.002, (head_end + medium_end) / 2 / n, 'MEDIUM',
            transform=ax.transAxes, fontsize=8, color='#E65100',
            va='center', ha='left', rotation=90, alpha=0.7)
    ax.text(1.002, (medium_end + n) / 2 / n, 'TAIL',
            transform=ax.transAxes, fontsize=8, color='#B71C1C',
            va='center', ha='left', rotation=90, alpha=0.7)

    # Legend
    legend_patches = [
        mpatches.Patch(color='#2196F3', alpha=0.85, label='Head classes (>10 % prevalence)'),
        mpatches.Patch(color='#FF9800', alpha=0.85, label='Medium classes (1–10 %)'),
        mpatches.Patch(color='#F44336', alpha=0.85, label='Tail classes (<1 %)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8)

    ax.set_xlabel('Average Precision (%)', fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_title(
        f'Per-Class AP — CXR-LT 2024\n'
        f'DiCaP {args.net}  lb_ratio={args.lb_ratio}  (overall mAP {your_overall:.1f} %)  '
        f'vs  Contest Best (mAP {BEST_OVERALL_MAP:.1f} %)',
        fontsize=11, pad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    fname = (f'per_class_mAP'
             f'_{args.dataset_name}'
             f'_lb{args.lb_ratio}'
             f'_{args.net}.png')
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved → {fpath}')
    return fpath


def main():
    args = get_args()
    print('Running inference to collect per-class AP...')
    ap_per_class, train_freq, class_names = run_inference_and_get_ap(args)

    # Print table (same as evaluate_cxr.py)
    sort_idx = np.argsort(train_freq)[::-1]
    print(f'\n{"Class":<35} {"Train+Val #":>11}  {"AP":>6}')
    print('-' * 58)
    for i in sort_idx:
        if train_freq[i] <= TAIL_THRESH:
            marker = ' <-- tail'
        elif train_freq[i] <= HEAD_THRESH:
            marker = ' <-- medium'
        else:
            marker = ''
        ap_str = f'{ap_per_class[i]:.1f}' if not np.isnan(ap_per_class[i]) else '  N/A'
        print(f'{class_names[i]:<35} {train_freq[i]:>11.0f}  {ap_str:>6}{marker}')
    valid_aps = ap_per_class[~np.isnan(ap_per_class)]
    head_mask = train_freq > HEAD_THRESH
    mid_mask  = (train_freq > TAIL_THRESH) & (train_freq <= HEAD_THRESH)
    tail_mask = train_freq <= TAIL_THRESH
    print('-' * 58)
    print(f'Overall mAP:              {valid_aps.mean():.2f}')
    print(f'Head mAP   (>10%, {head_mask.sum()} cls): {np.nanmean(ap_per_class[head_mask]):.2f}')
    print(f'Medium mAP (1-10%, {mid_mask.sum()} cls): {np.nanmean(ap_per_class[mid_mask]):.2f}')
    print(f'Tail mAP   (≤1%,  {tail_mask.sum()} cls): {np.nanmean(ap_per_class[tail_mask]):.2f}')

    # Save plot
    out_dir = args.out_dir
    plot(ap_per_class, train_freq, class_names, args, out_dir)


if __name__ == '__main__':
    main()
