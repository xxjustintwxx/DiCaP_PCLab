#!/bin/bash
# Evaluate all four models on CXR-LT 2024 Task 2 gold standard test set (406 images, 26 classes).
# Run once: format the Task 2 data (creates formatted_task2_test_*.npy in data/cxr/)
#   python data/cxr/format_cxr2024_task2.py

device_id=0
net='tresnet_l'

# Model checkpoints and their output directories
declare -A CHECKPOINTS
CHECKPOINTS["cxr/0.066"]="output/cxr_ours/cxr/tresnet_l/0.066/fine_tune/best_model.pth.tar"
CHECKPOINTS["cxr_tail_lb/40.0"]="output/cxr_ours/cxr_tail_lb/tresnet_l/40.0/fine_tune/best_model.pth.tar"
CHECKPOINTS["cxr_hybrid_lb/40.0"]="output/cxr_ours/cxr_hybrid_lb/tresnet_l/40.0/fine_tune/best_model.pth.tar"
CHECKPOINTS["cxr/1.0"]="output/cxr_ours/cxr/tresnet_l/1.0/fine_tune/best_model.pth.tar"

declare -A DATASET_NAMES
DATASET_NAMES["cxr/0.066"]="cxr"
DATASET_NAMES["cxr_tail_lb/40.0"]="cxr_tail_lb"
DATASET_NAMES["cxr_hybrid_lb/40.0"]="cxr_hybrid_lb"
DATASET_NAMES["cxr/1.0"]="cxr"

declare -A LB_RATIOS
LB_RATIOS["cxr/0.066"]="0.066"
LB_RATIOS["cxr_tail_lb/40.0"]="40.0"
LB_RATIOS["cxr_hybrid_lb/40.0"]="40.0"
LB_RATIOS["cxr/1.0"]="1.0"

for key in "cxr/0.066" "cxr_tail_lb/40.0" "cxr_hybrid_lb/40.0" "cxr/1.0"
do
    ckpt="${CHECKPOINTS[$key]}"
    ds="${DATASET_NAMES[$key]}"
    lb="${LB_RATIOS[$key]}"
    out_dir="output/cxr_ours/${key}"

    echo "=========================================="
    echo "Evaluating: ${ds}  lb=${lb}"
    echo "Checkpoint: ${ckpt}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$device_id python evaluate_task2.py \
        --checkpoint $ckpt \
        --dataset_name $ds \
        --lb_ratio $lb \
        --net $net \
        2>&1 | tee ${out_dir}/task2_per_class_mAP.txt

    CUDA_VISIBLE_DEVICES=$device_id python plot_per_class_mAP_task2.py \
        --checkpoint $ckpt \
        --dataset_name $ds \
        --lb_ratio $lb \
        --net $net \
        --out_dir $out_dir

    CUDA_VISIBLE_DEVICES=$device_id python plot_class_tp_task2.py \
        --checkpoint $ckpt \
        --dataset_name $ds \
        --lb_ratio $lb \
        --net $net \
        --out_dir $out_dir

    echo ""
done

echo "All Task 2 evaluations complete."
