device_id=0
dataset_dir='./data'               # training scripts append dataset_name → ./data/cxr_hybrid_lb
eval_dataset_dir='./data/cxr_hybrid_lb'  # evaluate/plot scripts use this directly
dataset_name='cxr_hybrid_lb'
net='tresnet_l'
output='cxr_ours'

# lb_ratio is ignored in pre-split mode (formatted_unlabeled_images.npy exists),
# but kept at 1.0 as a sentinel so checkpoint paths stay interpretable.
lb_ratio=40.0

CUDA_VISIBLE_DEVICES=$device_id python warm_up.py \
--dataset_name $dataset_name --dataset_dir $dataset_dir --lb_ratio $lb_ratio \
--net $net --loss_lb asl --warmup_epochs 12 --lr 1e-4 --output $output

CUDA_VISIBLE_DEVICES=$device_id python main.py \
--dataset_name $dataset_name --dataset_dir $dataset_dir --lb_ratio $lb_ratio \
--net $net --loss_lb asl --main_epochs 40 --warmup_epochs 12 --lr 1e-4 \
--output $output --method main --ub_epoch_size 50000

CUDA_VISIBLE_DEVICES=$device_id python fine_tune.py \
--dataset_name $dataset_name --dataset_dir $dataset_dir --lb_ratio $lb_ratio \
--net $net --loss_lb asl --output $output \
--method main --FT_method fine_tune --FT_lr 1e-4 --FT_epochs 20

# Per-class mAP evaluation + plot
python evaluate_cxr.py \
    --checkpoint ./output/${output}/${dataset_name}/${net}/${lb_ratio}/fine_tune/best_model.pth.tar \
    --dataset_dir ${eval_dataset_dir} \
    --lb_ratio $lb_ratio \
    --net $net \
    2>&1 | tee ./output/${output}/${dataset_name}/${net}/${lb_ratio}/fine_tune/per_class_mAP.txt

python plot_per_class_mAP.py \
    --checkpoint ./output/${output}/${dataset_name}/${net}/${lb_ratio}/fine_tune/best_model.pth.tar \
    --dataset_dir ${eval_dataset_dir} \
    --lb_ratio $lb_ratio \
    --net $net \
    --out_dir ./output/${output}/${dataset_name}/${net}/${lb_ratio}

python plot_class_tp.py \
    --checkpoint ./output/${output}/${dataset_name}/${net}/${lb_ratio}/fine_tune/best_model.pth.tar \
    --dataset_dir ${eval_dataset_dir} \
    --lb_ratio $lb_ratio \
    --net $net \
    --out_dir ./output/${output}/${dataset_name}/${net}/${lb_ratio}
