device_id=0
dataset_dir='./data'
net='chexfound'
output='cxr_ours'
#output='cxr_ours_weighted'
#class_weight_gamma=0.5   # 0.0=off, 0.5=sqrt reweighting, 1.0=full inverse-freq

for lb_ratio in 1.0
do
    for dataset_name in 'cxr'
    do

    CUDA_VISIBLE_DEVICES=$device_id python warm_up.py \
    --dataset_name $dataset_name --dataset_dir $dataset_dir --lb_ratio $lb_ratio \
    --net $net --loss_lb asl --warmup_epochs 12 --lr 1e-4 --output $output 
    # \
    # --class_weight_gamma $class_weight_gamma

    CUDA_VISIBLE_DEVICES=$device_id python main.py \
    --dataset_name $dataset_name --dataset_dir $dataset_dir --lb_ratio $lb_ratio \
    --net $net --loss_lb asl --main_epochs 25 --warmup_epochs 12 --lr 1e-4 \
    --output $output --method main --ub_epoch_size 50000
    # \
    # --class_weight_gamma $class_weight_gamma

    CUDA_VISIBLE_DEVICES=$device_id python fine_tune.py \
    --dataset_name $dataset_name --dataset_dir $dataset_dir --lb_ratio $lb_ratio \
    --net $net --loss_lb asl --output $output \
    --method main --FT_method fine_tune --FT_lr 1e-4 --FT_epochs 20 
    # \
    # --class_weight_gamma $class_weight_gamma

    # Per-class mAP evaluation + plot — runs automatically after fine_tune
    if [ "$dataset_name" = "cxr" ]; then
        python evaluate_cxr.py \
            --checkpoint ./output/${output}/${dataset_name}/${net}/${lb_ratio}/fine_tune/best_model.pth.tar \
            --dataset_dir ${dataset_dir}/${dataset_name} \
            --lb_ratio $lb_ratio \
            --net $net \
            2>&1 | tee ./output/${output}/${dataset_name}/${net}/${lb_ratio}/fine_tune/per_class_mAP.txt

        python plot_per_class_mAP.py \
            --checkpoint ./output/${output}/${dataset_name}/${net}/${lb_ratio}/fine_tune/best_model.pth.tar \
            --dataset_dir ${dataset_dir}/${dataset_name} \
            --lb_ratio $lb_ratio \
            --net $net \
            --out_dir ./output/${output}/${dataset_name}/${net}/${lb_ratio}

        python plot_class_tp.py \
            --checkpoint ./output/${output}/${dataset_name}/${net}/${lb_ratio}/fine_tune/best_model.pth.tar \
            --dataset_dir ${dataset_dir}/${dataset_name} \
            --lb_ratio $lb_ratio \
            --net $net \
            --out_dir ./output/${output}/${dataset_name}/${net}/${lb_ratio}
    fi
    done
done
