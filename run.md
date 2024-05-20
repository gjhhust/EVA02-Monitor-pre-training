<!-- python run_class_finetuning.py \
        --data_path /data/jiahaoguo/dataset/jiankong/train \
        --disable_eval_during_finetuning \
        --nb_classes 21841 \
        --data_set muti_label \
        --output_dir output/eva02_B_pt_in21k_p14_ft_20 \
        --log_dir output/eva02_B_pt_in21k_p14_ft_20/tb_log \
        --model eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE \
        --finetune weight/eva02_B_pt_in21k_p14.pt \
        --input_size 640 \
        --lr 3e-4 \
        --warmup_lr 0.0 \
        --min_lr 0.0 \
        --layer_decay 0.7 \
        --epochs 20 \
        --warmup_epochs 1 \
        --drop_path 0.1 \
        --reprob 0.0 \
        --mixup 0.0 \
        --cutmix 0.0 \
        --batch_size 1 \
        --update_freq 2 \
        --crop_pct 1.0 \
        --zero_stage 1 \
        --partial_freeze 0 \
        --smoothing 0.1 \
        --weight_decay 0.05 \
        --scale 0.2 1.0 \
        --aa rand-m9-mstd0.5-inc1 \
        --enable_deepspeed \
        --muti_lables \
        --freeze_backbone
 -->

python  run_class_finetuning.py \
        --data_path /data/jiahaoguo/dataset/jiankong/train \
        --label_dir /data/jiahaoguo/dataset/jiankong/new_gt/train \
        --eval_data_path /data/jiahaoguo/dataset/jiankong \
        --disable_eval_during_finetuning \
        --nb_classes 87 \
        --data_set muti_label \
        --output_dir output/eva02_L_pt_m38m_p14_ft_20_448_newgt_87 \
        --log_dir output/eva02_L_pt_m38m_p14_ft_20_448_newgt_87/tb_log \
        --model eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE \
        --finetune weight/eva02_L_pt_m38m_p14.pt \
        --input_size 448 \
        --lr 3e-4 \
        --warmup_lr 0.0 \
        --min_lr 0.0 \
        --layer_decay 0.75 \
        --epochs 20 \
        --warmup_epochs 1 \
        --drop_path 0.15 \
        --reprob 0.0 \
        --mixup 0.0 \
        --cutmix 0.0 \
        --batch_size 10 \
        --update_freq 2 \
        --crop_pct 1.0 \
        --zero_stage 1 \
        --partial_freeze 0 \
        --smoothing 0.1 \
        --weight_decay 0.05 \
        --scale 0.2 1.0 \
        --aa rand-m9-mstd0.5-inc1 \
        --enable_deepspeed \
        --muti_lables 


# eval
python run_class_finetuning.py \
        --data_path /data/jiahaoguo/dataset/jiankong/dataset/test \
        --label_dir /data/jiahaoguo/dataset/jiankong/new_gt/test \
        --eval_data_path /data/jiahaoguo/dataset/jiankong/dataset/test \
        --nb_classes 87 \
        --data_set muti_label \
        --muti_lables \
        --model eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE \
        --finetune output/eva02_L_pt_m38m_p14_ft_20_448_newgt_87/checkpoint-19/mp_rank_00_model_states.pt \
        --input_size 448 \
        --batch_size 16 \
        --crop_pct 1.0 \
        --no_auto_resume \
        --eval \
        --enable_deepspeed \
        --plot \
        --plot_dir output/eva02_L_pt_m38m_p14_ft_20_448_newgt_87/show



# test1(MultiLabelSoftMarginLoss loss)
'mean_acc(0.5-0.9)': '0.637', 'mean_rec(0.5-0.9)': '0.483', 'mean_f1(0.5-0.9)': '0.547'
# new
'mean_acc(0.5-0.9)': '0.727', 'mean_rec(0.5-0.9)': '0.532', 'mean_f1(0.5-0.9)': '0.613'
# test1(BCELoss loss)

