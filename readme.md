## install

First, clone the repo and install required packages:
```bash
conda create --name asuka python=3.8 -y
conda activate asuka

git clone git@github.com:baaivision/EVA.git
cd EVA-02/asuka
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

Then, install [Apex](https://github.com/NVIDIA/apex#linux) and [xFormer](https://github.com/facebookresearch/xformers#installing-xformers) following the official instruction. 


Core packages: 
- [Pytorch](https://pytorch.org/) version 1.12.1 
- [torchvision](https://pytorch.org/vision/stable/index.html) version 0.13.0
- [timm](https://github.com/rwightman/pytorch-image-models) version 0.5.4 
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) version 0.6.5 (`fp16` training and ZeRO optimizer), fine-tuning with `bfloat16` requires version 0.8.1
- [Apex](https://github.com/NVIDIA/apex) (fused layer norm)
- [xFormer](https://github.com/facebookresearch/xformers) (fast and memory efficient MHSA)



## 准备数据集

数据集组织只需要满足如下：

1. 图片结尾是jpg和png
2. 标注和图片的相对地址一致，后缀为txt
3. txt标注格式： {"makeLabels":[{"code":4007,"name":"车辆出入口","index":0}, {"code":3012,"name":"机场","index":1}, {"code":2013,"name":"小路口","index":2}],"imagePath":"图片的相对地址"} 
4. class_map.json : 字典格式文件包含所有的分类，key为数字格式的分类名code，value是对应中文的分类名name

```
data_root_dir/               # Root data directory
├── class_map.json         
├── train.txt        
├── images/              
│   ├── video1/          
│   │   ├── 0000000.png    
│   │   └── 0000001.png    
│   ├── video2/            
│   │   └── ...        
│   └── ...            
└── labels/          
    ├── video1/         
    │   ├── 0000000.txt    
    │   └── 0000001.txt   
    ├── video2/          
    │   └── ...      
    └── ...      
```

note. label_dir即为标注目录，如果为空则表示标注和图片保存在同一目录下，否则将会通过替换data_dir为label_dir，再修改后缀txt来找到标注文件

## train

主要参数说明；

data_path: 训练集的根目录

label_dir： 训练集的标注根目录（注意，标注和图片相对地址必须一致，比如/xx/xx/video1/image1.png，其标注为/yy/yy/yy/video1/image1.txt，则data_path=/xx/xx   label_dir=/yy/yy/yy）

classes_json：类别映射的字典，格式是  number_index: 分类名

nb_classes：类别数量

output_dir、log_dir、input_size、epochs按需修改


```bash
eva02_L_pt_m38m_p14.pt作为预训练模型： https://gist.github.com/Yuxin-CV/6491f01a3a7a2f31115fb7a7a19c7148#file-eva02_l_pt_m38m_p14

python  run_class_finetuning.py \
        --data_path /data/jiahaoguo/dataset/jiankong/train \
        --label_dir /data/jiahaoguo/dataset/jiankong/new_gt/train \
        --classes_json /data/jiahaoguo/dataset/jiankong/classes_map.json \
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
```


## eval

主要参数说明；

eval_data_path: 测试集的根目录

label_dir： 测试集的标注根目录

classes_json：类别映射的字典，格式是  number_index: 分类名

nb_classes：类别数量

add_mode: 可选save_pred和plot，plot会在add_mode_dir下绘制可视化，save_pred则保存注释，并标注+ - x


```bash

python run_class_finetuning.py \
        --data_path /data/jiahaoguo/dataset/jiankong/dataset/test \
        --label_dir /data/jiahaoguo/dataset/jiankong/new_gt/test \
        --eval_data_path /data/jiahaoguo/dataset/jiankong/dataset/test \
        --classes_json /data/jiahaoguo/dataset/jiankong/classes_map.json \
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
        --add_mode plot \
        --add_mode_dir output/eva02_L_pt_m38m_p14_ft_20_448_newgt_87/show

```
