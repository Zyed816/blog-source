+++ 
draft = false
date = 2026-02-08T22:25:00+08:00
title = "[毕业设计]Day8"
description = ""
slug = "dissertation-day8"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

##  $\text{PubMed}$ 数据集

###  $\text{GRACE}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset PubMed --senario ID --gpu_id 0 --seed 39788
```

结果：

```bash
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.45 GiB (GPU 0; 4.00 GiB total capacity; 10.42 GiB already allocated; 0 bytes free; 10.48 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

由于 $\text{PubMed}$ 数据集结点数太大导致显存爆炸

将代码部署到 $\text{AutoDL}$ 运行

命令：

```bash
python main.py --model GRACE --dataset PubMed --senario ID --gpu_id 0 --seed 39788
```

结果：

```html
Seed: 39788
Dataset: PubMed
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
best val acc: 84.8231% ± 0.409805%
correspoding test acc: 84.3121% ± 0.329644%

Seed: 666
Dataset: PubMed
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
best val acc: 84.7006% ± 0.378443%
correspoding test acc: 84.0619% ± 0.299100%
```

###  $\text{IFL-GR}$

命令

```bash
python main.py --model GRACE --dataset PubMed --senario ID --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1 --batch_size 4096
```

结果：

```html
Seed: 39788
Dataset: PubMed
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
start_debias_epoch: 200
update_interval: 1
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 84.7471% ± 0.519791%
correspoding test acc: 84.1329% ± 0.285970%
```

###  $\text{GCA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset PubMed --senario ID --gpu_id 0 --seed 39788
```

结果：

```html
Seed: 39788
Dataset: PubMed
Model: GCA
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
drop_scheme: degree
best val acc: 84.0286% ± 0.135112%
correspoding test acc: 83.5649% ± 0.154123%
```

###   $\text{IFL-GC}$

命令

```bash
python main.py --model GCA --dataset PubMed --senario ID --gpu_id 0 --seed 39788 --drop_scheme degree --theroy_view --start_debias_epoch 100 --update_interval 20 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1 --batch_size 4096
```

结果

```html
Seed: 39788
Dataset: PubMed
Model: GCA
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
drop_scheme: degree
start_debias_epoch: 100
update_interval: 20
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 83.7834% ± 0.601778%
correspoding test acc: 83.3485% ± 0.167351%
```

###   $\text{DGI}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model DGI --dataset PubMed --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果：

```html
Seed: 39788
Dataset: PubMed
Model: DGI
Num_epochs: 500
Repeat_times: 3
best val acc: 85.3345% ± 0.041839%
correspoding test acc: 84.6671% ± 0.103549%
```
###  $\text{COSTA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model COSTA --dataset PubMed --senario ID --gpu_id 0 --seed 39788
```

结果：

```html
Seed: 39788
Dataset: PubMed
Model: COSTA
Num_epochs: 500
Repeat_times: 3
best val acc: 84.4935% ± 0.315877%
correspoding test acc: 84.1093% ± 0.178073%
```

###  $\text{BGRL}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model BGRL --dataset PubMed --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果：

```html
Seed: 39788
Dataset: PubMed
Model: BGRL
Num_epochs: 500
Repeat_times: 3
best val acc: 83.7200% ± 0.305002%
correspoding test acc: 83.5920% ± 0.270945%
```

###  $\text{MVGRL}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model MVGRL --dataset PubMed --senario ID --gpu_id 0 --seed 39788 --learning_rate 0.001 --gradient_drop_threshold 1e-5 --tolerance_epoch_num 20 --mvgrl_alpha 0.2
```

结果：

```html
Seed: 39788
Dataset: PubMed
Model: MVGRL
Num_epochs: 500
Repeat_times: 3
best val acc: 84.8612% ± 0.228927%
correspoding test acc: 84.5961% ± 0.189276%
```

###  $\text{GBT}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GBT --dataset PubMed --senario ID --gpu_id 0 --seed 39788 --weight_decay 0 --num_hidden 512 --batch_size 4 --p_x 0.1 --p_e 0.4 --gradient_drop_threshold 1e-3
```

结果：

```html
Seed: 39788
Dataset: PubMed
Model: GBT
Num_epochs: 500
Repeat_times: 3
best val acc: 86.2728% ± 0.177003%
correspoding test acc: 86.0364% ± 0.210547%
```


##  $\text{CiteSeer}$ 数据集

###  $\text{GRACE}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model CiteSeer --dataset PubMed --senario ID --gpu_id 0 --seed 39788
```

结果

```html
Seed: 39788
Dataset: CiteSeer
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
best val acc: 67.5432% ± 0.811513%
correspoding test acc: 66.5064% ± 1.090965%

Seed: 666
Dataset: CiteSeer
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
best val acc: 68.0691% ± 0.829854%
correspoding test acc: 65.1442% ± 2.487042%
```

###  $\text{IFL-GR}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset CiteSeer --senario ID --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```html
Dataset: CiteSeer
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
start_debias_epoch: 200
update_interval: 1
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 67.5432% ± 0.811513%
correspoding test acc: 66.5264% ± 1.104854%

Seed: 666
Dataset: CiteSeer
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
start_debias_epoch: 200
update_interval: 1
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 67.7686% ± 0.479117%
correspoding test acc: 67.2075% ± 0.455922%
```

###  $\text{GCA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset CiteSeer --senario ID --gpu_id 0 --seed 39788
```

结果

```html
Seed: 39788
Dataset: CiteSeer
Model: GCA
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
drop_scheme: degree
best val acc: 56.2735% ± 1.042858%
correspoding test acc: 54.7276% ± 1.204258%
```

###  $\text{IFL-GC}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset CiteSeer --senario ID --gpu_id 0 --seed 39788 --drop_scheme degree --theroy_view --start_debias_epoch 100 --update_interval 20 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```html
Seed: 39788
Dataset: CiteSeer
Model: GCA
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
drop_scheme: degree
start_debias_epoch: 100
update_interval: 20
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 57.8012% ± 0.937054%
correspoding test acc: 57.9928% ± 1.628151%

Seed: 666
Dataset: CiteSeer
Model: GCA
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
drop_scheme: degree
start_debias_epoch: 100
update_interval: 20
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 58.7779% ± 0.276618%
correspoding test acc: 56.9912% ± 0.381136%
```

###  $\text{DGI}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model DGI --dataset CiteSeer --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```html
Dataset: CiteSeer
Model: DGI
Num_epochs: 500
Repeat_times: 3
best val acc: 69.3714% ± 2.260890%
correspoding test acc: 67.6482% ± 2.238576%

Seed: 666
Dataset: CiteSeer
Model: DGI
Num_epochs: 500
Repeat_times: 3
best val acc: 67.9189% ± 0.973815%
correspoding test acc: 66.1458% ± 0.242048%
```

###  $\text{COSTA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model COSTA --dataset CiteSeer --senario ID --gpu_id 0 --seed 39788
```

结果

```html
Seed: 39788
Dataset: CiteSeer
Model: COSTA
Num_epochs: 500
Repeat_times: 3
best val acc: 69.3464% ± 1.476099%
correspoding test acc: 67.9487% ± 0.942997%

Seed: 666
Dataset: CiteSeer
Model: COSTA
Num_epochs: 500
Repeat_times: 3
best val acc: 68.3196% ± 0.876892%
correspoding test acc: 67.1875% ± 0.868107%
```

###  $\text{BGRL}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model BGRL --dataset CiteSeer --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```html
Seed: 39788
Dataset: CiteSeer
Model: BGRL
Num_epochs: 500
Repeat_times: 3
best val acc: 61.4826% ± 0.697640%
correspoding test acc: 62.4199% ± 0.185770%

Seed: 666
Dataset: CiteSeer
Model: BGRL
Num_epochs: 500
Repeat_times: 3
best val acc: 60.5810% ± 1.278462%
correspoding test acc: 59.8758% ± 1.743364%
```

###  $\text{MVGRL}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model MVGRL --dataset CiteSeer --senario ID --gpu_id 0 --seed 39788 --learning_rate 0.001 --gradient_drop_threshold 1e-5 --tolerance_epoch_num 20 --mvgrl_alpha 0.2
```

结果

```html
Seed: 39788
Dataset: CiteSeer
Model: MVGRL
Num_epochs: 500
Repeat_times: 3
best val acc: 70.5735% ± 0.492033%
correspoding test acc: 69.7516% ± 0.844678%

Seed: 666
Dataset: CiteSeer
Model: MVGRL
Num_epochs: 500
Repeat_times: 3
best val acc: 69.6218% ± 0.828341%
correspoding test acc: 69.4511% ± 0.723376%
```

###  $\text{GBT}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GBT --dataset CiteSeer --senario ID --gpu_id 0 --seed 39788 --weight_decay 0 --num_hidden 512 --batch_size 4 --p_x 0.1 --p_e 0.4 --gradient_drop_threshold 1e-3
```

结果

```html
Seed: 39788
Dataset: CiteSeer
Model: GBT
Num_epochs: 500
Repeat_times: 3
best val acc: 67.1926% ± 2.209541%
correspoding test acc: 66.9071% ± 1.198245%

Seed: 666
Dataset: CiteSeer
Model: GBT
Num_epochs: 500
Repeat_times: 3
best val acc: 66.3161% ± 0.872590%
correspoding test acc: 65.9255% ± 2.047389%
```

##  $\text{WiKiCS}$ 数据集

###   $\text{GRACE}$

命令

```bash
python main.py --model GRACE --dataset WikiCS --senario ID --gpu_id 0 --seed 39788
```

结果

```html
Seed: 39788
Dataset: WikiCS
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
best val acc: 79.7497% ± 0.468557%
correspoding test acc: 78.2225% ± 0.598408%

Seed: 666
Dataset: WikiCS
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
best val acc: 79.0734% ± 0.332162%
correspoding test acc: 77.6125% ± 0.526773%
```

###  $\text{IFL-GR}$

命令

```bash
python main.py --model GRACE --dataset WikiCS --senario ID --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```html
Seed: 39788
Dataset: WikiCS
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
start_debias_epoch: 200
update_interval: 1
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 79.7560% ± 0.588367%
correspoding test acc: 78.2624% ± 0.813896%

Seed: 666
Dataset: WikiCS
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
start_debias_epoch: 200
update_interval: 1
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 79.0734% ± 0.331078%
correspoding test acc: 77.6067% ± 0.521066%
```

###  $\text{GCA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset WikiCS --senario ID --gpu_id 0 --seed 39788
```

结果

```html
Seed: 39788
Dataset: WikiCS
Model: GCA
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
drop_scheme: degree
best val acc: 80.5208% ± 0.336464%
correspoding test acc: 78.9522% ± 0.530707%
```

###  $\text{IFL-GC}$

命令

```bash
python main.py --model GCA --dataset WikiCS --senario ID --gpu_id 0 --seed 39788 --drop_scheme degree --theroy_view --start_debias_epoch 100 --update_interval 20 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```html
Seed: 39788
Dataset: WikiCS
Model: GCA
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
drop_scheme: degree
start_debias_epoch: 100
update_interval: 20
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 80.4070% ± 0.380586%
correspoding test acc: 78.6614% ± 0.525538%
```

###  $\text{DGI}$

命令

```bash
python main.py --model DGI --dataset WikiCS --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```html
Seed: 39788
Dataset: WikiCS
Model: DGI
Num_epochs: 500
Repeat_times: 3
best val acc: 79.8319% ± 0.742523%
correspoding test acc: 78.5987% ± 0.954731%

Seed: 666
Dataset: WikiCS
Model: DGI
Num_epochs: 500
Repeat_times: 3
best val acc: 79.5285% ± 0.568898%
correspoding test acc: 78.0799% ± 0.696867%
```

###  $\text{COSTA}$

命令

```bash
python main.py --model COSTA --dataset WikiCS --senario ID --gpu_id 0 --seed 39788
```

结果

```html
Seed: 39788
Dataset: WikiCS
Model: COSTA
Num_epochs: 500
Repeat_times: 3
best val acc: 79.9899% ± 0.266954%
correspoding test acc: 78.4106% ± 0.858137%

Seed: 666
Dataset: WikiCS
Model: COSTA
Num_epochs: 500
Repeat_times: 3
best val acc: 79.7434% ± 0.266504%
correspoding test acc: 78.0685% ± 0.666399%
```

###  $\text{BGRL}$

命令

```bash
python main.py --model BGRL --dataset WikiCS --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```html
Seed: 39788
Dataset: WikiCS
Model: BGRL
Num_epochs: 500
Repeat_times: 3
best val acc: 79.1556% ± 0.340711%
correspoding test acc: 77.6923% ± 0.612260%

Seed: 666
Dataset: WikiCS
Model: BGRL
Num_epochs: 500
Repeat_times: 3
best val acc: 78.5678% ± 0.131669%
correspoding test acc: 77.1507% ± 0.073892%
```

###  $\text{MVGRL}$

命令

```bash
python main.py --model MVGRL --dataset WikiCS --senario ID --gpu_id 0 --seed 39788 --learning_rate 0.001 --gradient_drop_threshold 1e-5 --tolerance_epoch_num 20 --mvgrl_alpha 0.2
```

结果

```html
Seed: 39788
Dataset: WikiCS
Model: MVGRL
Num_epochs: 500
Repeat_times: 3
best val acc: 77.9927% ± 0.910301%
correspoding test acc: 76.3411% ± 0.890368%
```

###  $\text{GBT}$

命令

```bash
python main.py --model GBT --dataset WikiCS --senario ID --gpu_id 0 --seed 39788 --weight_decay 0 --num_hidden 512 --batch_size 4 --p_x 0.1 --p_e 0.4 --gradient_drop_threshold 1e-3
```

结果

```html
Seed: 39788
Dataset: WikiCS
Model: GBT
Num_epochs: 500
Repeat_times: 3
best val acc: 80.8178% ± 0.945051%
correspoding test acc: 79.3569% ± 0.935821%

Seed: 666
Dataset: WikiCS
Model: GBT
Num_epochs: 500
Repeat_times: 3
best val acc: 80.9948% ± 0.788546%
correspoding test acc: 79.3569% ± 0.819150%
```

##  $\text{Computers}$ 数据集
###   $\text{GRACE}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset Computers --senario ID --gpu_id 0 --seed 39788
```

结果

```html
Seed: 39788
Dataset: Computers
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
best val acc: 85.5057% ± 0.240403%
correspoding test acc: 85.7184% ± 1.009549%

Seed: 666
Dataset: Computers
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
best val acc: 86.5115% ± 0.821460%
correspoding test acc: 85.7136% ± 0.489218%
```

###  $\text{IFL-GR}$

命令

```bash
python main.py --model GRACE --dataset Computers --senario ID --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```html
Seed: 39788
Dataset: Computers
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
start_debias_epoch: 200
update_interval: 1
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 85.5178% ± 0.235776%
correspoding test acc: 85.7233% ± 1.015422%
```

###  $\text{GCA}$

命令

```bash
python main.py --model GCA --dataset Computers --senario ID --gpu_id 0 --seed 39788 --theroy_view
```

结果

```html
Seed: 39788
Dataset: Computers
Model: GCA
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
drop_scheme: degree
best val acc: 86.6751% ± 0.681310%
correspoding test acc: 86.9789% ± 0.158130%
```

###  $\text{IFL-GC}$

命令

```bash
python main.py --model GCA --dataset Computers --senario ID --gpu_id 0 --seed 39788 --drop_scheme degree --theroy_view --start_debias_epoch 100 --update_interval 20 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```html
Seed: 39788
Dataset: Computers
Model: GCA
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
drop_scheme: degree
start_debias_epoch: 100
update_interval: 20
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 85.3602% ± 0.668800%
correspoding test acc: 84.9137% ± 0.110334%
```

###  $\text{DGI}$

命令

```bash
python main.py --model DGI --dataset Computers --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```html
Seed: 39788
Dataset: Computers
Model: DGI
Num_epochs: 500
Repeat_times: 3
best val acc: 87.2205% ± 0.420863%
correspoding test acc: 86.8043% ± 0.557769%

Seed: 666
Dataset: Computers
Model: DGI
Num_epochs: 500
Repeat_times: 3
best val acc: 87.9355% ± 0.341812%
correspoding test acc: 87.6963% ± 0.408424%
```

###  $\text{COSTA}$

命令

```bash
python main.py --model COSTA --dataset Computers --senario ID --gpu_id 0 --seed 39788
```

结果

```html
Seed: 39788
Dataset: Computers
Model: COSTA
Num_epochs: 500
Repeat_times: 3
best val acc: 87.4205% ± 0.168581%
correspoding test acc: 86.8916% ± 0.406925%
```

###  $\text{BGRL}$

命令

```bash
python main.py --model BGRL --dataset Computers --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```html
Seed: 39788
Dataset: Computers
Model: BGRL
Num_epochs: 500
Repeat_times: 3
best val acc: 85.7299% ± 0.383907%
correspoding test acc: 85.8057% ± 0.611166%

```

###  $\text{MVGRL}$

命令

```bash
python main.py --model MVGRL --dataset Computers --senario ID --gpu_id 0 --seed 39788 --learning_rate 0.001 --gradient_drop_threshold 1e-5 --tolerance_epoch_num 20 --mvgrl_alpha 0.2
```

结果

```html
Seed: 39788
Dataset: Computers
Model: MVGRL
Num_epochs: 500
Repeat_times: 3
best val acc: 85.0997% ± 0.252907%
correspoding test acc: 84.1768% ± 0.257709%
```

###  $\text{GBT}$

命令

```bash
python main.py --model GBT --dataset Computers --senario ID --gpu_id 0 --seed 39788 --weight_decay 0 --num_hidden 512 --batch_size 4 --p_x 0.1 --p_e 0.4 --gradient_drop_threshold 1e-3
```

结果

```html
Seed: 39788
Dataset: Computers
Model: GBT
Num_epochs: 500
Repeat_times: 3
best val acc: 88.9838% ± 0.196350%
correspoding test acc: 89.1507% ± 0.542735%
```

##  $\text{Photo}$ 数据集
###   $\text{GRACE}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset Photo --senario ID --gpu_id 0 --seed 39788
```

结果

```html
Seed: 39788
Dataset: Photo
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
best val acc: 92.1133% ± 0.388814%
correspoding test acc: 91.5294% ± 0.246178%
```

###  $\text{IFL-GR}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset Photo --senario ID --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```html
Seed: 39788
Dataset: Photo
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
start_debias_epoch: 200
update_interval: 1
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 92.0915% ± 0.282385%
correspoding test acc: 91.3028% ± 0.177744%
```

###  $\text{GCA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset Photo --senario ID --gpu_id 0 --seed 39788
```

结果

```html
Seed: 39788
Dataset: Photo
Model: GCA
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
drop_scheme: degree
best val acc: 92.3965% ± 0.217320%
correspoding test acc: 91.7298% ± 0.312513%
```

###  $\text{IFL-GC}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset Photo --senario ID --gpu_id 0 --seed 39788 --drop_scheme degree --theroy_view --start_debias_epoch 100 --update_interval 20 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```html
Seed: 39788
Dataset: Photo
Model: GCA
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
drop_scheme: degree
start_debias_epoch: 100
update_interval: 20
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 91.7102% ± 0.239156%
correspoding test acc: 91.0850% ± 0.335484%
```

###  $\text{DGI}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model DGI --dataset Photo --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```html
Seed: 39788
Dataset: Photo
Model: DGI
Num_epochs: 500
Repeat_times: 3
best val acc: 92.1351% ± 0.242114%
correspoding test acc: 91.6078% ± 0.314451%
```

###  $\text{COSTA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model COSTA --dataset Photo --senario ID --gpu_id 0 --seed 39788
```

结果

```html
Seed: 39788
Dataset: Photo
Model: COSTA
Num_epochs: 500
Repeat_times: 3
best val acc: 92.4510% ± 0.240146%
correspoding test acc: 91.7908% ± 0.133308%
```

###  $\text{BGRL}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model BGRL --dataset Photo --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```html
Seed: 39788
Dataset: Photo
Model: BGRL
Num_epochs: 500
Repeat_times: 3
best val acc: 90.7190% ± 0.166635%
correspoding test acc: 90.2484% ± 0.246178%
```

###  $\text{MVGRL}$

命令

```bash
python main.py --model MVGRL --dataset Photo --senario ID --gpu_id 0 --seed 39788 --learning_rate 0.001 --gradient_drop_threshold 1e-5 --tolerance_epoch_num 20 --mvgrl_alpha 0.2
```

结果

```html
Seed: 39788
Dataset: Photo
Model: MVGRL
Num_epochs: 500
Repeat_times: 3
best val acc: 90.3050% ± 0.240640%
correspoding test acc: 89.6819% ± 0.121380%
```

###  $\text{GBT}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GBT --dataset Photo --senario ID --gpu_id 0 --seed 39788 --weight_decay 0 --num_hidden 512 --batch_size 4 --p_x 0.1 --p_e 0.4 --gradient_drop_threshold 1e-3
```

结果

```html
Seed: 39788
Dataset: Photo
Model: GBT
Num_epochs: 500
Repeat_times: 3
best val acc: 93.0610% ± 0.299911%
correspoding test acc: 92.7843% ± 0.548812%
```
