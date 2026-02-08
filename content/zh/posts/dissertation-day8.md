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

```
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
python main.py --model GRACE --dataset PubMed --senario ID --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果：再次爆显存

##  $\text{CiteSeer}$ 数据集

###  $\text{GRACE}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model CiteSeer --dataset PubMed --senario ID --gpu_id 0 --seed 39788
```

结果

```
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

```
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
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset CiteSeer --senario ID --gpu_id 0 --seed 39788 --theroy_view
```

结果

```
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
start_debias_epoch: 200
update_interval: 1
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 56.4989% ± 0.928312%
correspoding test acc: 56.3101% ± 1.957204%

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
start_debias_epoch: 200
update_interval: 1
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 58.8279% ± 0.373145%
correspoding test acc: 56.8910% ± 1.678633%
```

###  $\text{IFL-GC}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset CiteSeer --senario ID --gpu_id 0 --seed 39788 --drop_scheme degree --theroy_view --start_debias_epoch 100 --update_interval 20 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```
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

```
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

```
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

```
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

```
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

```
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
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset WikiCS --senario ID --gpu_id 0 --seed 39788
```

结果

```

```

###  $\text{IFL-GR}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset WiKiCS --senario ID --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```

```

###  $\text{GCA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset WiKiCS --senario ID --gpu_id 0 --seed 39788 --theroy_view
```

结果

```

```

###  $\text{IFL-GC}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset WiKiCS --senario ID --gpu_id 0 --seed 39788 --drop_scheme degree --theroy_view --start_debias_epoch 100 --update_interval 20 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```

```

###  $\text{DGI}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model DGI --dataset WiKiCS --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```

```

###  $\text{COSTA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model COSTA --dataset WiKiCS --senario ID --gpu_id 0 --seed 39788
```

结果

```

```

###  $\text{BGRL}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model BGRL --dataset WiKiCS --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```

```

###  $\text{MVGRL}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model MVGRL --dataset WiKiCS --senario ID --gpu_id 0 --seed 39788 --learning_rate 0.001 --gradient_drop_threshold 1e-5 --tolerance_epoch_num 20 --mvgrl_alpha 0.2
```

结果

```

```

###  $\text{GBT}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GBT --dataset WiKiCS --senario ID --gpu_id 0 --seed 39788 --weight_decay 0 --num_hidden 512 --batch_size 4 --p_x 0.1 --p_e 0.4 --gradient_drop_threshold 1e-3
```

结果

```

```

##  $\text{Computers}$ 数据集
###   $\text{GRACE}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset Computers --senario ID --gpu_id 0 --seed 39788
```

结果

```

```

###  $\text{IFL-GR}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset Computers --senario ID --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```

```

###  $\text{GCA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset Computers --senario ID --gpu_id 0 --seed 39788 --theroy_view
```

结果

```

```

###  $\text{IFL-GC}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset Computers --senario ID --gpu_id 0 --seed 39788 --drop_scheme degree --theroy_view --start_debias_epoch 100 --update_interval 20 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```

```

###  $\text{DGI}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model DGI --dataset Computers --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```

```

###  $\text{COSTA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model COSTA --dataset Computers --senario ID --gpu_id 0 --seed 39788
```

结果

```

```

###  $\text{BGRL}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model BGRL --dataset Computers --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```

```

###  $\text{MVGRL}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model MVGRL --dataset Computers --senario ID --gpu_id 0 --seed 39788 --learning_rate 0.001 --gradient_drop_threshold 1e-5 --tolerance_epoch_num 20 --mvgrl_alpha 0.2
```

结果

```

```

###  $\text{GBT}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GBT --dataset Computers --senario ID --gpu_id 0 --seed 39788 --weight_decay 0 --num_hidden 512 --batch_size 4 --p_x 0.1 --p_e 0.4 --gradient_drop_threshold 1e-3
```

结果

```

```

##  $\text{Photo}$ 数据集
###   $\text{GRACE}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset Photo --senario ID --gpu_id 0 --seed 39788
```

结果

```

```

###  $\text{IFL-GR}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset Photo --senario ID --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```

```

###  $\text{GCA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset Photo --senario ID --gpu_id 0 --seed 39788 --theroy_view
```

结果

```

```

###  $\text{IFL-GC}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset Photo --senario ID --gpu_id 0 --seed 39788 --drop_scheme degree --theroy_view --start_debias_epoch 100 --update_interval 20 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```

```

###  $\text{DGI}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model DGI --dataset Photo --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```

```

###  $\text{COSTA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model COSTA --dataset Photo --senario ID --gpu_id 0 --seed 39788
```

结果

```

```

###  $\text{BGRL}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model BGRL --dataset Photo --senario ID --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```

```

###  $\text{MVGRL}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model MVGRL --dataset Photo --senario ID --gpu_id 0 --seed 39788 --learning_rate 0.001 --gradient_drop_threshold 1e-5 --tolerance_epoch_num 20 --mvgrl_alpha 0.2
```

结果

```

```

###  $\text{GBT}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GBT --dataset Photo --senario ID --gpu_id 0 --seed 39788 --weight_decay 0 --num_hidden 512 --batch_size 4 --p_x 0.1 --p_e 0.4 --gradient_drop_threshold 1e-3
```

结果

```

```