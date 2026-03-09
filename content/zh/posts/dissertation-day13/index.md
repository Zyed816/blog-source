+++ 
draft = false
date = 2026-03-08T20:26:00+08:00
title = "[毕业设计]Day13"
description = ""
slug = "dissertation-day13"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

## 在 $\text{AutoDL}$  上配置实验环境 

```bash
# 1. 初始化 Conda（仅需一次）
conda init bash && source ~/.bashrc

# 2. 创建并激活环境
conda create -n gcl python=3.8 -y
conda activate gcl

# 3. 开启学术加速 (关键：否则 PyTorch 官方源极慢)
source /etc/network_turbo

# 安装 PyTorch 核心
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 安装 GNN 四大底层依赖（最稳定方式：直接对准预编译 Wheel 库）
# 增加 --default-timeout 以应对网络抖动
pip install --default-timeout=1000 \
    torch-scatter==2.1.0+pt113cu117 \
    torch-sparse==0.6.15+pt113cu117 \
    torch-cluster==1.6.0+pt113cu117 \
    torch-spline-conv==1.2.1+pt113cu117 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
 
# 安装 PyG 本体
pip install torch-geometric==2.6.1

# 1. 先关闭学术加速
unset http_proxy && unset https_proxy

# 2. 批量安装常用依赖（使用清华或阿里源）
pip install scikit-learn pandas matplotlib seaborn tqdm ogb optuna PyYAML networkx aiohttp SQLAlchemy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 重新进行 $\text{PubMed}$ 数据集上实验

###  $\text{GRACE}$

命令

```bash
python main.py --model GRACE --dataset PubMed --senario ID --gpu_id 0
```

结果

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
best val acc: 84.8274% ± 0.414787%
correspoding test acc: 84.3020% ± 0.321427%
```

###  $\text{IFL-GR}$

###  $\text{GCA}$

命令

```bash
python main.py --model GCA --dataset PubMed --senario ID --gpu_id 0 --batch_size 4096
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
best val acc: 85.0809% ± 0.297411%
correspoding test acc: 84.9038% ± 0.202917%
```

###  $\text{IFL-GC}$

命令

```bash
python main.py --model GCA --dataset PubMed --senario ID --gpu_id 0 --batch_size 2048 --theroy_view
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
start_debias_epoch: 200
update_interval: 1
R_W_threshold: 0.6
B_W_threshold: 0.6
best val acc: 85.0640% ± 0.020705%
correspoding test acc: 84.8058% ± 0.228762%
```


