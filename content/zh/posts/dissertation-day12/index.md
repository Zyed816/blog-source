+++ 
draft = false
date = 2026-03-06T20:26:00+08:00
title = "[毕业设计]Day12"
description = ""
slug = "dissertation-day12"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

## 修改 `gca_functional.py`

从 https://github.com/CRIPAC-DIG/GCA/blob/main/pGRACE 下载 `functional.py` 和 `utils.py` 并重命名为 `gca_functional.py` 和 `gca_utils.py`，重新进行实验

## 使用 ${\text{LLM}}$ 进行特征增强

### 在本地生成 `geometric_data_processed.pt`

新建文件 `make_geometric_cora.py` ，内容为

```python
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import torch
import os

# 1. 读取 PyG 自带的 Cora 数据
root = r'D:\dissertation\openSourceCode\IFL-GCL-origin\datasets'  # 如果你 Planetoid 的数据根目录不同，在这里改
dataset = Planetoid(root=root, name='Cora')
data: Data = dataset[0]

# 2. 保存成 geometric_data_processed.pt，形式是 [data][0] 能取到
save_dir = r'D:\dissertation\openSourceCode\IFL-GCL-origin\datasets\minilmdata\cora\processed'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'geometric_data_processed.pt')

torch.save([data], save_path)
print("saved to:", save_path)
```

运行得到

```html
D:\dissertation\datasets\minilmdata\cora\processed\geometric_data_processed.pt
```

### 增加路径参数

 `utils.py` 设置 ${\text{LLM}}$ 相关参数附近增加路径参数 `--minilm_root` ，用于指明本地目录

```python
	parser.add_argument('--llm_emb', action="store_true", help="Use llm to enhance the TAG feature")
    parser.add_argument('--llm_name', type=str, choices=['Llama-3.2-1B', 'Llama-3.2-3B', 'Qwen2.5-0.5B', 'Qwen2.5-1.5B', 'Qwen2.5-3B', 'Qwen2.5-7B'], default='Llama-3.2-1B')
    parser.add_argument('--llm_emb_dim', type=int, default=1024)
    # 新增：minilm 预处理数据的根路径
    parser.add_argument('--minilm_root', type=str,
                        default=r'D:\dissertation\openSourceCode\IFL-GCL-origin\datasets\minilmdata',
                        help='root directory of *minilmdata*/{dataset}/processed/geometric_data_processed.pt')
```

### 修改硬编码路径

 `main.py` 中修改为

```python
# data = torch.load(f'/data1/lvnuoyan/minilmdata/{args.dataset.lower()}/processed/geometric_data_processed.pt')[0]

data_path = os.path.join(
                args.minilm_root,
                args.dataset.lower(),
                'processed',
                'geometric_data_processed.pt',
            )
            data = torch.load(data_path)[0]
```

### 增加 $\text{LLM}$ 节点嵌入路径参数

`utils.py` 设置 ${\text{LLM}}$ 相关参数附近增加路径参数 `--llm_emb_root'`

```python
# 新增：llm 节点嵌入文件所在根目录
    parser.add_argument('--llm_emb_root', type=str,
                        default=r'D:\dissertation\openSourceCode\IFL-GCL-origin\datasets\TAG_LLM_emb',
                        help='root dir of {dataset}_{llm_name}_emb.pt')
```

### 修改 `main.py:get_dataset` 中加载 `llm_emb` 的部分

```python
    # llm-emb as feature (x)
                # llm_emb = torch.load(f'/home/wangzixu/DisShiftGRACE/datasets/TAG_LLM_emb/{args.dataset.lower()}_{args.llm_name}_emb.pt')
    llm_emb_path = os.path.join(
        args.llm_emb_root,
        f'{args.dataset.lower()}_{args.llm_name}_emb.pt'
    )
```

然后运行

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset Cora --senario ID --gpu_id 0 --seed 39788 --llm_emb --llm_name Llama-3.2-1B --theroy_view
```

注意此时的 `cora_Llama-3.2-1B_emb.pt` 并不真正是利用 `Llama-3.2-1B` 生成的节点增强，而是使用 `make_dummy_cora_llm_emb.py` ，用当前的 `data.x` 生成一个伪 `cora_Llama-3.2-1B_emb.pt`，当作“LLM 特征”跑实验，验证 IFL-GCL 代码本身逻辑

因此，接下来问题聚焦于如何获得 `TAG_LLM_emb` 文件夹下的各个 `.pt` 文件