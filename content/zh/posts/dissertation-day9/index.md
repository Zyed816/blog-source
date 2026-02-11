+++ 
draft = false
date = 2026-02-09T22:25:00+08:00
title = "[毕业设计]Day9"
description = ""
slug = "dissertation-day9"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

##  $\text{GOODTwitch}$ 数据集

###  $\text{GRACE}$

尝试运行：

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset GOODTwitch --senario ID --gpu_id 0 --seed 39788
```

遇到报错：

```html
Traceback (most recent call last):
  File "main.py", line 879, in <module>
    _dataset = get_dataset(args)
  File "main.py", line 821, in get_dataset
    from datasets.GOOD_twitch import GOODTwitch
ModuleNotFoundError: No module named 'datasets.GOOD_twitch'
```

解决：

1.在 $\text{GCL}$ 环境下运行命令 ``pip install git+https://github.com/divelab/GOOD.git``

![pic1](pic1.png)

报错：

![pic2](pic2.png)

运行 `pip install --no-deps git+https://github.com/divelab/GOOD.git` ，**跳过依赖安装，只装官方 GOOD 源码本体**

![pic3](pic3.png)

2.在 $\text{datasets/}$ 下新建 `init.py` ，保持内容为空

3.在 $\text{datasets/}$ 下新建 `GOOD_twitch.py` ，内容为

```python
// 这版 GOOD_twitch.py 存在缺陷，因篇幅原因省略
```

尝试运行命令：

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset GOODTwitch --senario OOD --gpu_id 0 --batch_size 4096 --num_epochs 500
```

报错：

```html
Traceback (most recent call last):               
  File "main.py", line 879, in <module>
    _dataset = get_dataset(args)
  File "main.py", line 822, in get_dataset
    _dataset = GOODTwitch(root='datasets', domain='language', generate=True, shift='covariate')
  File "D:\dissertation\openSourceCode\IFL-GCL-main\datasets\GOOD_twitch.py", line 162, in __init__
    loaded = _load_official_good_dataset("GOODTwitch", kwargs)
  File "D:\dissertation\openSourceCode\IFL-GCL-main\datasets\GOOD_twitch.py", line 83, in _load_official_good_dataset
    raise ImportError(msg)
ImportError: Cannot import/use official GOOD library to load dataset.
Tried dataset name: GOODTwitch
Tried entrypoints:
  - good.data.load_dataset: ModuleNotFoundError("No module named 'good'")
  - good.data.dataset_manager.load_dataset: ModuleNotFoundError("No module named 'good'")
  - good.data.good_datasets.GOODTwitch: ModuleNotFoundError("No module named 'good'")
  - good.data.GOODTwitch: ModuleNotFoundError("No module named 'good'")

Fix:
  1) Install GOOD: pip install git+https://github.com/divelab/GOOD.git
  2) Verify you can `import good` in the same env.
```

发现环境里其实已经安装了官方 $\text{GOOD}$ 库，只是顶层包名不是 $\text{good}$ ，而是 $\text{GOOD}$

在 $\text{GCL}$ 环境下运行 `import GOOD`

![pic4](pic4.png)

发现缺少 `munch` 包，安装完成后又提示还有很多包没有安装

![pic5](pic5.png)

在终端运行

```bash
 D:/SoftWare/anaconda/envs/GCL/python.exe -m pip install -U --force-reinstall "munch==2.5.0"
```

接着运行

```bash
D:/SoftWare/anaconda/envs/GCL/python.exe -m pip install -U "ruamel.yaml==0.17.21" "typed-argument-parser==1.7.2" "cilog>=1.2.3" "gdown>=4.4.0" "tensorboard==2.8.0" "protobuf==3.20.1"
```

验证发现还有问题

运行

```bash
D:/SoftWare/anaconda/envs/GCL/python.exe -m pip install -U gdown "munch==2.5.0" tqdm
```

替换 `GOOD_twitch.py`中内容为

```python
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
import shutil
import time
import torch
import warnings

def _find_official_good_twitch_py() -> Path:
    for p in sys.path:
        if not p:
            continue
        base = Path(p)
        candidate = base / "GOOD" / "data" / "good_datasets" / "good_twitch.py"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot find official GOODTwitch implementation file: "
        "GOOD/data/good_datasets/good_twitch.py in current python env."
    )


def _load_official_goodtwitch_class():
    # If user has fully working GOOD import (with rdkit etc), use it directly.
    try:
        from GOOD.data.good_datasets.good_twitch import GOODTwitch as OfficialGOODTwitch  # type: ignore
        return OfficialGOODTwitch
    except Exception:
        pass

    # Fallback: load ONLY the official good_twitch.py without importing GOOD package.
    official_path = _find_official_good_twitch_py()

    # Remove any half-imported GOOD module to avoid weird states.
    sys.modules.pop("GOOD", None)

    # Provide a minimal fake GOOD.register for the decorator used in good_twitch.py
    fake_good = types.ModuleType("GOOD")

    class _FakeRegister:
        def dataset_register(self, dataset_class):
            return dataset_class

    fake_good.register = _FakeRegister()
    sys.modules["GOOD"] = fake_good

    spec = importlib.util.spec_from_file_location("_official_good_twitch", str(official_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {official_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "GOODTwitch"):
        raise ImportError(f"{official_path} does not define GOODTwitch")

    return module.GOODTwitch


_OfficialGOODTwitch = _load_official_goodtwitch_class()

def _goodtwitch_expected_processed_files(root: Path, domain: str):
    processed_dir = root / "GOODTwitch" / domain / "processed"
    names = ["no_shift.pt", "covariate.pt", "concept.pt"]
    return [processed_dir / n for n in names]


def _goodtwitch_processed_ready(root: Path, domain: str) -> bool:
    return all(p.is_file() for p in _goodtwitch_expected_processed_files(root, domain))


class GOODTwitch(_OfficialGOODTwitch):
    """
    Official GOODTwitch + mask adapter for this repo.

    Official GOODTwitch provides:
      - train_mask (ID train)
      - id_val_mask / id_test_mask (ID val/test)
      - val_mask / test_mask (OOD val/test)

    This repo expects OOD fields:
      - PT_mask / SFT_mask / SFT_test_mask / Valid_mask / Test_mask
    """

    def __init__(self, root="datasets", domain="language", generate=False, shift="covariate", **kwargs):
        if generate:
            warnings.warn(
                "GOODTwitch(generate=True) 会触发 PyG Twitch 从 graphmining.ai 下载原始数据（当前常见 404）。"
                "已自动回退到 generate=False，改用官方 GOOD 预处理 zip（gdown）下载路径。",
                RuntimeWarning,
            )

        root_path = Path(root)

        # 如果预处理文件不存在，但 datasets/GOODTwitch 目录已经存在（通常是上次失败留下的残缺目录），
        # 官方 GOODTwitch._download() 会因为“目录存在”而直接跳过 gdown 下载，导致又去 process() -> PyG Twitch 404。
        if not _goodtwitch_processed_ready(root_path, domain):
            incomplete_dir = root_path / "GOODTwitch"
            if incomplete_dir.exists():
                backup_dir = root_path / f"GOODTwitch_incomplete_{int(time.time())}"
                warnings.warn(
                    f"检测到 {incomplete_dir} 已存在但预处理文件不齐全，将其重命名为 {backup_dir} "
                    f"以强制官方 GOODTwitch 走 gdown 下载预处理数据。",
                    RuntimeWarning,
                )
                try:
                    incomplete_dir.rename(backup_dir)
                except Exception:
                    try:
                        shutil.rmtree(incomplete_dir)
                    except Exception as e:
                        raise RuntimeError(
                            f"无法移动/删除残缺目录 {incomplete_dir}，请手动删除该目录后重试。原始异常: {e}"
                        )

        super().__init__(root=root, domain=domain, shift=shift, generate=False, **kwargs)

        data = self._data
        data.PT_mask = data.train_mask
        data.SFT_mask = data.id_val_mask
        data.SFT_test_mask = data.id_test_mask
        data.Valid_mask = data.val_mask
        data.Test_mask = data.test_mask

        if not hasattr(data, "domain_id"):
            if hasattr(data, "env_id"):
                data.domain_id = data.env_id
            else:
                data.domain_id = torch.zeros(data.num_nodes, dtype=torch.long)
```

接着运行命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset GOODTwitch --senario OOD --gpu_id 0 --batch_size 4096 --num_epochs 500 --seed 39788
```

#### 在 $\text{AutoDL}$ 上运行

1.执行命令

```bash
source /etc/network_turbo

pip install --no-deps git+https://github.com/divelab/GOOD.git
python -m pip install -U --force-reinstall "munch==2.5.0"
python -m pip install -U "ruamel.yaml==0.17.21" "typed-argument-parser==1.7.2" "cilog>=1.2.3" "gdown>=4.4.0" "tensorboard==2.8.0" "protobuf==3.20.1"

python -m pip install -U gdown "munch==2.5.0" tqdm
```

2.新建 `datasets/__init__.py`

3.新建 `datasets/GOOD_twitch.py` ，内容与本地 `GOOD_twitch.py` 相同

命令

```bash
python main.py --model GRACE --dataset GOODTwitch --senario OOD --gpu_id 0 --batch_size 4096 --num_epochs 500 --seed 39788
```

结果

```html
Seed: 39788
Dataset: GOODTwitch
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
best OOD-val acc: 53.2540% ± 5.353651%
corresponding OOD-test acc: 59.0069% ± 6.180440%
corresponding val acc: 65.2092% ± 1.076897%
corresponding test acc: 65.1017% ± 0.710453%
```

##  $\text{IFL-GR}$

命令

```bash
python main.py --model GRACE --dataset GOODTwitch --senario OOD --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```html
Seed: 39788
Dataset: GOODTwitch
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
best OOD-val acc: 53.9002% ± 5.120682%
corresponding OOD-test acc: 62.4001% ± 4.116907%
corresponding val acc: 64.4256% ± 0.624112%
corresponding test acc: 64.6551% ± 0.359390%
```

###  $\text{GCA}$

命令

```bash
python main.py --model GCA --dataset GOODTwitch --senario OOD --gpu_id 0 --batch_size 4096 --num_epochs 500 --seed 39788
```

结果

```html
Seed: 39788
Dataset: GOODTwitch
Model: GCA
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
drop_scheme: degree
best OOD-val acc: 53.5033% ± 3.159712%
corresponding OOD-test acc: 61.3731% ± 3.033851%
corresponding val acc: 63.9555% ± 0.720549%
corresponding test acc: 64.5434% ± 0.219288%
```

###  $\text{IFL-GC}$

命令

```bash
python main.py --model GCA --dataset GOODTwitch --senario OOD --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1 
```

结果

```html
Seed: 39788
Dataset: GOODTwitch
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
best OOD-val acc: 53.5084% ± 3.159794%
corresponding OOD-test acc: 61.3678% ± 3.034544%
corresponding val acc: 63.9398% ± 0.698395%
corresponding test acc: 64.5493% ± 0.220231%
```

###  $\text{DGI}$

命令

```bash
python main.py --model DGI --dataset GOODTwitch --senario OOD --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```html
Seed: 39788
Dataset: GOODTwitch
Model: DGI
Num_epochs: 500
Repeat_times: 3
best OOD-val acc: 61.0746% ± 1.277574%
corresponding OOD-test acc: 56.0002% ± 4.513072%
corresponding val acc: 64.0339% ± 1.296671%
corresponding test acc: 64.2320% ± 0.025388%
```

###  $\text{COSTA}$

命令

```bash
python main.py --model COSTA --dataset GOODTwitch --senario OOD --gpu_id 0 --seed 39788
```

结果

```html
Seed: 39788
Dataset: GOODTwitch
Model: COSTA
Num_epochs: 500
Repeat_times: 3
best OOD-val acc: 55.2028% ± 2.928977%
corresponding OOD-test acc: 59.6475% ± 3.052626%
corresponding val acc: 64.4413% ± 0.356678%
corresponding test acc: 64.9293% ± 0.472346%
```

###  $\text{BGRL}$

命令

```bash
python main.py --model BGRL --dataset GOODTwitch --senario OOD --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```html
Seed: 39788
Dataset: GOODTwitch
Model: BGRL
Num_epochs: 500
Repeat_times: 3
best OOD-val acc: 58.2506% ± 1.870426%
corresponding OOD-test acc: 61.1296% ± 3.090921%
corresponding val acc: 63.9868% ± 0.499034%
corresponding test acc: 64.8842% ± 0.148965%
```

###  $\text{MVGRL}$

命令

```bash
python main.py --model MVGRL --dataset GOODTwitch --senario OOD --gpu_id 0 --seed 39788 --learning_rate 0.001 --gradient_drop_threshold 1e-5 --tolerance_epoch_num 20 --mvgrl_alpha 0.2
```

结果：卡在预处理阶段

```html

```

###  $\text{GBT}$

命令

```bash
python main.py --model GBT --dataset GOODTwitch --senario OOD --gpu_id 0 --seed 39788 --weight_decay 0 --num_hidden 512 --batch_size 4 --p_x 0.1 --p_e 0.4 --gradient_drop_threshold 1e-3
```

结果

```html
Seed: 39788
Dataset: GOODTwitch
Model: GBT
Num_epochs: 500
Repeat_times: 3
best OOD-val acc: 59.4566% ± 2.212121%
corresponding OOD-test acc: 58.5623% ± 0.883274%
corresponding val acc: 62.6704% ± 0.580904%
corresponding test acc: 63.3192% ± 0.206772%
```

##  $\text{GOODCora}$ 数据集

在 `datasets/` 下新建 `GOOD_cora.py` ，内容参考本地

###  $\text{GRACE}$

命令

```bash
python main.py --model GRACE --dataset GOODCora --domain_flag degree --senario OOD --gpu_id 0 --batch_size 4096 --num_epochs 500
```

结果

```html
Seed: 39788
Dataset: GOODCora
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
best OOD-val acc: 56.9875% ± 0.847826%
corresponding OOD-test acc: 49.6526% ± 1.098903%
corresponding val acc: 64.2179% ± 0.936948%
corresponding test acc: 64.4828% ± 0.934301%
```
###  $\text{IFL-GR}$

命令

```bash
python -W ignore main.py --model GRACE --dataset GOODCora --domain_flag degree --senario OOD --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```html
Seed: 39788
Dataset: GOODCora
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
best OOD-val acc: 54.5677% ± 0.934908%
corresponding OOD-test acc: 46.8056% ± 0.990121%
corresponding val acc: 63.7681% ± 1.122493%
corresponding test acc: 63.6986% ± 0.424240%
```

###  $\text{GCA}$

命令

```bash
python -W ignore main.py --model GCA --dataset GOODCora --domain_flag degree --senario OOD --gpu_id 0 --batch_size 4096 --num_epochs 500
```

结果

```html
Seed: 39788
Dataset: GOODCora
Model: GCA
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
drop_scheme: degree
best OOD-val acc: 53.7463% ± 0.585045%
corresponding OOD-test acc: 45.2519% ± 0.000000%
corresponding val acc: 61.8941% ± 0.154033%
corresponding test acc: 63.4330% ± 0.099880%
```

###  $\text{IFL-GC}$

命令

```bash
python -W ignore main.py --model GCA --dataset GOODCora --domain_flag degree --senario OOD --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```html
Seed: 39788
Dataset: GOODCora
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
best OOD-val acc: 52.8916% ± 0.408142%
corresponding OOD-test acc: 43.3893% ± 1.055147%
corresponding val acc: 63.0185% ± 1.139058%
corresponding test acc: 62.3520% ± 0.380526%
```

###  $\text{DGI}$

命令

```bash
python -W ignore main.py --model DGI --dataset GOODCora --domain_flag degree --senario OOD --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```html
Seed: 39788
Dataset: GOODCora
Model: DGI
Num_epochs: 500
Repeat_times: 3
best OOD-val acc: 54.4678% ± 0.534417%
corresponding OOD-test acc: 44.7983% ± 0.708123%
corresponding val acc: 63.8681% ± 1.749482%
corresponding test acc: 63.6486% ± 0.303268%
```

###  $\text{COSTA}$

命令

```bash
python -W ignore main.py --model COSTA --dataset GOODCora --domain_flag degree --senario OOD --gpu_id 0 --seed 39788
```

结果

```html
Seed: 39788
Dataset: GOODCora
Model: COSTA
Num_epochs: 500
Repeat_times: 3
best OOD-val acc: 54.4789% ± 0.881456%
corresponding OOD-test acc: 46.8732% ± 0.851010%
corresponding val acc: 62.6187% ± 1.026619%
corresponding test acc: 63.1706% ± 0.210747%
```

###  $\text{BGRL}$

命令

```bash
python -W ignore main.py --model BGRL --dataset GOODCora --domain_flag degree --senario OOD --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```html
Seed: 39788
Dataset: GOODCora
Model: BGRL
Num_epochs: 500
Repeat_times: 3
best OOD-val acc: 45.1992% ± 0.816738%
corresponding OOD-test acc: 32.8894% ± 0.860154%
corresponding val acc: 58.0960% ± 0.942264%
corresponding test acc: 58.6684% ± 0.237407%
```

###  $\text{MVGRL}$

命令

```bash
python -W ignore main.py --model MVGRL --dataset GOODCora --domain_flag degree --senario OOD --gpu_id 0 --seed 39788 --learning_rate 0.001 --gradient_drop_threshold 1e-5 --tolerance_epoch_num 20 --mvgrl_alpha 0.2
```

结果

```html
Seed: 39788
Dataset: GOODCora
Model: MVGRL
Num_epochs: 500
Repeat_times: 3
best OOD-val acc: 42.4242% ± 1.263940%
corresponding OOD-test acc: 29.3476% ± 1.116893%
corresponding val acc: 55.7471% ± 0.231725%
corresponding test acc: 56.8376% ± 0.199419%
```

###  $\text{GBT}$

命令

```bash
python -W ignore main.py --model GBT --dataset GOODCora --domain_flag degree --senario OOD --gpu_id 0 --seed 39788 --weight_decay 0 --num_hidden 512 --batch_size 4 --p_x 0.1 --p_e 0.4 --gradient_drop_threshold 1e-3
```

结果

```html
Seed: 39788
Dataset: GOODCora
Model: GBT
Num_epochs: 500
Repeat_times: 3
best OOD-val acc: 55.6333% ± 0.258418%
corresponding OOD-test acc: 47.9251% ± 0.817632%
corresponding val acc: 62.7186% ± 0.821933%
corresponding test acc: 63.9048% ± 0.371229%
```



##  $\text{GOODCBAS}$ 数据集

在 `datasets/` 下新建 `GOOD_cbas.py` ，内容参考本地

###  $\text{GRACE}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset GOODCBAS --senario OOD --gpu_id 0 --batch_size 4096 --num_epochs 500 --seed 39788           
```

结果

```html
Seed: 39788
Dataset: GOODCBAS
Model: GRACE
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
best OOD-val acc: 55.2381% ± 7.939682%
corresponding OOD-test acc: 49.0476% ± 5.259696%
corresponding val acc: 55.3571% ± 11.007882%
corresponding test acc: 55.5060% ± 3.763145%
```

###  $\text{IFL-GR}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GRACE --dataset GOODCBAS --senario OOD --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```html
Dataset: GOODCBAS
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
best OOD-val acc: 55.2381% ± 7.939682%
corresponding OOD-test acc: 49.0476% ± 5.259696%
corresponding val acc: 55.3571% ± 11.007882%
corresponding test acc: 55.5060% ± 3.763145%
```

###  $\text{GCA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset GOODCBAS --senario OOD --gpu_id 0 --batch_size 4096 --num_epochs 500
```

结果

```html
Seed: 39788
Dataset: GOODCBAS
Model: GCA
Num_epochs: 500
Repeat_times: 3
tau: 0.4
drop_edge_rate_1: 0.2
drop_edge_rate_2: 0.4
drop_feature_rate_1: 0.3
drop_feature_rate_2: 0.4
drop_scheme: degree
best OOD-val acc: 55.2381% ± 3.749528%
corresponding OOD-test acc: 56.1905% ± 4.714045%
corresponding val acc: 56.5476% ± 3.035131%
corresponding test acc: 55.8036% ± 2.933096%
```

###  $\text{IFL-GC}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GCA --dataset GOODCBAS --senario OOD --gpu_id 0 --seed 39788 --theroy_view --start_debias_epoch 200 --update_interval 1 --B_W_threshold 0.6 --R_W_threshold 0.6 --norm_sim_matrix global --stay_diag_eye 1
```

结果

```html
Seed: 39788
Dataset: GOODCBAS
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
best OOD-val acc: 55.2381% ± 3.749528%
corresponding OOD-test acc: 56.1905% ± 4.714045%
corresponding val acc: 56.5476% ± 3.035131%
corresponding test acc: 55.8036% ± 2.933096%
```

###  $\text{DGI}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model DGI --dataset GOODCBAS --senario OOD --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```html
Seed: 39788
Dataset: GOODCBAS
Model: DGI
Num_epochs: 500
Repeat_times: 3
best OOD-val acc: 55.7143% ± 1.166424%
corresponding OOD-test acc: 59.0476% ± 2.693740%
corresponding val acc: 50.5952% ± 3.035131%
corresponding test acc: 57.0685% ± 2.007550%
```

###  $\text{COSTA}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model COSTA --dataset GOODCBAS --senario OOD --gpu_id 0 --seed 39788
```

结果

```html
Seed: 39788
Dataset: GOODCBAS
Model: COSTA
Num_epochs: 500
Repeat_times: 3
best OOD-val acc: 56.6667% ± 7.766432%
corresponding OOD-test acc: 39.0476% ± 4.416009%
corresponding val acc: 58.3333% ± 5.520011%
corresponding test acc: 56.3988% ± 3.210632%
```

###  $\text{BGRL}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model BGRL --dataset GOODCBAS --senario OOD --gpu_id 0 --seed 39788 --gradient_drop_threshold 1e-4
```

结果

```html
Seed: 39788
Dataset: GOODCBAS
Model: BGRL
Num_epochs: 500
Repeat_times: 3
best OOD-val acc: 52.8571% ± 2.332847%
corresponding OOD-test acc: 56.1905% ± 4.856209%
corresponding val acc: 52.3810% ± 2.227177%
corresponding test acc: 54.0923% ± 2.837154%
```

###  $\text{MVGRL}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model MVGRL --dataset GOODCBAS --senario OOD --gpu_id 0 --seed 39788 --learning_rate 0.001 --gradient_drop_threshold 1e-5 --tolerance_epoch_num 20 --mvgrl_alpha 0.2
```

结果

```html
Seed: 39788
Dataset: GOODCBAS
Model: MVGRL
Num_epochs: 500
Repeat_times: 3
best OOD-val acc: 55.7143% ± 3.499271%
corresponding OOD-test acc: 51.4286% ± 5.084323%
corresponding val acc: 50.5952% ± 3.669294%
corresponding test acc: 57.8869% ± 4.236500%
```

###  $\text{GBT}$

命令

```bash
& D:/SoftWare/anaconda/envs/GCL/python.exe main.py --model GBT --dataset GOODCBAS --senario OOD --gpu_id 0 --seed 39788 --weight_decay 0 --num_hidden 512 --batch_size 4 --p_x 0.1 --p_e 0.4 --gradient_drop_threshold 1e-3
```

结果

```html
Seed: 39788
Dataset: GOODCBAS
Model: GBT
Num_epochs: 500
Repeat_times: 3
best OOD-val acc: 54.2857% ± 1.166424%
corresponding OOD-test acc: 52.8571% ± 3.499271%
corresponding val acc: 50.0000% ± 3.857584%
corresponding test acc: 56.1012% ± 1.213505%
```


