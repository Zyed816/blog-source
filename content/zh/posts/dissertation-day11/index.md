+++ 
draft = false
date = 2026-02-11T21:15:00+08:00
title = "[毕业设计]Day11"
description = ""
slug = "dissertation-day11"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

## 使用 $\text{LLM}$ 作为增强器

### 设计思路

 $\text{IFL-GCL}$ 提出语义引导的图对比学习方法，目的是为了从未标记样本 $D_U$ 中动态地找出正样本 $D_U^+$ 并加入到正样本集 $D^+$ ，结点语义越明确，模型计算出的相似度 $s_\theta$ 就应该越精准，挖掘出来的正样本也就越可靠

LLM（如 Llama, Qwen）对文本的理解能力极强。如果我们用 LLM 来处理图节点上的文本，生成的特征就是“高质量语义特征”

因此，如果在直接使用原始数据进行图对比学习之前，先使用 $\text{LLM}$ 对结点的文本特征进行增强，就能更好地激发 $\text{IFL-GCL}$ 的优越性

### 数据集

-  $\text{Cora}$
-  $\text{CiteSeer}$

为什么使用这两个数据集？

因为 $\text{Cora}$ 和 $\text{CiteSeer}$ 是“文本属性图”（Text-Attributed Graphs），每个节点代表一篇论文，都有原始的文本内容（标题、摘要等），可以使用 $\text{LLM}$ 进行特征增强

### 特征增强流程

1. 拿到节点 $v$ 的原始文本 $T_v$。
2. 把 $T_v$ 喂给 $\text{LLM}$。
3. 取 $\text{LLM}$ 最后一层隐藏层的输出向量，作为这个节点的新特征 $X'_v$。
4. **丢弃**原始的词袋特征 $X$，用 $X'$ 代替

### 剩余流程与评价指标

其他流程与之前实验相同，仍然使用不同的图对比学习方法 ($\text{GRACE,GCL,IFL-GR,IFL-GC}$) 训练结构相同的**线性**分类器，以分类器的准确率作为图对比学习方法优劣的指标
