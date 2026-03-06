+++ 
draft = false
date = 2026-02-06T21:45:00+08:00
title = "[毕业设计]Day6"
description = ""
slug = "dissertation-day6"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

## 实验设置

### 数据集

使用 $9$ 个数据集，分**独立同分布** $\text{(independent and identically distributed,IID)}$ 和**分布外** $\text{(out-of-distribution,OOD)}$ 两种情景

-  $\text{IID}$ ： $\text{Cora,PubMed,CiteSeer,WikiCS,Computers,Photo}$ ，**特点**：训练集和测试集分布相同
-  $\text{OOD}$： $\text{GOODTwitch,GOODCora,GOODCBAS}$ ，**特点**：测试集的数据分布和训练集不一样（比如图的结构变了，或者特征分布变了）

在 $\text{OOD}$ 情境下测试的原因是：传统方法 $\text{(GRACE/GCA)}$ 太依赖数据增强 $\text{(Augmentation)}$ ，很容易过拟合到某种特定的增强模式上。我们希望证明，**语义引导** $\text{(Semantic Guidance)}$ 能让模型学到更本质的知识，从而在环境变化时也能表现良好

### 基线模型

共与 $7$ 种基线方法进行对比，最主要的两个基线方法是： $\text{GRACE}$ 和 $\text{GCA}$ ，因为它们都使用了 $InfoNCE$ 损失函数。我们对这两种方法进行优化。具体来说，使用它们进行 $\text{warm-up}$ 阶段，然后按照 $\text{IFL-GCL}$ 理论进行重采样 $\text{(resampling)}$ ，使用新的损失函数进一步训练编码器。按照这种思路得到的训练方法分别称为 $\text{IFL-GR}$ 以及 $\text{IFL-GC}$ 

### 评价指标和流程

所有图对比学习 $\text{(GCL)}$ 方法最终都得到一个编码器 $f_\theta$，它能把语义相近的结点映射到相近的向量，如何衡量一个编码器的优劣呢？我们使用有标签的数据集，将原图 $G$ 中每个结点 $n_i$ 经过编码器 $f_\theta$ 编码后得到的编码向量 $H_i$ 作为输入，将原图 $G$ 按 $\text{1:1:8}$ 划分为训练集，验证集和测试集，训练一个分类器 $\text{classifier}$ ，以分类器在测试集上的准确率作为衡量编码器 $f_\theta$ 的指标

如果编码器 $f_\theta$ 能很好地捕捉原图中各个结点的语义，它会将语义相近（实际标签相同）的结点编码为距离接近 的向量，此时训练一个简单分类器实现结点群向实际标签的映射就比较简单，意味着在测试集上的准确率高

如果编码器 $f_\theta$ 无法准确捕捉语义，它会将不同语义的结点拉近，相同语义结点可能被推远，这就导致无法得到一个良好的分类器实现结点向实际标签的准确映射

注意，分类器 $\text{classifier}$ 必须是简单的**线性分类器**，避免复杂分类器对标签的拟合掩盖了对编码器编码效果的区分

因此该对比方案是合理的
