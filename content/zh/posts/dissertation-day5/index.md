+++ 
draft = false
date = 2026-02-05T21:28:00+08:00
title = "[毕业设计]Day5"
description = ""
slug = "dissertation-day5"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

## 理论优势分析

### 更合理的假设

在将 $GCL$ 重定义为一个正-未标记学习 $(\text{P-U learning})$ 问题时，首先要回答：为什么从未标记样本中挖掘出正样本是可能的？

传统的 $\text{(P-U learning)}$ 方法依赖于 $SCAR(\text{Select Completely At Random})$ 假设，认为：已知正样本 $D_L^+$ 是从所有正样本中**完全随机**抽取的，因此，已标记正样本 $D_L^+$ 于潜在正样本 $D_U^+$ **独立同分布**。

然而，这种假设在 $GCL$ 中完全无法成立，因为已标记正样本 $D_L^+$ 是人为增强产生的，而 $D_U^+$ 是原图中天然存在的，语义相似的结点，他们的分布显然不同。

相比于 $SCAR$ 假设， $IOD$ 假设明显更加合理：如果两个结点语义相似的可能性（是正样本的可能性）越高，那么计算这两个结点的相似度越高，反过来，如果计算得到这两个结点的相似度越高，那么这两个结点语义相似的可能性（是正样本的可能性）越高

### 更严格的损失函数

很多工作在把新样本加入 $Loss$ 时，是通过加法添加的，类似
$$
L_{other} = -\log(P_{\text{old}} + \alpha P_{\text{new}})
$$
训练过程中，只要 $P_{\text{old}}$ 足够大， $P_{\text{old}} + \alpha P_{\text{new}}$ 就很大， $L_{other}$ 很小，导致模型“偷懒”，不去优化 $P_{new}$

我们使用**乘积**形式将新样本加入到 $Loss$ 
$$
L_{\text{our}} = -\log(P_{\text{old}} \cdot P_{\text{new}}^{\text{weight}})
$$
只要 $P_{new}$ 很小，乘积就很小， $L_{our}$ 就很大，促使模型优化 $P_{new}$ ，也就是说，我们的损失对于 $D_U^+$ 的似然概率更敏感，对 $D_U^+$ 的要求更严格，从而获得效果更好的偏差矫正
