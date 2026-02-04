+++ 
draft = false
date = 2026-02-04T00:00:00+08:00
title = "[毕业设计]Day4"
description = ""
slug = "dissertation-day4"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

> 未完待续。。。

## 更新训练目标

我们已经证明， $s_\theta(n,n')$ 可以搭配上一个超参数 $t_s$ 便捷的实现一个分类器，用于从未标记样本中发现潜在的**可能的**正样本，然后进行重采样 ($\text{resampling}$) 将这些可能的正样本 $D_U^+$ 从未标记样本集 $D_U$ 中取出，加入正样本集 $D^+$ ，进而消除采样偏差
$$
y(x) = h(x=(n,n'); \theta) := \text{sign}(s_\theta(n,n') - t_s)
$$
一个随之而来的问题是，在计算损失函数 $L$ 时，新加入的潜在正样本 $D_U^+$ 与原本的正样本 $D_L^+$ 是否应该具有相同的权重呢？直觉上来看是否定的，因为 $D_L^+$ 是我们自己增强生成的， $\text{100\%}$ 是正样本，而新加入的知识**潜在**正样本，因此我们应对训练目标进行更新 ($\text{updating training objective}$) ，具体来说就是修改损失函数 $L$

$\text{Info-NCE}$ 中损失函数：
$$
\begin{aligned}
L &= \frac{1}{2N} \sum_{i=1}^N (l_{u_i, v_i} + l_{v_i, u_i}) \\
\text{其中}，l_{u_i, v_i} &= -\log \frac{s_{\theta}(u_i, v_i)}{\sum_{j \neq i, j=1}^N s_{\theta}(u_i, u_j) + \sum_{j=1}^N s_{\theta}(u_i, v_j)} \\ \\
s_{\theta}(u_i, v_j) &= \exp(\text{cos}(\mathbf{U}_i, \mathbf{V}_j) / \tau) 
\end{aligned}
$$
其中 $l_{u_i,v_i}$ 可以被理解为以下的负对数似然概率的形式：
$$
J_{u_i,v_i} = l_{u_i, v_i} = -\log \frac{s_{\theta}(u_i, v_i)}{\sum_{j \neq i, j=1}^N s_{\theta}(u_i, u_j) + \sum_{j=1}^N s_{\theta}(u_i, v_j)} = - \log P_{u_i,v_i}
$$
其中， $P_{u_i,v_i}$ 表示在所有候选节点中，选中 $v_i$ 作为 $u_i$ 的正样本的**归一化概率**，也就是结点对 $(u_i,v_i)$ 为正样本的**近似**概率

> 这个转换是如何得到的？
>
> 分子 $s_{\theta}(u, v)$ 是 $u$ 与 $v$ 的相似度，分母是 $u$ 与除自身外其他所有结点（包括 $v$）的相似度之和，当 $(u,v)$ 是正样本（语义接近）时， $P_{u,v}$ 为 $1$ ，当 $(u,v)$ 语义差异大时， $P_{u,v}$ 趋近于 $0$ ，因此 $P_{u,v}$ 的大小可以看作 $(u,v)$ 是正样本的可能性的近似

进而得到，当 $n$ 作为锚点结点时， $P_{n,n^{\prime}}$ 可以看作结点对 $(n,n^{\prime})$ 是正样本的概率近似

结合之前的重要结论
$$
\begin{aligned}
&\forall \mathbf{x} = (n, n'), \mathbf{\hat{x}} = (\hat{n}, \hat{n}') \in D : \\
&p(y = +1|\mathbf{x}) \leq p(y = +1|\mathbf{\hat{x}}) \Leftrightarrow s_{\theta}(n, n') \leq s_{\theta}(\hat{n}, \hat{n}')
\end{aligned}
$$
可以得到
$$
\begin{aligned}
&\forall n', n'' \in D : \\
&p(y = +1|(n, n')) \leq p(y = +1|(n, n'')) \Leftrightarrow \mathbf{P}_{n,n'} \leq \mathbf{P}_{n,n''}
\end{aligned}
$$
这也启示我们对于不同的结点对 $(u,v),(u^{\prime},v^{\prime})$ ，由于他们实际语义接近的概率不同，在计算损失函数时要分配不同的权重，才能避免模型盲目地将他们拉近

接下来我们完成对传统 $GCL$ 损失函数的调整 ：

传统 $GCL$ 中正样本集
$$
D^+ = D_L^{+} = \{ \underbrace{(u_1, v_1), (v_1, u_1)}_{\text{第1对}}, \underbrace{(u_2, v_2), (v_2, u_2)}_{\text{第2对}}, \dots, \underbrace{(u_N, v_N), (v_N, u_N)}_{\text{第N对}} \}
$$
原始损失函数
$$
\begin{aligned}
L &= \frac{1}{2N} \sum_{i=1}^N (l_{u_i, v_i} + l_{v_i, u_i}) \\
\text{其中}，l_{u_i, v_i} &= -\log \frac{s_{\theta}(u_i, v_i)}{\sum_{j \neq i, j=1}^N s_{\theta}(u_i, u_j) + \sum_{j=1}^N s_{\theta}(u_i, v_j)} = - \log P_{u_i,v_i}
\end{aligned}
$$
做以下推导
$$
L = \frac{1}{2N} \sum_{i=1}^N (l_{u_i, v_i} + l_{v_i, u_i}) = \frac{1}{2N} \sum_{(n,n^{\prime}) \in D_L^+} -\log P_{n,n^{\prime}}
$$
重采样 ($\text{resampling}$) 后的 $\text{GCL}$ 中，正样本集
$$
D^+ = D_L^+ \cup D_U^+
$$
因此，损失函数应该分为两部分：对于来自 $D_L^+$ 的样本 $(n,n^\prime)$ 的损失函数 $L_{D_L^+}$ 以及对于来自 $D_U^+$ 的样本 $(n,n'')$ 的损失函数 $L_{D_U^+}$

首先，$L_{D_L^+}$ 和传统 $GCL$ 中的损失函数并无区别，即
$$
L_{D_L^+} = L = \frac{1}{2|D_L^+|} \sum_{(n,n^{\prime}) \in D_L^+} -\log P_{n,n^{\prime}}
$$

可以发现 $L$ 其实是 $D^+$ 中所有样本 $(n,n^{\prime})$ 的 $l_{n, n^{\prime}}$ 值的平均值。当结点数量大时，根据大数定律， $L$ 就是 $D^+$ 中任意样本 $(n,n^{\prime})$ 的 $l_{n, n^{\prime}}$ 的期望
$$
L = \mathbb{E}_{(n,n') \in D_L^+} [l_{n,n^{\prime}}] = \mathbb{E}_{(n,n') \in D_L^+} [- \log P_{n,n'}]
$$
