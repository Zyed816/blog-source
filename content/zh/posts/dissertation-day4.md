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

可以发现 $L$ 其实是 $D^+$ 中所有样本 $(n,n^{\prime})$ 的 $l_{n, n^{\prime}}$ 值的平均值。当结点数量大时，根据大数定律， $L$ 就是 $D^+$ 中任意样本 $(n,n^{\prime})$ 的 $l_{n, n^{\prime}}$ 的期望
$$
L = \mathbb{E}_{(n,n') \in D_L^+} [l_{n,n^{\prime}}] = \mathbb{E}_{(n,n') \in D_L^+} [- \log P_{n,n'}]
$$
重采样 ($\text{resampling}$) 后的 $\text{GCL}$ 中，正样本集
$$
D^+ = D_L^+ \cup D_U^+
$$
因此， $D_U^+$ 中结点的也应被添加到损失函数中，得到
$$
L^{\text{corrected}} = \mathbb{E}_{(n,n') \in D_L^+} [-\log P_{n,n'}] + \mathbb{E}_{(n,n'') \in D_U^+} [\text{weight} \times (-\log P_{n,n''})]
$$
$\text{weight}$ 体现了对 $D_U^+$ 中不同结点对有不同的置信度，对于语义相似置信度高的结点， $\text{weight}$ 更大，模型会更努力地将他们拉近，反之则 $\text{weight}$ 更小，模型将他们拉近的力度更小。具体来说， $\text{weight}$ 被设计为
$$
\text{weight} = \beta \cdot \hat{s}_{\theta}(n, n'') \\
\text{其中，} \hat{s}_{\theta}(n, n'') = \frac{s_{\theta}(n, n'') - \min \{ s_{\theta}(n_i, n''_j) \}_{i,j=1}^N}{\max \{ s_{\theta}(n_i, n''_j) \} - \min \{ s_{\theta}(n_i, n''_j) \}_{i,j=1}^N}
$$
 $\hat{s}_{\theta}(n, n'') \in [0,1]$ 是对来自 $D_U^+$ 中样本的归一化打分，如果样本 $(n,n'')$ 的相似度在所有 $D_U^+$ 样本中是最高的，那么它将得到的初步权重为 $1$ ，相似度最低的样本得到的初步权重为 $0$ ，接着再用一个超参数 $\beta$ 表示对来自 $D_U^+$ 的样本的整体置信度，由于他们只是潜在的正样本，因此 $\beta$ 一般小于 $1$

至此，我们得到修改后的损失函数
$$
\begin{aligned}
L^{\text{corrected}} &= \mathbb{E}_{(n,n') \in D_L^+} [-\log P_{n,n'}] + \mathbb{E}_{(n,n'') \in D_U^+} [\text{weight} \times (-\log P_{n,n''})] \\
&= \mathbb{E}_{\substack{(n,n') \in D_L^+ \\ (n,n'') \in D_U^+}} \left[ - \log \left( P_{n,n'} \prod (P_{n,n''})^{\text{weight}} \right) \right] \\
&= \mathbb{E}_{(n,n') \in D_L^+} \left[ - \log \left( P_{n,n'} \cdot \prod_{n'' \in \mathcal{N}(n)} (P_{n,n''})^{\text{weight}} \right) \right] \\
\text{其中，} \mathcal{N}(n) &= \{n''|(n,n'') \in D_U^+\}\\
\text{weight} &= \beta \cdot \hat{s}_{\theta}(n, n'') \\
\hat{s}_{\theta}(n, n'') &= \frac{s_{\theta}(n, n'') - \min \{ s_{\theta}(n_i, n''_j) \}_{i,j=1}^N}{\max \{ s_{\theta}(n_i, n''_j) \} - \min \{ s_{\theta}(n_i, n''_j) \}_{i,j=1}^N} \\
P_{u_i,v_i} &= \frac{s_{\theta}(u_i, v_i)}{\sum_{j \neq i, j=1}^N s_{\theta}(u_i, u_j) + \sum_{j=1}^N s_{\theta}(u_i, v_j)} \\
s_{\theta}(u_i, v_j) &= \exp(\text{cos}(\mathbf{U}_i, \mathbf{V}_j) / \tau)
\end{aligned}
$$
对于 $P_{u,v}$ 的理解：

- 分子是 $(u,v)$ 的相似度
- 分母是结点 $u$ 与其他一切结点的相似度之和

## 补充概率论证明

证明：$E(X+Y) = E(X) + E(Y)$ 永远成立
$$
\begin{aligned}
E(X+Y) &= \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} (x+y) f(x,y) \,dx\,dy \\
&= \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} x f(x,y) \,dx\,dy + \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} y f(x,y) \,dx\,dy \\
&= \int_{-\infty}^{\infty} x f_X(x) \,dx + \int_{-\infty}^{\infty} y f_Y(y) \,dy \\ &= E(X) + E(Y)
\end{aligned}
$$

## 算法流程

为了降低单个错误 $D_U^+$ 样本，即语义并不近似的结点对 $(u,v)$ 被错误的加入到正样本集中的累积影响，对 $D_U^+$ 采取动态更新策略。具体来说，由于初始编码器 $\theta^{(0)}$ 完全不能识别语义，因此需要先进行 $M$ 个回合的传统 $GCL$ ，得到能够初步识别语义的编码器 $\theta^{(1)}$ 。接着进行 $T$ 个回个的训练，每个回合开始时，利用当前的编码器 $\theta^{t}(t \in [1,T])$ 构造分类器，挖掘潜在正样本，实现重采样 $\text{(resampling)}$ ，然后计算损失函数并更新编码器 $\theta^{(t)}$ $K$ 次。最终得到效果最好的编码器 $\theta_{corrected}^*$ 
$$
\begin{array}{l}
\hline
\mathbf{Algorithm\ 1} \text{ Pre-training process of semantic-guided sampling for GCL} \\
\hline
\textbf{Input: } \text{Graph } \mathbf{G}, aug_1, aug_2, \text{encoder } \theta, \text{epochs } M, K, \text{threshold } t_s, \text{times } T, \text{learning rate } \alpha \\
\textbf{Output: } \text{optimized graph encoder } \theta_{corrected}^* \\
\hline
\mathbf{G}^{aug_1} \leftarrow aug_1(\mathbf{G}), \mathbf{G}^{aug_2} \leftarrow aug_2(\mathbf{G}) \\
D^+ \leftarrow D^{aug+}, D^- \leftarrow D^{aug-} \hspace{5em} \triangleright \text{Eq.(1)} \\
\theta^{(0)} \leftarrow \theta \\
L^{(0)} \leftarrow L(\theta^{(0)}) \hspace{9.4em} \triangleright \text{Eq.(3)} \\
\textbf{for } i = 1 \textbf{ to } M \textbf{ do} \\
\quad \theta^{(0)} \leftarrow \theta^{(0)} - \alpha \cdot \nabla_{\theta} L^{(0)} \hspace{4.5em} \triangleright \text{Warming Up } \theta^{(0)} \\
\textbf{end for} \\
\theta^{(1)} \leftarrow \theta^{(0)} \\
\textbf{for } t = 1, \dots, T \textbf{ do} \\
\quad h^{(t)}(n, n'; \theta^{(t)}) \leftarrow \text{sign}(s_{\theta^{(t)}}(n, n') - t_s) \hspace{1.2em} \triangleright \text{Free-Lunch} \\
\quad D_U^{+, (t)} \leftarrow \{(n, n')\}_{h^{(t)}(n, n'; \theta^{(t)}) = 1} \\
\quad D^+ \leftarrow D^{aug+} \cup D_U^{+, (t)} \\
\quad D^- \leftarrow D^{aug-} - D_U^{+, (t)} \hspace{4.8em} \triangleright \text{Resample} \\
\quad L^{(t)} \leftarrow L^{corrected}(\theta^{(t)}) \hspace{4.6em} \triangleright \text{Eq.(27)} \\
\quad \textbf{for } i = 1 \textbf{ to } K \textbf{ do} \\
\quad \quad \theta^{(t)} \leftarrow \theta^{(t)} - \alpha \cdot \nabla_{\theta} L^{(t)} \hspace{1.3em} \triangleright \text{Update Objective} \\
\quad \textbf{end for} \\
\textbf{end for} \\
\textbf{Return } \theta_{corrected}^* \leftarrow \theta^{(T)} \\
\hline
\end{array}
$$
