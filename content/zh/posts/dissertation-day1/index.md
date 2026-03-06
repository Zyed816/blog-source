+++ 
draft = false
date = 2026-02-01T21:54:58+08:00
title = "[毕业设计]Day1"
description = ""
slug = "dissertation-day1"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

## 图对比学习的定义

> **Problem Formulation**. Let **G** = (**X**, **A**) denote a graph, where **X** ∈ $\mathbb{R}^{N\times F}$ denotes the nodes’ feature map, and $x_i$ is the feature of $i$-th node $n_i$. **A** ∈ $\mathbb{R}^{N\times N}$ denotes the adjacency matrix, where $A_{ij}$ = 1 if and only if there is an edge from $n_i$ to $n_j$. GCL aims at training a GNN encoder $f_\theta$(**G**) that maps graph **G** into the node representations **H** ∈ $\mathbb{R}^{N\times d}$ in a low-dimensional space, which captures the essential intrinsic information from both features and structure.

对于每篇论文有一个长度为 $3$ 的特征向量 $x_i$ ，每个位置上的数字代表与某领域的关联度，各领域顺序为 $\text{[图 医药 芯片]}$。例如，某论文特征向量为 $\text{[1.0 0 0]}$ ，说明这是一片图领域的论文，不涉及医药以及芯片领域的内容。

现有五篇论文，它们的特征向量 $x_{1-5}$ 构成特征矩阵 $X$
$$
X = \begin{pmatrix}
1.0 & 1.0 & 0.0 \\
1.0 & 0.9 & 0.0 \\
0.9 & 1.0 & 0.1 \\
0.0 & 0.1 & 1.0 \\
0.1 & 0.0 & 0.9
\end{pmatrix}
\begin{matrix}
\leftarrow n_1(\text{图+医药}) \\
\leftarrow n_2(\text{图+医药}) \\
\leftarrow n_3(\text{图+医药}) \\
\leftarrow n_4(\text{芯片}) \\
\leftarrow n_5(\text{芯片})
\end{matrix}
$$
邻接矩阵 $A$ 用来表示论文之间的引用关系，$A_{ij}$ = 1 表示论文 $n_i$ 引用了论文 $n_j$，这五篇论文的引用关系如下
$$
A = \begin{pmatrix}
0 & 1 & 1 & 0 & 0 \\
1 & 0 & 1 & 0 & 0 \\
1 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 \\
\end{pmatrix}
$$
图对比学习的目的就是得到一个编码器 $f_\theta$ ，它能将输入 $G = (X,A)$ 映射成一个 $N \times d$ 的矩阵 $H$，例如一个 $f_\theta$ 将  $G$ 映射成
$$
H = \begin{pmatrix}
0.97 & 0.97 & 0.03 \\
0.97 & 0.97 & 0.03 \\
0.97 & 0.97 & 0.03 \\
0.05 & 0.05 & 0.95 \\
0.05 & 0.05 & 0.95 \\
\end{pmatrix}
$$
在这个矩阵中， $n_1$ , $n_2$ , $n_3$ 对应的向量 $h_1$ , $h_2$ , $h_3$ 非常接近，可以将他们归为同一类论文。 $n_4$ , $n_5$ 同理

## 通用范式

### 得到增强视图的节点表示($\text{node representation}$)

> 节点表示(node representaion)指的是经过编码器 $f_\theta$ 映射后的 $N * d$ 矩阵，也就是之前提到的 $H$ 。这里得到两个节点表示，分别记为 $U$ ，$V$ 

利用特征掩蔽($\text{feature masking}$)和边丢弃($\text{edge dropping}$)技术生成原图 $G$ 的两个增强视图 $G^{aug1}$ 和 $G^{aug2}$ 。例如，论文 $n_1$ 的特征向量经过特征掩蔽后变为
$$
n_1 = \begin{pmatrix}
1.0 & 0.0 & 0.0
\end{pmatrix}
$$
那么
$$
X^{aug1} = \begin{pmatrix}
1.0 & 0.0 & 0.0 \\
1.0 & 0.9 & 0.0 \\
0.9 & 1.0 & 0.1 \\
0.0 & 0.1 & 1.0 \\
0.1 & 0.0 & 0.9
\end{pmatrix}
\qquad % 这里增加了1个大空格
A^{aug1} = A = \begin{pmatrix}
0 & 1 & 1 & 0 & 0 \\
1 & 0 & 1 & 0 & 0 \\
1 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 \\
\end{pmatrix}
\qquad % 这里增加了1个大空格
G^{aug1} = (X^{aug1},A^{aug1})
$$
论文 $n_1$ 的邻接矩阵向量经过边丢弃，失去 $n_1$ 对 $n_2$ 的引用
$$
a_1 = \begin{pmatrix}
0 & 0 & 1 & 0 & 0
\end{pmatrix}
$$
那么
$$
X^{aug2} = X =  \begin{pmatrix}
1.0 & 1.0 & 0.0 \\
1.0 & 0.9 & 0.0 \\
0.9 & 1.0 & 0.1 \\
0.0 & 0.1 & 1.0 \\
0.1 & 0.0 & 0.9
\end{pmatrix}
\qquad % 这里增加了1个大空格
A^{aug2} = \begin{pmatrix}
0 & 0 & 1 & 0 & 0 \\
1 & 0 & 1 & 0 & 0 \\
1 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 \\
\end{pmatrix}
\qquad % 这里增加了1个大空格
G^{aug2} = (X^{aug2},A^{aug2})
$$
然后分别经过 $f_\theta$ 映射得到
$$
U = f_\theta(\mathbf{G}^{aug1}) = \begin{pmatrix}
U_1 \\
U_2 \\
U_3 \\
U_4 \\
U_5 \\
\end{pmatrix}
\qquad
V = f_\theta(\mathbf{G}^{aug2}) = \begin{pmatrix}
V_1 \\
V_2 \\
V_3 \\
V_4 \\
V_5 \\
\end{pmatrix}
$$

### 划分正负样本对集合

由同一个结点编码得到的向量 $u_i$ $v_i$ 组成两个正样本对 $\{(u_i,v_i),(v_i,u_i)\}$ 。正样本对集合
$$
D^{aug+} = \{(u_i,v_i),(v_i,u_i)\}^{N}_{i=1} = \{(u_1,v_1),(v_1,u_1),\ldots,(u_5,v_5),(v_5,u_5)\}
$$
其他结点对组成负样本对集合
$$
D^{aug-} = \{(u_i,v_i),(v_i,u_i)\}^{N}_{i \ne j,\ i,j=1}
$$

### 计算损失函数并优化$f_\theta$

$\text{InfoNCE}$ 损失函数计算方式
$$
L = \frac{1}{2N} \sum_{i=1}^N (l_{u_i, v_i} + l_{v_i, u_i})
$$
其中
$$
l_{u_i, v_i} = -\log \frac{s_{\theta}(u_i, v_i)}{\sum_{j \ne i,\ j=1}^N s_{\theta}(u_i, u_j) + \sum_{j=1}^N s_{\theta}(u_i, v_j)}
$$
$s_\theta(u_i,v_i)$ 为 $u_i$ , $v_i$ 之间的余弦相似度，计算方法为
$$
s_{\theta}(u_i, v_j) = \exp(\text{cos}(\mathbf{U}_i, \mathbf{V}_j) / \tau)
$$
注意：$u_i,v_i$ 表示不同增强视图中的结点，$\mathbf{U_i,V_i}$ 表示该节点的编码向量 

得到总损失 $L$ 后反向传播优化 $f_\theta$ 
