+++ 
draft = false
date = 2026-01-27T19:39:46+08:00
title = "[算法随笔]C++中的堆"
description = ""
slug = "heat-in-Cpp"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++


## 题目

[最后一块石头的重量](https://leetcode.cn/problems/last-stone-weight/?envType=problem-list-v2&envId=shujujiegouyusuanfa-xuliegudi-dui)

有一堆石头，每块石头的重量都是正整数。

每一回合，从中选出两块 **最重的** 石头，然后将它们一起粉碎。假设石头的重量分别为 `x` 和 `y`，且 `x <= y`。那么粉碎的可能结果如下：

- 如果 `x == y`，那么两块石头都会被完全粉碎；
- 如果 `x != y`，那么重量为 `x` 的石头将会完全粉碎，而重量为 `y` 的石头新重量为 `y-x`。

最后，最多只会剩下一块石头。返回此石头的重量。如果没有石头剩下，就返回 `0`。

**示例：**

```
输入：[2,7,4,1,8,1]
输出：1
解释：
先选出 7 和 8，得到 1，所以数组转换为 [2,4,1,1,1]，
再选出 2 和 4，得到 2，所以数组转换为 [2,1,1,1]，
接着是 2 和 1，得到 1，所以数组转换为 [1,1,1]，
最后选出 1 和 1，得到 0，最终数组转换为 [1]，这就是最后剩下那块石头的重量。
```

**提示：**

- `1 <= stones.length <= 30`
- `1 <= stones[i] <= 1000`

## 暴力

### 思想

每次要选取数组中的最大值和次大值，可以通过 `sort` 实现从小到大排列，然后取最后一个元素即为最大值 `y` ，倒数第二个元素即为次大值 `x` ，然后再计算  `y - x` ，如果结果不为零就将结果 `push_back` 回数组。不断模拟直到数组的大小变为 `1` 或 `0`

### 代码

```C++
class Solution {
public:
    int lastStoneWeight(vector<int>& stones) {
        int ans;
        while(true) {
            if(stones.size() == 1) {
                ans = stones[0];
                break;
            } else if(stones.size() == 0) {
                ans = 0;
                break;
            }
            sort(stones.begin(), stones.end());
            int n = stones.size();
            int y = stones[n - 1];
            int x = stones[n - 2];
            stones.pop_back();
            stones.pop_back();
            if(y > x) {
                stones.push_back(y - x);
            }
        }
        return ans;
    }
};
```

### 复杂度

时间复杂度： `O(n * nlogn)`

空间复杂度： `O(1)`

## 使用堆

### 什么是堆

堆是一种满足特定条件的**完全二叉树**，分为**大顶堆**和**小顶堆**

- 大顶堆：任意节点的值大于其子节点的值

- 小顶堆：任意节点的值小于其子节点的值

**完全二叉树**：由高向低、从左到右依次填写各个节点的值，不得跳跃

### 使用优先级队列（priority_queue）创建和维护堆

堆（heaps） 是一种特殊的数据组织方式，STL 中的 `priority_queue` 容器适配器底层就是采用堆来组织数据存储的。
实际上，堆通常用作实现优先队列，大顶堆相当于元素按从大到小顺序出队的优先队列。从使用角度来看，我们可以将“优先队列”和“堆”看作等价的数据结构。因此，这里对两者不做特别区分，统一使用“堆“来命名。

#### 实例

**1.包含头文件**

```cpp
#include <queue>
```

**2.创建优先队列**

```cpp
std::priority_queue<int> maxHeap; // 创建一个大顶堆，默认情况下是大顶堆
```

**3.根据现有迭代器创建一个优先队列**

```cpp
priority_queue<int> pq(stones.begin(), stones.end());
```

**3.插入元素**

使用 `push` 方法将元素插入堆中，堆会**自动维护其性质。**

```cpp
maxHeap.push(10);
```

**4.访问堆顶元素**

```cpp
int mx = maxHeap.top();
```

**5.移除堆顶元素**

```cpp
maxHeap.pop();
```

**6.检查堆是否为空**

```cpp
bool isEmpty = maxHeap.empty();
```

**7.获取堆中元素数量**

```cpp
int maxHeapSize = maxHeap.size();
```

### 思想

根据 `stones` 创建一个大顶堆 `pq` ，通过两次取堆顶操作得到 `y` 和 `x` ，计算 `y - x` ，判断是否将结果重新入堆，直到堆大小变为 `1` 或 `0` 

### 代码

```cpp
class Solution {
public:
    int lastStoneWeight(vector<int>& stones) {
        int ans = 0;
        priority_queue<int> pq(stones.begin(), stones.end());
        while(pq.size() > 0) {
            if(pq.size() == 1) {
                ans = pq.top();
                break;
            }
            int y = pq.top();   pq.pop();
            int x = pq.top();   pq.pop();
            if(y - x) {
                pq.push(y - x);
            }
        }
        return ans;
    }
};
```

### 复杂度

时间复杂度： `O(nlogn)`

空间复杂度： `O(1)`
