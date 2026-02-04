+++ 
draft = false
date = 2026-01-31T20:43:58+08:00
title = "[Breakpoint]查找和最小的K对数字"
description = ""
slug = "breakpoint-kSmallestPairs"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

## 题目

[查找和最小的K对数字](https://leetcode.cn/problems/find-k-pairs-with-smallest-sums/?envType=problem-list-v2&envId=shujujiegouyusuanfa-xuliegudi-dui)

## 暴力解法

### 思想

找到所有的 `(u, v)` ，按照 `u + v` 从小到大排列，取出前 `k` 个即为答案

### 代码

```cpp
class Solution {
public:
    vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        vector<vector<int>> ans;
        vector<vector<int>> record;
        for(int i = 0; i < nums1.size(); i ++) {
            for(int j = 0; j < nums2.size(); j ++) {
                record.push_back({nums1[i], nums2[j]});
            }
        }
        sort(record.begin(), record.end(), [](const vector<int>& a, const vector<int>& b) {
            return (a[0] + a[1]) < (b[0] + b[1]);
        });
        for(int i = 0; i < k && i < record.size(); i++) {
            ans.push_back(record[i]);
        }
        return ans;
    }
};
```

### 复杂度

时间复杂度： `O(nmlog(nm))` ，其中 `n = nums1.size(), m = nums2.size()`

空间复杂度： `O(nm)` ， `record` 中有 `nm` 个元素

## 使用堆

### 思想

想象面前有一排高低不等的凳子，编号为 `i` 的凳子高度为 `nums1[i]` ,每个凳子后面都有一队人，并且第 `j` 个人的身高为 `nums[j]` ，题目的意思就是找出使得人站上凳子之后总高度最低的 `k` 个人凳组合 `{nums1[i], nums2[j]}`

我们可以先让每支队伍的第一个人都站上凳子，从中选择最低的一个，记录下来，然后让下一个人（如果有）站上去，重复这个过程直至得到 `k` 组数据

瓶颈在于每次如何快速选择出总高度最低的那个人凳组合，如果每次排序，时间复杂度将为 `O(k * nlogn)` ，其中 `n` 为 `nums1` 的大小

如果维护一个大小为 `n` 的小顶堆，堆中的值为当前所有的总高度，就可以在 `O(1)` 的时间内找到最小值。将其弹出，表示当前站在凳子上的人下来，如何表示让队伍中的下一个人站上这个凳子呢？因此我们要知道发生人员变动的凳子位置 `i` 以及现在站在凳子上的人在其队伍中的序号 `j` ，堆中就不能只存储总高度 `height` ，而是应该同时存储位置信息 `{i, j}` ，结构应为 `{height, {i, j}}` 。构建小顶堆，每次选择堆顶，记录答案后将该结点弹出，并判断 `j + 1` 是否小于 `nums2.size()` ，如果小于，将 `{height, {i, j + 1}}` 入堆表示下一个人站上凳子，其中 `height = nums1[i] + nums2[j + 1]`

### 代码

```cpp
class Solution {
public:
    vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        vector<vector<int>> ans;
        int n = nums1.size();
        int m = nums2.size();
        if(n == 0 || m == 0)    return ans;

        using PII = pair<int, pair<int, int>>;
        priority_queue<PII, vector<PII>, greater<PII>> pq;
        for(int i = 0; i < n; i ++) {
            pq.push({nums1[i] + nums2[0], {i, 0}});
        }

        while(k --) {
            pair<int, pair<int, int>> top = pq.top();
            pq.pop();
            int curSum = top.first;
            int i = top.second.first;
            int j = top.second.second;

            ans.push_back({nums1[i], nums2[j]});
            if(j + 1 <  m) {
                pq.push({nums1[i] + nums2[j + 1], {i, j + 1}});
            }
        }

        return ans;
    }
};
```

注意：

`priority_queue` 的定义原型是 `priority_queue<Type, Container, Comparator>`

- 默认是：`priority_queue<Type, vector<Type>, less<Type>>` (大顶堆)

- **规则**：如果你想改 `Comparator`，你**不能**省略中间的 `Container`。

### 复杂度

时间复杂度： `O(max(k, n) * logn)` ，`priority_queue` 的 `push` 和 `pop` 操作时间复杂度均为 `O(log(heap_size))` ，构建堆时间复杂度为 `O(nlogn)` ，更新和记录答案时间复杂度为 `O(klogn)`

空间复杂度： `O(n)` 堆中元素最多 `n` 个

## 剪枝

### 思想

之前我们观察所有凳子，构建了一个大小为 `n = nums1.size()` 的堆，但是总高度最低的人凳组合根本不可能出现在第 `k` 个凳子之后，因为：

- 因为 `nums[2].size() >= 1` ，因此前 `k` 个凳子一定可以产生大于等于 `k` 中可能
- 在这种情况下，如果编号为 `K(K > k)` 的凳子可以产生一组答案 `{nums1[K], nums2[j]}` ，这个答案一定可以被 `nums1[k], nums2[j]` 替换

因此，只需要构建一个大小为 `min(n, k)` 的堆

### 代码

在构建堆时，如果堆大小等于 `k`， 停止构建

```cpp
for(int i = 0; i < n && i < k; i ++) {
            pq.push({nums1[i] + nums2[0], {i, 0}});
        }
```

### 复杂度

时间复杂度： `O(k * log(min(n, k)))`

空间复杂度： `O(k)`

