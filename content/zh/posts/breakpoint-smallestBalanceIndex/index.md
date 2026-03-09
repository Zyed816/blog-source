+++ 
draft = false
date = 2026-03-08T20:26:00+08:00
title = "[Breakpoint]找出最小平衡下标"
description = ""
slug = "breakpointSmallestBalancedIndex"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

## 题目

[找出最小平衡下标](https://leetcode.cn/problems/find-the-smallest-balanced-index/description/)

## 尝试解决

### 思路

计算前缀和与后缀积，遍历下标，满足条件时返回

### 代码

```cpp
class Solution {
public:
    int smallestBalancedIndex(vector<int>& nums) {
        int n = nums.size();
        vector<int> sumLeft(n, 0);
        vector<int> pdtRight(n, 1);
        for(int i = 1; i < n; i ++) {
            sumLeft[i] = sumLeft[i - 1] + nums[i - 1]; 
        }
        for(int i = n - 2; i >= 0; i --) {
            pdtRight[i] = pdtRight[i + 1] * nums[i + 1];
        }

        int ans = -1;
        for(int i = 0; i < n; i ++) {
            if(sumLeft[i] == pdtRight[i]) {
                ans = i;
                break;
            }
        }
        return ans;
    }
};
```

### 结果

溢出

```bash
Line 11: Char 43: runtime error: signed integer overflow: 5283864854040960000 * 896 cannot be represented in type 'value_type' (aka 'long long') (solution.cpp)
SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior prog_joined.cpp:20:43
```

将 `sumLeft` ， `pdtRight` 存储类型改为 `long long` 仍然溢出

> C++ 中 int 范围为 $-2^{31} \sim 2^{31} - 1$ 即 $-2147483648 \sim 2147483647$ ，约 $-2 \times 10 ^{10} \sim   2 \times 10 ^{10}$ 
>
> long long 范围为 $-2^{63} \sim 2^{63} - 1$ ，约 $-10 ^{20} \sim 10 ^{20}$

## 题解

由于 $nums[i] > 0$ ，所以 $sumLeft$ 是严格递增的， $pdtRight$ 是（非严格）递减的，画出函数图像，只存在零个或一个解

可以先把 $sumLeft$ 算出来，然后倒着遍历 $nums$ ，同时计算 $pdt$ ，这样可以将 $pdtRight$ 简化为一个变量。

倒着遍历时，$pdt$ 不断增大， $sumLeft$ 不断减小，如果 $pdt == sumLeft$ ，返回下标；如果发现 $pdt$ 已经比 $sumLeft$ 大了，返回 $-1$

可以写出代码

```cpp
class Solution {
public:
    int smallestBalancedIndex(vector<int>& nums) {
        int n = nums.size();
        vector<long long> sumLeft(n, 0);
        for(int i = 1; i < n; i ++) {
            sumLeft[i] = sumLeft[i - 1] + nums[i - 1]; 
        }

        long long pdt = 1;
        int ans = -1;
        for(int i = n - 1; i >= 0; i --) {
            if(pdt > sumLeft[i]) {
             	break;
            } else if(pdt == sumLeft[i]) {
                ans = i;
                break;
            }
            pdt *= nums[i];
        }
        return ans;
    }
};
```

但是对于输入

```
nums = [1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,1000000000,359738368,1,536870913,536870912,64]
```

 $ptd$ 还是会溢出，原因是上面的代码在下标为 $i$ 时，只是判断了 $i$ 位置 $pdt$ 和 $sumLeft[i]$ 的关系，然后直接计算 `pdt *= nums[i]` ，即 $i - 1$ 位置上的 $pdt$ ，而没有提前判断是否溢出。因此在下标为 $i$ 时就要先判断 $i - 1$ 位置上的 $pdt$ 会不会溢出，判断方法为比较 $pdt \times nums[i]$ 与 $sumLeft[i - 1]$ 的大小，如果 $pdt \times nums[i] > sumLeft[i - 1]$ ，说明后面不会有满足条件的结果了，返回 $-1$ ，而为了避免溢出，选择除法形式即比较 $pdt$ 与 $sumLeft[i - 1] / nums[i]$ 的大小关系

```cpp
class Solution {
public:
    int smallestBalancedIndex(vector<int>& nums) {
        int n = nums.size();
        vector<long long> sumLeft(n, 0);
        for(int i = 1; i < n; i ++) {
            sumLeft[i] = sumLeft[i - 1] + nums[i - 1]; 
        }

        long long pdt = 1;
        int ans = -1;
        for(int i = n - 1; i > 0; i --) {		// 因为下面有索引 [i - 1] ，所以这里 i > 0
            if(pdt == sumLeft[i]) {				// 又因为 sumLeft[0] = 0 而 pdtRight[0] > 0
                ans = i;						// 所以 0 不可能是答案，因此不用考虑特判 i = 0
                break;
            }
            if(pdt > sumLeft[i - 1] / nums[i]) {
                break;
            }
            pdt *= nums[i];
        }
        
        return ans;
    }
};
```

此时时间复杂度 $O(n)$ ，空间复杂度 $O(n)$

先计算 $nums$ 的和，然后在倒序遍历的过程中减去遍历的数，也能求出前缀和。这样可以做到 $O(1)$ 空间

### 代码

```cpp
class Solution {
public:
    int smallestBalancedIndex(vector<int>& nums) {
        int n = nums.size();
        long long sum = reduce(nums.begin(), nums.end(), 0LL);

        long long pdt = 1;
        int ans = -1;
        for(int i = n - 1; i > 0; i --) {		// 因为下面有索引 [i - 1] ，所以这里 i > 0
            sum -= nums[i];
            if(pdt == sum) {				// 又因为 sumLeft[0] = 0 而 pdtRight[0] > 0
                ans = i;						// 所以 0 不可能是答案，因此不用考虑特判 i = 0
                break;
            }
            if(pdt > sum / nums[i]) {
                break;
            }
            pdt *= nums[i];
        }
        
        return ans;
    }
};
```

### 复杂度

时间复杂度： $O(n)$

空间复杂度： $O(1)$

