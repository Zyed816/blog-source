+++ 
draft = false
date = 2026-02-05T21:31:00+08:00
title = "[breakpoint]丑数"
description = ""
slug = "breakpoint-uglyNumber"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

## 题目

[丑数](https://leetcode.cn/problems/ugly-number/description/?envType=study-plan-v2&envId=primers-list)

## 暴力

### 思想

找出 $n$ 所有的因数 $i$ ，判断 $i$ 是否是素数并且 $i \in \{2,3,5\}$

### 代码

```cpp
class Solution {
private:
    bool isPrime(int n) {
        for(int i = 2; i * i < n; i ++) {
            if(n % i == 0) {
                return false;
            }
        }
        return true;
    }
public:
    bool isUgly(int n) {
        if(n <= 0)   return false;
        unordered_set<int> primes = {2, 3, 5};
        for(int i = 2; i <= n; i ++) {
            if(n % i == 0 && isPrime(i) && !primes.contains(i)) {
                return false;
            }
        }
        return true;
    }
};
```

### 复杂度

时间复杂度： $O(n * \sqrt{n})$ ，外层循环 $O(n)$ ， $isPrime()$ 函数 $O(\sqrt{n})$

空间复杂度： $O(1)$

## 不断缩小 $n$

### 思想

“丑数”（Ugly Number）的定义是：因子**只包含** $2, 3, 5$ 的正整数。 与其去寻找 $n$ 有哪些质因子，不如**反向操作**：不断尝试用 $2, 3, 5$ 去除 $n$。如果最后剩下的值是 $1$，说明它只包含这些因子；否则，它就不是丑数。

### 代码

```cpp
class Solution {
public:
    bool isUgly(int n) {
        // 丑数必须是正整数
        if (n <= 0) return false;

        // 依次用 2, 3, 5 尽可能地除 n
        int factors[] = {2, 3, 5};
        for (int f : factors) {
            while (n % f == 0) {
                n /= f;
            }
        }

        // 如果最后剩下的数是 1，说明所有的质因子都在 {2, 3, 5} 之中
        return n == 1;
    }
};
```

### 复杂度

时间复杂度： $O(logn)$ ， $n$ 有 $O(logn)$ 个因子 $2$ ，因子 $3$ 和因子 $5$ 

空间复杂度： $O(1)$