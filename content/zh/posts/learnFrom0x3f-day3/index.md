+++ 
draft = false
date = 2026-03-11T17:46:26+08:00
title = "[跟着灵神学算法]Day3"
description = ""
slug = ""
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

## 不定长滑动窗口（越长越合法/求最短/最小）

### 题目

[长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/description/)

### 代码（模板）

```cpp
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int ans = INT_MAX;
        int sum = 0;
        int i = 0;
        for(int j = 0; j < nums.size(); j ++) {
            sum += nums[j];
            while(sum >= target) {
                ans = min(ans, j - i + 1);
                sum -= nums[i ++];
            }
        }
        return ans == INT_MAX ? 0 : ans;
    }
};
```

### 复杂度

时间复杂度： $O(n)$

空间复杂度： $O(1)$

### 一点思考

越短越合法/求最长/最大维护的是一个满足限制的滑动窗口， $j$ 每次右移后，更新统计量，然后根据限制条件移动 $i$ ，最终得到下一个满足限制的窗口，更新答案

越长越合法/求最短/最小维护的是一个不满足限制的滑动窗口， $j$ 每次右移后，更新统计量，然后根据限制条件移动 $i$ ，过程中窗口是合法的，不断更新答案，最终得到下一个不满足限制的窗口

## 不定长滑动窗口（求子数组个数）

### 越短越合法

#### 题目

[乘积小于K的子数组](https://leetcode.cn/problems/subarray-product-less-than-k/description/)

#### 代码（模板）

```cpp
class Solution {
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        int ans = 0;
        int pdt = 1;
        int i = 0;
        for(int j = 0; j < nums.size(); j ++) {
            pdt *= nums[j]; 
            while(i <= j && pdt >= k) {
                pdt /= nums[i ++];
            }
            ans += j - i + 1;
        }
        return ans;
    }
};
```

维护的是一个满足限制的滑动窗口，统计子数组个数时，由于子数组越短，越能满足题目要求，所以除了 $[i,j]$，还有 $[i+1,j],[i+2,j],…,[i,j]$ 都是满足要求的，这样的数组有 $j - i + 1$ 个，因此有 `ans += j - i + 1`

#### 复杂度

时间复杂度： $O(n)$

空间复杂度： $O(1)$

### 越长越合法

#### 题目

[包含所有三种字符的子字符串数目](https://leetcode.cn/problems/number-of-substrings-containing-all-three-characters/description/)

#### 代码（模板）

```cpp
class Solution {
public:
    int numberOfSubstrings(string s) {
        unordered_map<char, int> cnt;
        int ans = 0;
        int i = 0;
        for(int j = 0; j < s.size(); j ++) {
            cnt[s[j]] ++;
            while(cnt['a'] > 0 && cnt['b'] > 0 && cnt['c'] > 0) {
                cnt[s[i ++]] --;
            }
            ans += i;
        }
        return ans;
    }
};
```

维护的是一个不满足限制的滑动窗口， $[i,j]$ 是不满足的，但是 $[i - 1, j]$ 是满足的，因为越长越满足，对于一个 $j$ ，除了 $[i - 1,j]$ 外， $[i - 2,j],[i - 3,j] ... [0,j]$ 都是满足的，一共有 $i$ 个，所以 `ans += i`

#### 复杂度

时间复杂度： $O(n)$

空间复杂度： $O(1)$

### 恰好型滑动窗口

#### 题目

[和相同的二元子数组](https://leetcode.cn/problems/binary-subarrays-with-sum/description/)

#### 代码（模板）

```cpp
class Solution {
private:
    int solve(vector<int>& nums, int k) {
        int ans = 0;
        int sum = 0;
        int i = 0;
        for(int j = 0; j < nums.size(); j ++) {
            sum += nums[j];
            while(i <= j && sum >= k) {
                sum -= nums[i];
                i ++;
            }
            ans += i;
        }
        return ans;
    }
public:
    int numSubarraysWithSum(vector<int>& nums, int goal) {
        return solve(nums, goal) - solve(nums, goal + 1);
    }
};
```

转化为两个”越长越合法“类型的问题，求差

在

```cpp
			while(i <= j && sum >= k) {
                sum -= nums[i];
                i ++;
            }
```

位置要注意判断是否会出现 `sum >= k` 恒成立的情况，如果存在，要补充限制条件 `i <= j`

#### 复杂度

时间复杂度： $O(n)$

空间复杂度： $O(1)$


