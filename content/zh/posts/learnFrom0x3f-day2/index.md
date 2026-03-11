+++ 
draft = false
date = 2026-03-11T10:10:55+08:00
title = "[跟着灵神学算法]Day2"
description = ""
slug = ""
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

## 不定长滑动窗口（越短越合法/求最长/最大）

### 题目

[无重复的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/)

### 模板

```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int ans = 0;
        int i = 0;
        unordered_map<char, int> cnt;
        for(int j = 0; j < s.size(); j ++) {
            while(cnt[s[j]] >= 1) {
                cnt[s[i ++]] --;
            }
            cnt[s[j]] ++;
            ans = max(ans, j - i + 1);
        }
        return ans;
    }
};
```

 $i$ 标志滑动窗口左端点， $j$ 标志滑动窗口右端点， $j$ 每次右移之后，先调整窗口左端点 $i$ ，调整到窗口可以包含 $s[j]$ 为止，然后更新答案，并将 $s[j]$ 记录进 $cnt$

注意 $0 \le cnt[s[k]] \le 1(0 \le k \le j),cnt[s[l]] = 1(i \le k \le j)$ ，不会有负数出现

### 复杂度

时间复杂度： $O(n)$ ，$n$ 为 $s$ 长度

空间复杂度： $O(1)$

### 优化模板

根据 [尽可能使字符串相等](https://leetcode.cn/problems/get-equal-substrings-within-budget/description/) 可以得到结论：$j$ 每次右移后，先更新统计量，然后根据限制条件移动 $i$ 到合法位置，更方便

```cpp
class Solution {
public:
    int equalSubstring(string s, string t, int maxCost) {
        int ans = 0;
        int i = 0;
        int cost = 0;
        for(int j = 0; j < s.size(); j ++) {
            // 先更新统计量
            cost += abs(s[j] - t[j]);
            // 然后根据限制条件移动 i
            while(cost > maxCost) {
                cost -= abs(s[i] - t[i]);
                i ++;
            }
            ans = max(ans, j - i + 1);
        }
        return ans;
    }
};
```


