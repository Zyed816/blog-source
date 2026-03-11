+++ 
draft = false
date = 2026-03-11T10:08:21+08:00
title = "[跟着灵神学算法]Day1"
description = ""
slug = ""
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

## 定长滑动窗口

### 题目

[1456.定长子串中元音的最大数目](https://leetcode.cn/problems/maximum-number-of-vowels-in-a-substring-of-given-length/)

### 我的思路

使用 $\text{left}$ ， $\text{right}$ 表滑动窗口的左边界和右边界，

### 我的代码

```cpp
class Solution {
public:
    int maxVowels(string s, int k) {
        int ans = 0;
        int left = 0;
        int right = 0;
        int curNum = 0;
        unordered_set<char> vowels = {'a', 'e', 'i', 'o', 'u'};
        for(right; right < k; right ++) {
            if(vowels.contains(s[right])) {
                curNum ++;
            }
        }
        right --;
        ans = curNum;
        left ++;
        right ++;
        for(right; right < s.size();left ++, right ++) {
            if(vowels.contains(s[left - 1])) {
                curNum --;
            }
            if(vowels.contains(s[right])) {
                curNum ++;
            }
            ans = max(ans, curNum);
        }
        return ans;
    }
};
```

### 模板

> 总结成三步：入-更新-出。
>
> 入：下标为 i 的元素进入窗口，更新相关统计量。如果窗口左端点 i−k+1<0，则尚未形成第一个窗口，重复第一步。
> 更新：更新答案。一般是更新最大值/最小值。
> 出：下标为 i−k+1 的元素离开窗口，更新相关统计量，为下一个循环做准备。
> 以上三步适用于所有定长滑窗题目
>
> 作者：灵茶山艾府
> 链接：https://leetcode.cn/problems/maximum-number-of-vowels-in-a-substring-of-given-length/solutions/2809359/tao-lu-jiao-ni-jie-jue-ding-chang-hua-ch-fzfo/
> 来源：力扣（LeetCode）
> 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```cpp
class Solution {
public:
    int maxVowels(string s, int k) {
        int cnt = 0;
        int ans = 0;
        int i = 0;
        int j = 0;
        while(j < s.size()) {
            if(s[j] == 'a' || s[j] == 'e' || s[j] == 'i' || s[j] == 'o' || s[j] == 'u') {
                cnt ++;
            }
            
            if(j - i + 1 == k) {
                ans = ans > cnt ? ans : cnt;
                if(s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' || s[i] == 'u') {
                    cnt --;
                }
                i ++;
            }

            j ++;
        }
        return ans;
    }
};
```

### 复杂度

时间复杂度： $O(n)$

空间复杂度： $O(1)$
