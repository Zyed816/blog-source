+++ 
draft = false
date = 2026-02-14T20:43:58+08:00
title = "[算法随笔]std::unordered_set用法"
description = ""
slug = "usageOfUnderedSet"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

 `unordered_set` 不保证元素有序性，能在 $O(1)$ 的时间内完成查找，增加和删除任务

## 构造函数

创建一个空的 `unordered_set`

```cpp
std::unordered_set<int> uset;
```

## 插入元素

使用 `insert()` 方法

```cpp
uset.insert(10)
```

## 查找元素

使用 `find()` 方法

```cpp
auto it = uset.find(10);
if(it != uset.end()) {
	// 元素存在
}
```

使用 `contains()` 方法

```cpp
if(uset.contains(10)) {
	// 元素存在
}
```

## 删除元素

### 删除指定元素

```cpp
uset.erase(10);
```

### 利用迭代器删除

```cpp
auto it = uset.find(10);
uset.erase(it);
```

### 删除范围内元素

```cpp
uset.erase(uset.begin(), uset.end());
```

### 删除所有元素

```cpp
uset.clear();
```

## 获取大小和空检查

```cpp
size_t size = uset.size();
bool isEmpty = uset.empty();
```


