+++ 
draft = false
date = 2026-01-28T21:30:58+08:00
title = "[算法随笔]C++自定义sort比较函数"
description = ""
slug = "custom-sort-comparison-function"
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

## 基础用法

```cpp
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> nums = {4, 3, 5, 1, 8, 1};
    
    // 1. default: less<>()
    sort(nums.begin(), nums.end());
    for(int n : nums) {
        cout << n << ' ';
    }
    cout << " : sort with the default third parameter" << endl;

    // 2. sort the elements in non-descending order -- same as default
    sort(nums.begin(), nums.end(), less<>());
    for(int n : nums) {
        cout << n << ' ';
    }
    cout << " : sort with parameter: less<>()" << endl;

    // 3. sort the elements in non-growing order
    sort(nums.begin(), nums.end(), greater<>());
    for(int n : nums) {
        cout << n << ' ';
    }
    cout << " : sort with parameter: greater<>()" << endl;
}

Output:
1 1 3 4 5 8  : sort with the default third parameter
1 1 3 4 5 8  : sort with parameter: less<>()
8 5 4 3 1 1  : sort with parameter: greater<>()
```

STL库函数的实现是：快速排序 + 堆排序 + 插入排序
首先调用快排，对于数据进行依据pivot的分割，递归调用，
对于较大的数据，建立heap堆排序，
对于较小的数据：插入排序

## 自定义比较函数

### 传入自定义函数指针

```cpp
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

// 自定义 cmp 实现从大到小排序
bool cmp(int a, int b) {
    return a > b;
}

int main() {
    vector<int> nums = {4, 3, 5, 1, 8, 1};

    sort(nums.begin(), nums.end(), cmp);
    for(int n : nums) {
        cout << n << ' ';
    }
    cout << " : sort with function pointer: cmp" << endl;
}

Output:
8 5 4 3 1 1  : sort with function pointer: cmp
```

### 传入一个包含仿函数的对象

以下对于仿函数的介绍引用自：https://www.cnblogs.com/kazusarua/p/17960553

> **仿函数**：
>
> 定义一个类，类里面定义了某个方法，将该类的对象作为函数的入参，那么在函数中就能调用这个类中的方法。
>
> 或者，定义一个类，类里面重载函数运算符（），将该类的对象作为函数的入参，那么在函数中同样能调用重载符（）里面的方法
>
> 函数对象的出现是为了代替函数指针的，最明显的一个特点是：可以使用内联函数。而如果使用内联函数的指针，编译器会把它当普通函数对待。另外，函数对象是类封装的，代码不但看起来简洁，设计也灵活，比如还可以用关联，聚合，依赖的类之间的关系，与用到他们的类组合在一起，这样有利于资源的管理（这点可能是它相对于函数最显著的优点了）。
>
> 仿函数（Functor）也是 **STL 六大模块**之一，其余 5 个分别是容器（Container）、算法（Algorithm）、迭代器（Iterator）、适配器（Adapter）和分配器（Allocator）。
>

```cpp
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

// 利用包含仿函数的类实现从大到小排序
class cmp2 {
    public:
    bool operator() (int a, int b) {
        return a > b;
    }
};

int main() {
    vector<int> nums = {4, 3, 5, 1, 8, 1};
    
    sort(nums.begin(), nums.end(), cmp2());
    for(int n : nums) {
        cout << n << ' ';
    }
    cout << " : sort with objects containing functors " << endl;

}

Output:
8 5 4 3 1 1  : sort with objects containing functors
```

### 重载 < 运算符

```cpp
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

// 1. 定义一个自定义结构体来包装你的数据
struct Pair {
    int x, y;

    // 2. 在结构体内部重载 < 运算符
    // const 修饰符表示这个函数不会修改自身
    bool operator<(const Pair& other) const {
        return (x + y) < (other.x + other.y);
    }
};

int main() {
    // 3. 将数据存储为 vector<Pair> 而不是 vector<vector<int>>
    vector<Pair> nums = {{1, 3}, {2, 2}, {1, 1}, {3, 5}};

    // 4. 直接调用 sort，无需第三个参数
    // 因为 Pair 类型已经自带了 < 的逻辑
    sort(nums.begin(), nums.end());

    for(const auto& v : nums) {
        cout << '{' << v.x << ',' << v.y << "} "; 
    }
    cout << endl;
    
    return 0;
}

Output:
{1,1} {1,3} {2,2} {3,5}
```

### 使用lambda匿名函数

```cpp
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;


int main() {
    vector<vector<int>> nums = {{1, 3}, {2, 2}, {1, 1}, {3, 5}};
    
    sort(nums.begin(), nums.end(), [](const vector<int>&a, const vector<int>&b) {
        return a[0] + a[1] < b[0] + b[1];
    });

    for(const vector<int>& v : nums) {
        cout << '{' << v[0] << ',' << v[1] << "} "; 
    }
    cout << "sort with lambda function" << endl;
    
}

Output:
{1,1} {1,3} {2,2} {3,5} sort with lambda function
```


