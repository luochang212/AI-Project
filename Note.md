# 基础

## 安装

ubuntu

```
sudo apt update
sudo apt install build-essential
sudo apt install python3.8
```

install miniconda

```
pip install jupyter d2l torch torchvision
```

把远端8888端口映射到本地8888端口（以远端 用户名: ubuntu ip: 100.20.65.33 为例）

```
ssh -L8888:localhost:8888 ubuntu@100.20.65.33
```

幻灯片插件

```
pip install rise
```

## 线性代数

矩阵乘法，矩阵与向量乘法，向量正交

矩阵范数，矩阵F范数

正定矩阵，正交矩阵，置换矩阵

特征值，特征向量

偏导，可微，可积，连续

## 线性回归

线性回归可以看成是最简单的神经网络

线性回归是有显式解的

均方误差

马尔可夫链

## softmax回归：多分类问题

分类

类别的置信度

使用softmax操作子得到每个类的预测置信度：是一种计算每种类别概率的方法

argmax：取 arg i 最大化 o_i

恒量预测值和真实值的区别：交叉熵

## 感知机

解决分类问题，输入参数X可以是一个向量

等于计算一个超平面，对X进行线性分割

批量大小为1的梯度下降

无法解决XOR问题：多层感知机

## 多层感知机

如果激活函数使用线性函数，那么最终还是一个线性的模型，所以必须使用非线性的激活函数

非线性激活函数：
- sigmoid函数（0/1）
- tanh函数（-1/1）
- ReLU函数（0/+inf）




