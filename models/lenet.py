"""
LeNet-5 模型实现
================
LeNet-5 是 Yann LeCun 于 1998 年提出的经典卷积神经网络，最初用于手写数字识别（MNIST）。
这是最早的 CNN 之一，奠定了卷积→池化→全连接的基本架构范式。

原始 LeNet-5 输入为 32x32 灰度图，本项目适配为 32x32 RGB 彩色图（3 通道）。
结构简洁、参数量少，是理解 CNN 基础的最佳起点。

结构概览:
    输入 (3, 32, 32)
    → Conv1 (6 个 5x5 卷积核) → ReLU → MaxPool (2x2)
    → Conv2 (16 个 5x5 卷积核) → ReLU → MaxPool (2x2)
    → Flatten
    → FC1 (120) → ReLU
    → FC2 (84) → ReLU
    → FC3 (10)  → 输出各类别得分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """LeNet-5 卷积神经网络（适配 CIFAR-10 版本）。

    与原始 LeNet-5 的区别:
    - 输入通道从 1 改为 3（适应 RGB 彩色图）
    - 输出类别从 10 保持不变（CIFAR-10 也是 10 类）
    - 全连接层输入维度根据 CIFAR-10 的 32x32 输入重新计算

    设计思想:
    - 卷积层提取空间特征（边缘、纹理、形状等）
    - 池化层降低空间分辨率，减少参数量，提供平移不变性
    - 全连接层整合全局特征，进行分类决策
    """

    def __init__(self, num_classes: int = 10):
        """初始化 LeNet-5 的各层。

        参数:
            num_classes: 分类类别数，CIFAR-10 默认为 10。
        """
        super(LeNet5, self).__init__()

        # ========== 特征提取器（卷积层）==========
        # 第一组：卷积 + 池化
        # 输入: (3, 32, 32) → 输出: (6, 28, 28)
        # 计算公式: 输出尺寸 = (输入尺寸 - 卷积核大小 + 2×填充) / 步长 + 1
        #          = (32 - 5 + 0) / 1 + 1 = 28
        self.conv1 = nn.Conv2d(
            in_channels=3,          # RGB 三通道输入
            out_channels=6,         # 6 个卷积核，学习 6 种不同的低级特征
            kernel_size=5,          # 5x5 卷积核，比现代常用的 3x3 大
            stride=1,               # 步长为 1，逐像素滑动
            padding=0,              # 不填充，原始 LeNet 的设计
        )
        # 池化后: (6, 14, 14)，分辨率减半，减少后续计算量
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二组：卷积 + 池化
        # 输入: (6, 14, 14) → 输出: (16, 10, 10)
        self.conv2 = nn.Conv2d(
            in_channels=6,          # 接收第一层 6 个特征图
            out_channels=16,        # 16 个卷积核，学习更复杂的中级特征组合
            kernel_size=5,          # 5x5 卷积核
            stride=1,
            padding=0,
        )
        # 池化后: (16, 5, 5)，最终特征图大小为 5x5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ========== 分类器（全连接层）==========
        # 经过两次池化后，特征图大小为 (16, 5, 5) = 400 个特征
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # 第一层全连接：400 → 120
        self.fc2 = nn.Linear(120, 84)             # 第二层全连接：120 → 84
        self.fc3 = nn.Linear(84, num_classes)     # 输出层：84 → 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数:
            x: 输入张量，形状为 (batch_size, 3, 32, 32)

        返回:
            输出张量，形状为 (batch_size, num_classes)，每个值代表该类别的未归一化得分
        """
        # 特征提取阶段
        x = self.pool1(F.relu(self.conv1(x)))  # Conv1 → ReLU → MaxPool
        x = self.pool2(F.relu(self.conv2(x)))  # Conv2 → ReLU → MaxPool

        # 将多维特征图展平为一维向量
        x = torch.flatten(x, start_dim=1)  # 从通道维度开始展平，保留 batch 维度

        # 分类阶段
        x = F.relu(self.fc1(x))  # 第一层全连接 + ReLU 激活
        x = F.relu(self.fc2(x))  # 第二层全连接 + ReLU 激活
        x = self.fc3(x)          # 输出层（不接激活函数，由损失函数内部处理）

        return x
