"""
SimpleCNN 模型实现
==================
这是一个教学用的三层卷积神经网络，在 LeNet 的基础上引入了两个现代训练技巧:
1. 批归一化（BatchNorm）: 对每层输出进行归一化，加速训练、允许更大的学习率
2. Dropout: 训练时随机"丢弃"部分神经元，防止过拟合

结构比 LeNet 深但比 VGG/ResNet 浅，是理解"现代 CNN 训练技巧"的最佳示范。

设计原则:
- 使用 conv_block 辅助函数封装"卷积+BatchNorm+ReLU+池化"模式，体现 DRY 原则
- 每层通道数翻倍（32→64→128），是 CNN 设计的常见模式
- 自适应平均池化解耦了特征图尺寸和全连接层的关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """三层卷积神经网络，包含 BatchNorm 和 Dropout。

    结构概览:
        输入 (3, 32, 32)
        → Conv Block 1 (3→32)  → (32, 16, 16)
        → Conv Block 2 (32→64) → (64, 8, 8)
        → Conv Block 3 (64→128)→ (128, 4, 4)
        → 自适应平均池化
        → Dropout → FC (2048→256) → ReLU
        → Dropout → FC (256→128)  → ReLU
        → FC (128→10) → 输出
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        """初始化 SimpleCNN 各层。

        参数:
            num_classes: 分类类别数，默认为 10。
            dropout: Dropout 丢弃率，默认为 0.3。值越大正则化越强。
        """
        super(SimpleCNN, self).__init__()

        # ========== 卷积块 ==========
        # 每个卷积块: Conv2d → BatchNorm2d → ReLU → MaxPool2d
        # BatchNorm 的作用: 对每个 mini-batch 的激活值做标准化
        #   - 使得每层输入分布稳定 → 缓解"内部协变量偏移"
        #   - 允许使用更大的学习率 → 训练更快
        #   - 有一定的正则化效果 → 减轻过拟合

        # Block 1: 3 → 32 通道，从 RGB 三通道提取 32 种低级特征
        self.block1 = self._make_conv_block(3, 32)
        # Block 2: 32 → 64 通道，组合低级特征形成中级特征
        self.block2 = self._make_conv_block(32, 64)
        # Block 3: 64 → 128 通道，组合中级特征形成高级特征
        self.block3 = self._make_conv_block(64, 128)

        # ========== 全局池化 ==========
        # 自适应平均池化：无论输入特征图多大，都输出 (4, 4) 的固定尺寸
        # 好处: 解耦了前面的卷积结构和后面的全连接层，修改卷积结构时不用重新计算维度
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # ========== 分类器 ==========
        # 经过自适应池化后，特征维度固定为 128 × 4 × 4 = 2048
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    @staticmethod
    def _make_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """创建一个标准的卷积块。

        组合: Conv2d → BatchNorm2d → ReLU → MaxPool2d
        使用 3x3 卷积 + padding=1 保持空间尺寸不变（仅在池化时降采样）。
        这是 VGG 发扬光大的设计模式：小卷积核 + 深网络。

        参数:
            in_channels: 输入通道数。
            out_channels: 输出通道数（卷积核数量）。

        返回:
            组合好的 Sequential 模块。
        """
        return nn.Sequential(
            # 3x3 卷积 + 1 像素填充，保持空间尺寸不变
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # 批归一化：稳定训练，加速收敛
            nn.BatchNorm2d(out_channels),
            # ReLU 激活：引入非线性，解决梯度消失问题（正半轴梯度恒为 1）
            nn.ReLU(inplace=True),  # inplace=True 节省内存
            # 2x2 最大池化：空间尺寸减半，提取最显著特征
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数:
            x: 输入张量，形状为 (batch_size, 3, 32, 32)

        返回:
            输出张量，形状为 (batch_size, num_classes)
        """
        # 特征提取阶段
        x = self.block1(x)   # (B, 3, 32, 32) → (B, 32, 16, 16)
        x = self.block2(x)   # (B, 32, 16, 16) → (B, 64, 8, 8)
        x = self.block3(x)   # (B, 64, 8, 8) → (B, 128, 4, 4)

        # 自适应平均池化（此处输入已是 4x4，池化为 4x4 等价于恒等映射）
        x = self.global_pool(x)  # (B, 128, 4, 4)

        # 展平 + 分类
        x = torch.flatten(x, start_dim=1)  # (B, 128*4*4) = (B, 2048)
        x = F.relu(self.fc1(self.dropout(x)))  # Dropout → FC → ReLU
        x = F.relu(self.fc2(self.dropout(x)))  # Dropout → FC → ReLU
        x = self.fc3(x)                          # FC → 输出 logits

        return x
