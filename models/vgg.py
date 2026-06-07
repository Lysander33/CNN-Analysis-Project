"""
VGG-11 模型实现
===============
VGG（Visual Geometry Group）由牛津大学于 2014 年提出，位列 ILSVRC 2014 第二名。
其核心创新在于用多个堆叠的 3×3 小卷积核代替大卷积核（如 5×5、7×7）。

核心思想（感受野分析）:
- 两个堆叠的 3×3 卷积 = 一个 5×5 卷积的感受野（但参数量更少）
- 三个堆叠的 3×3 卷积 = 一个 7×7 卷积的感受野
- 同时，多个卷积层之间有更多的非线性激活（ReLU），增强了网络表达能力

VGG-11 的"11"表示共有 11 个带权重的层（8 个卷积层 + 3 个全连接层）。

注意事项:
- VGG 参数量很大（~9.2M），在小数据集（CIFAR-10 只有 50K 训练样本）上容易过拟合
- 这是本次实验的教学重点之一：对比深而宽的网络（VGG）vs 有跳跃连接的网络（ResNet）
"""

from typing import List, Union

import torch
import torch.nn as nn


# VGG-11 的卷积层配置
# 列表中的每个数字代表一个卷积层的输出通道数
# 'M' 代表一个最大池化层
VGG11_CONFIG = [
    64, 'M',           # Block 1: 1 conv → pool
    128, 'M',          # Block 2: 1 conv → pool
    256, 256, 'M',     # Block 3: 2 convs → pool
    512, 512, 'M',     # Block 4: 2 convs → pool
    512, 512, 'M',     # Block 5: 2 convs → pool
]


class VGG11(nn.Module):
    """VGG-11 卷积神经网络（适配 CIFAR-10 版本）。

    与原始 VGG-11 的区别:
    - 原始 VGG 输入为 224×224，使用 5 次池化将分辨率降到 7×7
    - CIFAR-10 输入为 32×32，5 次池化后将分辨率降到 1×1
    - 因此分类器输入维度相应调整

    结构概览:
        输入 (3, 32, 32)
        → Conv Block 1 (3→64) + Pool → (64, 16, 16)
        → Conv Block 2 (64→128) + Pool → (128, 8, 8)
        → Conv Block 3 (128→256) ×2 + Pool → (256, 4, 4)
        → Conv Block 4 (256→512) ×2 + Pool → (512, 2, 2)
        → Conv Block 5 (512→512) ×2 + Pool → (512, 1, 1)
        → FC (512→512) → ReLU → Dropout
        → FC (512→512) → ReLU → Dropout
        → FC (512→10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        """初始化 VGG-11。

        参数:
            num_classes: 分类类别数，默认为 10。
            dropout: 全连接层的 Dropout 比率，默认为 0.5。
                     VGG 使用较高的 dropout 来对抗其巨大的参数量带来的过拟合。
        """
        super(VGG11, self).__init__()

        # 根据配置创建卷积层
        self.features = self._make_layers(VGG11_CONFIG)

        # 自适应平均池化：将特征图统一池化为 1×1
        # 这样无论输入特征图多大，都输出 (512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ========== 分类器 ==========
        # 经过全局池化后，特征维度为 512 × 1 × 1 = 512
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),      # 第一个全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),      # Dropout 防止过拟合
            nn.Linear(512, 512),      # 第二个全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),  # 输出层
        )

        # 初始化权重（使用何恺明初始化，适合 ReLU 激活）
        self._initialize_weights()

    @staticmethod
    def _make_layers(config: List[Union[int, str]]) -> nn.Sequential:
        """根据配置列表构建卷积层。

        VGG 的设计模式:
        - 所有卷积核都是 3×3，padding=1（保持空间尺寸不变）
        - 只有池化层（'M'）才做降采样
        - 这种规律性使得网络结构非常整洁、易于理解

        参数:
            config: VGG 配置列表，整数表示卷积输出通道，'M' 表示最大池化。

        返回:
            组合好的 Sequential 模块。
        """
        layers: List[nn.Module] = []
        in_channels = 3  # RGB 输入

        for item in config:
            if item == 'M':
                # 添加最大池化层，2×2，步长 2 → 空间尺寸减半
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # 添加卷积层：3×3 卷积 + 批归一化 + ReLU
                out_channels = item
                layers.append(nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                ))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """初始化模型权重。

        卷积层使用 Kaiming 初始化（适合 ReLU），批归一化层使用标准初始化，
        全连接层使用 Xavier 初始化。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming/He 初始化：专门为 ReLU 激活设计
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数:
            x: 输入张量，形状为 (batch_size, 3, 32, 32)

        返回:
            输出张量，形状为 (batch_size, num_classes)
        """
        x = self.features(x)   # 卷积特征提取
        x = self.avgpool(x)    # 全局平均池化 → (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # 展平 → (B, 512)
        x = self.classifier(x) # 全连接分类器
        return x
