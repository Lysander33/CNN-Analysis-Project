"""
ResNet-18 模型实现
==================
ResNet（Residual Network）由何恺明等人于 2015 年提出，在 ILSVRC 2015 中获得冠军。
它的核心创新是"跳跃连接"（Skip Connection），也叫"残差连接"（Residual Connection）。

为什么需要跳跃连接？
- 直觉上，更深的网络应该表现更好，但实验发现 56 层网络的训练误差反而高于 20 层
- 原因：梯度在反向传播经过很多层后会逐渐消失（梯度消失问题）
- 跳跃连接提供了一个"高速公路"让梯度直接流过 → 解决了深层网络的退化问题

残差块的数学表达:
    output = F(x) + x
    其中 F(x) 是两层 3×3 卷积，x 是输入的恒等映射
    如果 F(x) 的维度和 x 不匹配，则通过 1×1 卷积调整 x 的维度

本实现是标准的 CIFAR-10 版 ResNet-18（不使用 7×7 大卷积核和初始最大池化）。
"""

from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet 的基本残差块。

    结构: 两个 3×3 卷积（带 BatchNorm 和 ReLU）+ 跳跃连接

    如果输入输出维度不同，跳跃连接会使用 1×1 卷积调整形状。
    下采样通过第一个卷积的 stride=2 实现（而不是池化）。
    """

    # expansion: 基础块的通道扩展倍数
    # BasicBlock 不变，Bottleneck（ResNet-50/101/152 使用）会扩展 4 倍
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Sequential] = None,
    ):
        """初始化 BasicBlock。

        参数:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            stride: 第一个卷积的步长。若为 2 则做空间降采样。
            downsample: 跳跃连接中的降采样模块（1×1 卷积），用于匹配维度。
                       只有在 in_channels != out_channels 或 stride != 1 时需要。
        """
        super(BasicBlock, self).__init__()

        # 第一个卷积层（可能带下采样 stride=2）
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二个卷积层（保持空间尺寸不变）
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接中的维度匹配模块（如果需要的话）
        # 用 1×1 卷积调整通道数和空间尺寸，使其与 F(x) 的输出匹配
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        残差连接的工作流程:
        1. 保存输入 identity = x（跳跃连接的起点）
        2. x → conv1 → bn1 → relu → conv2 → bn2（主路径）
        3. 如果 downsample 存在，identity = downsample(identity)（调维）
        4. output = relu(主路径输出 + identity)（残差相加 + 激活）

        关键: 即使主路径学不到有用的特征（F(x)≈0），
        输出也至少等于 identity → 网络至少不会比浅层网络差！
        """
        identity = x  # 保存跳跃连接的输入

        # 主路径: conv1 → bn1 → relu → conv2 → bn2
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果维度不匹配，调整跳跃连接的维度
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差相加: F(x) + x
        # 这一步是 ResNet 的精髓！
        out += identity
        out = F.relu(out, inplace=True)

        return out


class ResNet18(nn.Module):
    """ResNet-18 卷积神经网络（CIFAR-10 适配版本）。

    与原始 ResNet-18 的区别:
    - 原始版本第一个卷积是 7×7 stride=2，适用于 224×224 的 ImageNet 图片
    - CIFAR-10 图片只有 32×32，7×7 太大了 → 改用 3×3 stride=1
    - 去掉初始最大池化层，保留更多空间信息

    结构概览:
        输入 (3, 32, 32)
        → Conv1 (3→64, 3×3, stride=1) + BN + ReLU
        → Layer1: 2× BasicBlock(64→64)   → (64, 32, 32)
        → Layer2: 2× BasicBlock(64→128)  → (128, 16, 16)  [第一个 block stride=2]
        → Layer3: 2× BasicBlock(128→256) → (256, 8, 8)    [第一个 block stride=2]
        → Layer4: 2× BasicBlock(256→512) → (512, 4, 4)    [第一个 block stride=2]
        → 自适应平均池化 → (512, 1, 1)
        → FC (512→10)
    """

    def __init__(self, num_classes: int = 10):
        """初始化 ResNet-18。

        参数:
            num_classes: 分类类别数，默认为 10。
        """
        super(ResNet18, self).__init__()

        self.in_channels = 64  # 记录当前通道数（用于 _make_layer）

        # ========== Stem 层 ==========
        # CIFAR-10 专用: 3×3 卷积，步长 1，不降采样，不加最大池化
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ========== 四个残差层 ==========
        # 每个 layer 包含多个 BasicBlock
        # 通道数逐渐翻倍（64→128→256→512），空间尺寸逐渐减半
        self.layer1 = self._make_layer(64, 2, stride=1)   # 64→64, 32×32
        self.layer2 = self._make_layer(128, 2, stride=2)  # 64→128, 16×16
        self.layer3 = self._make_layer(256, 2, stride=2)  # 128→256, 8×8
        self.layer4 = self._make_layer(512, 2, stride=2)  # 256→512, 4×4

        # ========== 全局池化 + 分类器 ==========
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """创建由多个残差块组成的一层。

        每层的第一个块可能做降采样（stride=2），其余块保持尺寸不变。

        参数:
            out_channels: 该层的输出通道数。
            num_blocks: 该层包含的残差块数量（ResNet-18 每层 2 个）。
            stride: 第一个残差块的步长。

        返回:
            nn.Sequential: 组合好的多个残差块。
        """
        downsample = None
        # 如果输入输出维度不同（通道数变了或做了降采样），需要跳跃连接适配
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,      # 1×1 卷积：只改变通道数，不改变空间信息
                    stride=stride,       # 匹配主路径的下采样步长
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers: List[nn.Module] = []
        # 第一个块可能需要降采样
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        # 后续块保持尺寸不变
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Kaiming 初始化（适合 ReLU 激活的卷积层）。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数:
            x: 输入张量，形状为 (batch_size, 3, 32, 32)

        返回:
            输出张量，形状为 (batch_size, num_classes)
        """
        # Stem
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)

        # 四个残差层
        x = self.layer1(x)  # (B, 64, 32, 32)
        x = self.layer2(x)  # (B, 128, 16, 16)
        x = self.layer3(x)  # (B, 256, 8, 8)
        x = self.layer4(x)  # (B, 512, 4, 4)

        # 全局池化 + 分类
        x = self.avgpool(x)         # (B, 512, 1, 1)
        x = torch.flatten(x, 1)     # (B, 512)
        x = self.fc(x)              # (B, num_classes)

        return x
