"""
数据加载模块
============
负责 CIFAR-10 数据集的下载、预处理和数据加载器的创建。

CIFAR-10 数据集:
- 10 个类别: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 训练集: 50,000 张 32x32 彩色图片
- 测试集: 10,000 张 32x32 彩色图片
- 在本项目中，训练集会按 9:1 拆分为训练集和验证集
"""

import platform
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

# CIFAR-10 数据集的通道均值和标准差（在完整训练集上计算得出）
# 使用这些值进行标准化可以让模型训练更稳定、收敛更快
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# CIFAR-10 的 10 个类别名称，按顺序排列
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


class TransformSubset(Dataset):
    """为数据集子集应用独立的 transform，不改变原始数据集。

    问题背景:
        random_split 返回的 Subset 共享底层同一个 Dataset 实例。
        直接修改其 transform 会影响所有引用该 Dataset 的子集。
        本类为每个子集提供独立的 transform，互不干扰。

    参数:
        base_dataset: 原始完整数据集（如 CIFAR-10），包含 data 和 targets。
        indices: 该子集在原数据集中的索引列表。
        transform: 应用于该子集的 torchvision 变换。
    """

    def __init__(self, base_dataset: Dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = list(indices)  # 转为 list 确保可索引
        self.transform = transform

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.indices)


def get_cifar10_transforms(train: bool = True) -> transforms.Compose:
    """获取 CIFAR-10 的数据预处理/增强流水线。

    训练时使用数据增强来防止过拟合:
    - 随机裁剪（先填充 4 像素再裁回 32x32）：让模型学会平移不变性
    - 随机水平翻转：让模型学会左右对称性
    - 标准化：将像素值缩放到标准正态分布，加速训练收敛

    验证/测试时只做标准化，不做增强，以保证评估的公平性。

    参数:
        train: True 表示训练模式（包含数据增强），False 表示评估模式。

    返回:
        torchvision.transforms.Compose: 组合后的变换流水线。
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),          # 随机裁剪：增强平移鲁棒性
            transforms.RandomHorizontalFlip(p=0.5),        # 随机水平翻转：增强对称性
            transforms.ToTensor(),                          # 将 PIL 图像转为 Tensor (0-1)
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),  # 标准化到标准正态分布
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])


def get_cifar10_dataloaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_dir: str = "./data",
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建 CIFAR-10 的训练、验证和测试数据加载器。

    首次运行时会自动下载数据集（约 170MB）到 data_dir 目录。
    训练集按 val_split 比例随机拆分为训练集和验证集。

    参数:
        batch_size: 每个批次的样本数。默认 128。
        num_workers: 数据加载的并行进程数。Windows 上自动设为 0。
        data_dir: 数据集存储目录。默认 "./data"。
        val_split: 验证集占训练集的比例。默认 0.1（10%）。

    返回:
        (train_loader, val_loader, test_loader): 三个 DataLoader 元组。
    """
    # Windows 系统上多进程数据加载可能有问题，自动降级为单进程
    if platform.system() == "Windows":
        num_workers = 0
        print("[信息] 检测到 Windows 系统，已将 num_workers 设为 0")

    # 获取训练和测试的变换流水线
    train_transform = get_cifar10_transforms(train=True)
    test_transform = get_cifar10_transforms(train=False)

    # 下载并加载数据集（首次运行会自动下载约 170MB）
    # 优化：训练集只加载一次（不带 transform），通过 TransformSubset 分别应用变换
    # 这样避免了之前两次加载 CIFAR-10 的冗余 I/O 操作
    print("[信息] 正在加载 CIFAR-10 数据集（首次运行会自动下载约 170MB）...")
    full_train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=None,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform,
    )

    # 将训练集拆分为训练集和验证集（固定随机种子保证可复现）
    val_size = int(len(full_train_set) * val_split)
    train_size = len(full_train_set) - val_size
    generator = torch.Generator().manual_seed(42)  # 固定种子，保证每次拆分一致

    train_subset, val_subset = random_split(
        range(len(full_train_set)), [train_size, val_size],
        generator=generator,
    )

    # 为训练集和验证集分别包装独立的 transform（互不影响）
    train_set = TransformSubset(full_train_set, train_subset.indices, train_transform)
    val_set = TransformSubset(full_train_set, val_subset.indices, test_transform)

    print(f"[信息] 数据集大小 - 训练集: {train_size}, 验证集: {val_size}, 测试集: {len(test_set)}")

    # pin_memory 只在 CUDA 可用时启用（CPU 上启用会触发警告）
    use_pin_memory = torch.cuda.is_available()

    # 创建数据加载器（共享参数避免重复）
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
