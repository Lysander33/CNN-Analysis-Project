"""
模型注册中心
============
提供模型工厂函数，根据名称字符串动态创建对应的 CNN 模型实例。

使用方式:
    from models import get_model, list_models

    # 查看所有可用模型
    print(list_models())  # ['lenet', 'simple_cnn', 'vgg', 'resnet']

    # 创建模型实例
    model = get_model('lenet', num_classes=10)

    # 带额外参数的模型
    model = get_model('simple_cnn', num_classes=10, dropout=0.3)

添加新模型的步骤:
    1. 在 models/ 目录下创建新的模型文件（如 your_model.py）
    2. 在本文件中导入并注册到 MODEL_REGISTRY 字典
    3. 在 configs/ 目录下创建对应的 YAML 配置文件
"""

from typing import Any, Dict, List

import torch.nn as nn

from models.lenet import LeNet5
from models.simple_cnn import SimpleCNN
from models.vgg import VGG11
from models.resnet import ResNet18

# 模型注册表: 将名称字符串映射到对应的模型类
# 使用字典的好处: 可以通过配置文件中的字符串名称动态选择模型
MODEL_REGISTRY: Dict[str, type] = {
    "lenet": LeNet5,
    "simple_cnn": SimpleCNN,
    "vgg": VGG11,
    "resnet": ResNet18,
}


def get_model(model_name: str, **kwargs: Any) -> nn.Module:
    """模型工厂函数 —— 根据名称创建模型实例。

    这是项目中创建模型的主要入口。所有模型都通过此函数创建，
    确保了一致的创建流程和错误处理。

    参数:
        model_name: 模型名称，必须是 MODEL_REGISTRY 中的键。
        **kwargs: 传递给模型构造函数的额外参数（如 num_classes, dropout）。

    返回:
        nn.Module: 实例化的模型对象。

    异常:
        ValueError: 当 model_name 不在注册表中时抛出。
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"未知的模型名称: '{model_name}'。"
            f"当前支持的模型: {available}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)


def list_models() -> List[str]:
    """列出所有已注册的模型名称。

    返回:
        List[str]: 模型名称列表，按注册顺序排列。
    """
    return list(MODEL_REGISTRY.keys())
