"""
可视化工具模块
==============
提供一系列绘图函数，用于对比分析不同 CNN 模型在 CIFAR-10 上的表现。

生成的图表包括:
1. 训练曲线对比图 —— 各模型的 loss 和 accuracy 随 epoch 的变化
2. 模型性能对比柱状图 —— 测试准确率、参数量、训练时间
3. 混淆矩阵 —— 每个模型在各个类别上的分类表现
4. 各类别准确率 —— 不同模型在每种物体上的识别能力差异
5. 推理速度对比 —— 衡量模型的部署效率

所有图表均保存到 results/plots/ 目录。
"""

import os
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 使用中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 为每个模型分配一致的颜色，便于在多个图表中对照
MODEL_COLORS: Dict[str, str] = {
    "lenet": "#3498db",       # 蓝色
    "simple_cnn": "#2ecc71",  # 绿色
    "vgg": "#e74c3c",         # 红色
    "resnet": "#9b59b6",      # 紫色
}

# 模型显示名称（用于图表标签）
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "lenet": "LeNet-5",
    "simple_cnn": "SimpleCNN",
    "vgg": "VGG-11",
    "resnet": "ResNet-18",
}


def _ensure_dir(path: str) -> None:
    """确保输出目录存在。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_training_curves(
    all_histories: Dict[str, dict],
    save_path: str,
) -> None:
    """绘制所有模型的训练曲线对比图。

    包含两个子图:
    - 上图: 训练损失和验证损失曲线
    - 下图: 训练准确率和验证准确率曲线

    通过对比训练曲线和验证曲线，可以判断:
    - 模型是否欠拟合（训练和验证指标都差）
    - 模型是否过拟合（训练指标好但验证指标差，两条曲线分叉）
    - 模型是否已经收敛（曲线趋于平稳）

    参数:
        all_histories: 字典，键为模型名称，值为训练历史字典。
        save_path: 图片保存路径。
    """
    _ensure_dir(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- 左图: 损失曲线 ----
    ax = axes[0]
    for model_name, history in all_histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        color = MODEL_COLORS.get(model_name, "#333333")
        label = MODEL_DISPLAY_NAMES.get(model_name, model_name)

        # 训练损失（实线）
        ax.plot(epochs, history["train_loss"], color=color, linewidth=1.5,
                label=f"{label} (训练)")
        # 验证损失（虚线）
        ax.plot(epochs, history["val_loss"], color=color, linewidth=1.5,
                linestyle="--", label=f"{label} (验证)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("损失 (Loss)")
    ax.set_title("训练与验证损失曲线")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ---- 右图: 准确率曲线 ----
    ax = axes[1]
    for model_name, history in all_histories.items():
        epochs = range(1, len(history["train_acc"]) + 1)
        color = MODEL_COLORS.get(model_name, "#333333")
        label = MODEL_DISPLAY_NAMES.get(model_name, model_name)

        ax.plot(epochs, history["train_acc"], color=color, linewidth=1.5,
                label=f"{label} (训练)")
        ax.plot(epochs, history["val_acc"], color=color, linewidth=1.5,
                linestyle="--", label=f"{label} (验证)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("准确率 (%)")
    ax.set_title("训练与验证准确率曲线")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表已保存] {save_path}")


def plot_model_comparison_bar(
    results: Dict[str, dict],
    save_path: str,
) -> None:
    """绘制模型关键指标对比柱状图。

    三个对比维度:
    1. 测试准确率 —— 越高越好，衡量模型的分类能力
    2. 参数量 —— 模型复杂度，影响内存占用和过拟合风险
    3. 训练时间 —— 计算代价，衡量训练效率

    参数:
        results: 字典，键为模型名称，值包含 test_acc, num_params, training_time。
        save_path: 图片保存路径。
    """
    _ensure_dir(save_path)
    model_names = list(results.keys())
    display_names = [MODEL_DISPLAY_NAMES.get(n, n) for n in model_names]
    colors = [MODEL_COLORS.get(n, "#333333") for n in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 测试准确率 ----
    ax = axes[0]
    accs = [results[n]["test_acc"] for n in model_names]
    bars = ax.bar(display_names, accs, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("测试准确率 (%)")
    ax.set_title("测试准确率对比")
    ax.set_ylim(0, max(accs) * 1.2)
    # 在柱子上标注数值
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # ---- 参数量（对数尺度） ----
    ax = axes[1]
    params = [results[n]["num_params"] for n in model_names]
    bars = ax.bar(display_names, params, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("参数量（对数尺度）")
    ax.set_title("模型参数量对比")
    ax.set_yscale("log")  # 对数尺度：因为参数量可能跨几个数量级
    # 在柱子上标注数值
    for bar, p in zip(bars, params):
        if p >= 1_000_000:
            label = f"{p/1_000_000:.1f}M"
        elif p >= 1_000:
            label = f"{p/1_000:.0f}K"
        else:
            label = str(p)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # ---- 训练时间 ----
    ax = axes[2]
    times = [results[n]["training_time"] for n in model_names]
    bars = ax.bar(display_names, times, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("训练时间 (秒)")
    ax.set_title("训练时间对比")
    # 在柱子上标注数值
    for bar, t in zip(bars, times):
        if t >= 60:
            label = f"{t/60:.1f}分钟"
        else:
            label = f"{t:.0f}秒"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                label, ha="center", va="bottom", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表已保存] {save_path}")


def plot_confusion_matrices(
    all_cms: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: str,
) -> None:
    """绘制所有模型的混淆矩阵。

    混淆矩阵的每一行表示真实类别，每一列表示预测类别。
    对角线上的值表示正确分类的比例。
    使用热力图可以直观地看出哪些类别容易混淆。

    参数:
        all_cms: 字典，键为模型名称，值为混淆矩阵（已按行归一化）。
        class_names: 类别名称列表。
        save_path: 图片保存路径。
    """
    _ensure_dir(save_path)
    n_models = len(all_cms)
    model_names = list(all_cms.keys())
    display_names = [MODEL_DISPLAY_NAMES.get(n, n) for n in model_names]

    # 根据模型数量动态调整子图布局
    if n_models <= 2:
        nrows, ncols = 1, n_models
    else:
        nrows, ncols = 2, 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        cm = all_cms[model_name]

        # 使用 seaborn 热力图
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            vmin=0, vmax=1, ax=ax, cbar=(idx == n_models - 1),
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(f"{display_names[idx]} 混淆矩阵", fontsize=12, fontweight="bold")
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")

    # 隐藏多余的子图
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表已保存] {save_path}")


def plot_per_class_accuracy(
    all_per_class: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: str,
) -> None:
    """绘制每个模型在各个类别上的准确率对比。

    这张图可以帮助回答:
    - 哪些类别普遍比较难识别？（如猫、鸟、鹿等外观相似的动物）
    - 不同架构在不同类别上是否有各自擅长的？
    - 更深/更宽的模型是否在所有类别上都有提升？

    参数:
        all_per_class: 字典，键为模型名称，值为各类别准确率数组。
        class_names: 类别名称列表。
        save_path: 图片保存路径。
    """
    _ensure_dir(save_path)
    n_classes = len(class_names)
    n_models = len(all_per_class)
    model_names = list(all_per_class.keys())
    display_names = [MODEL_DISPLAY_NAMES.get(n, n) for n in model_names]

    # 分组柱状图
    x = np.arange(n_classes)          # 类别位置
    width = 0.8 / n_models             # 每组柱子的宽度
    offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, n_models)

    fig, ax = plt.subplots(figsize=(16, 6))

    for idx, model_name in enumerate(model_names):
        color = MODEL_COLORS.get(model_name, "#333333")
        label = display_names[idx]
        offset = offsets[idx]
        accs = all_per_class[model_name]
        bars = ax.bar(x + offset, accs, width, color=color, label=label,
                      edgecolor="white", linewidth=0.8)

    ax.set_xlabel("类别")
    ax.set_ylabel("准确率 (%)")
    ax.set_title("各模型在不同类别上的准确率对比")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表已保存] {save_path}")


def plot_inference_speed(
    inference_speeds: Dict[str, float],
    save_path: str,
) -> None:
    """绘制各模型的推理速度对比。

    推理速度（samples/second）对于部署场景非常重要:
    移动端或实时应用需要轻量级、快速的模型。

    参数:
        inference_speeds: 字典，键为模型名称，值为每秒处理的样本数。
        save_path: 图片保存路径。
    """
    _ensure_dir(save_path)
    model_names = list(inference_speeds.keys())
    display_names = [MODEL_DISPLAY_NAMES.get(n, n) for n in model_names]
    colors = [MODEL_COLORS.get(n, "#333333") for n in model_names]
    speeds = [inference_speeds[n] for n in model_names]

    fig, ax = plt.subplots(figsize=(10, 5))

    # 水平柱状图
    bars = ax.barh(display_names, speeds, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_xlabel("推理速度 (样本/秒)")
    ax.set_title("模型推理速度对比（越高越好）")

    # 在柱子末尾标注数值
    for bar, speed in zip(bars, speeds):
        ax.text(bar.get_width() + max(speeds) * 0.01, bar.get_y() + bar.get_height()/2,
                f"{speed:.0f} 样本/秒", va="center", fontsize=10, fontweight="bold")

    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表已保存] {save_path}")
