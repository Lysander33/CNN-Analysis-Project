"""
CNN 分析项目 —— 主入口
======================
本脚本是项目的命令行入口，支持两种运行模式:

1. 单模型训练模式:
   python main.py --model lenet
   python main.py --model resnet --epochs 30 --batch_size 64

2. 全模型对比模式:
   python main.py --compare

在对比模式下，会依次训练所有 CNN 模型，
然后自动生成 5 张对比分析图表和一份控制台总结报告。

实验目的:
    在相同的数据集、相同的训练超参数下，对比不同 CNN 架构
    （LeNet-5、SimpleCNN、VGG-11、ResNet-18）在 CIFAR-10 上的表现，
    分析架构设计对准确率、收敛速度、参数量和训练时间的影响。
"""

import argparse
import json
import os
import platform
import random
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

# 导入项目模块
from models import get_model, list_models
from utils.data_loader import CIFAR10_CLASSES, get_cifar10_dataloaders
from utils.train_eval import Trainer
from utils.plot_utils import (
    MODEL_DISPLAY_NAMES,
    plot_confusion_matrices,
    plot_inference_speed,
    plot_model_comparison_bar,
    plot_per_class_accuracy,
    plot_training_curves,
)


def set_seed(seed: int = 42) -> None:
    """设置全局随机种子，确保实验可复现。

    影响范围:
    - Python 标准库的 random 模块
    - NumPy 随机数生成器
    - PyTorch CPU/GPU 随机数生成器
    - CUDA 后端设为确定性模式（可能略微降低性能）

    参数:
        seed: 随机种子值，默认 42。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[可复现性] 随机种子已设为 {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """从 YAML 文件加载模型配置。

    参数:
        config_path: YAML 配置文件的路径。

    返回:
        配置字典。

    异常:
        FileNotFoundError: 配置文件不存在时抛出。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """用命令行参数覆盖配置文件中的值。

    这样可以在不修改 YAML 文件的情况下快速调整超参数。

    参数:
        config: 原始配置字典。
        args: 解析后的命令行参数。

    返回:
        修改后的配置字典。
    """
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    return config


def train_single_model(
    model_name: str,
    config: Dict[str, Any],
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """训练单个 CNN 模型并返回结果。

    完整的训练流程:
    1. 加载数据
    2. 构建模型
    3. 训练 + 验证
    4. 测试评估
    5. 收集指标

    参数:
        model_name: 模型名称（如 "lenet"）。
        config: 配置字典。
        device: 训练设备。

    返回:
        结果字典，包含 test_acc, num_params, training_time 等指标。
        如果出错则返回 None。
    """
    try:
        # ========== 1. 加载数据 ==========
        train_loader, val_loader, test_loader = get_cifar10_dataloaders(
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            data_dir=config["data"]["data_dir"],
            val_split=config["data"]["val_split"],
        )

        # ========== 2. 构建模型 ==========
        model_cfg = config["model"].copy()
        model_cfg.pop("name", None)  # 移除 name 字段，剩余作为 kwargs
        model = get_model(model_name, **model_cfg)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[模型信息] {MODEL_DISPLAY_NAMES.get(model_name, model_name)}")
        print(f"[参数量] {num_params:,}")

        # ========== 3. 训练 ==========
        trainer = Trainer(model, device, config)
        train_start = time.time()
        trainer.fit(train_loader, val_loader)
        training_time = time.time() - train_start

        # ========== 4. 测试与评估 ==========
        metrics = trainer.evaluate_metrics(test_loader)

        result = {
            **metrics,
            "num_params": num_params,
            "training_time": training_time,
            "history": trainer.history,
            "best_val_acc": trainer.best_val_acc,
        }

        return result

    except Exception as e:
        print(f"[错误] 训练 {model_name} 时发生异常: {e}")
        traceback.print_exc()
        return None


def run_comparison(
    models_to_train: List[str],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """运行全模型对比实验。

    依次训练所有模型，收集结果，生成对比图表和总结报告。

    参数:
        models_to_train: 要训练的模型名称列表。
        args: 命令行参数。
        device: 训练设备。
    """
    all_results: Dict[str, dict] = {}
    all_histories: Dict[str, dict] = {}

    # 从首个可用的配置文件读取 output 路径（不依赖训练成功的顺序）
    output_config = None
    for model_name in models_to_train:
        cfg_path = os.path.join(args.config_dir, f"{model_name}.yaml")
        if os.path.exists(cfg_path):
            output_config = load_config(cfg_path)["output"]
            break
    if output_config is None:
        print("[错误] 未找到任何配置文件，退出。")
        return

    total_start = time.time()

    print("\n" + "=" * 70)
    print("CNN 架构对比分析实验")
    print("=" * 70)
    print(f"[设备] {device}")
    print(f"[模型列表] {[MODEL_DISPLAY_NAMES.get(m, m) for m in models_to_train]}")
    print("=" * 70)

    for idx, model_name in enumerate(models_to_train, 1):
        print(f"\n{'#' * 70}")
        print(f"# [{idx}/{len(models_to_train)}] 训练模型: {MODEL_DISPLAY_NAMES.get(model_name, model_name)}")
        print(f"{'#' * 70}")

        # 加载配置
        config_path = os.path.join(args.config_dir, f"{model_name}.yaml")
        if not os.path.exists(config_path):
            print(f"[警告] 未找到配置文件 {config_path}，跳过 {model_name}")
            continue
        config = load_config(config_path)
        config = apply_cli_overrides(config, args)

        # 训练
        result = train_single_model(model_name, config, device)
        if result is not None:
            all_results[model_name] = result
            all_histories[model_name] = result["history"]

    if not all_results:
        print("[错误] 没有任何模型训练成功，退出。")
        return

    # ========== 保存结果 ==========
    summary_path = os.path.join(output_config["log_dir"], "comparison_summary.json")
    # 只保存可序列化的字段
    serializable = {}
    for name, res in all_results.items():
        serializable[name] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in res.items()
            if k not in ("history", "confusion_matrix", "per_class_acc")
        }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n[结果摘要已保存] {summary_path}")

    # ========== 生成对比图表 ==========
    plot_dir = output_config["plot_dir"]
    print(f"\n[生成对比图表...]")

    # 1. 训练曲线
    plot_training_curves(
        all_histories,
        os.path.join(plot_dir, "training_curves.png"),
    )

    # 2. 模型对比柱状图
    plot_model_comparison_bar(
        all_results,
        os.path.join(plot_dir, "model_comparison.png"),
    )

    # 3. 混淆矩阵
    all_cms = {name: res["confusion_matrix"] for name, res in all_results.items()}
    plot_confusion_matrices(
        all_cms,
        list(CIFAR10_CLASSES),
        os.path.join(plot_dir, "confusion_matrices.png"),
    )

    # 4. 各类别准确率
    all_per_class = {name: res["per_class_acc"] for name, res in all_results.items()}
    plot_per_class_accuracy(
        all_per_class,
        list(CIFAR10_CLASSES),
        os.path.join(plot_dir, "per_class_accuracy.png"),
    )

    # 5. 推理速度
    inference_speeds = {name: res["inference_speed"] for name, res in all_results.items()}
    plot_inference_speed(
        inference_speeds,
        os.path.join(plot_dir, "inference_speed.png"),
    )

    # ========== 控制台总结报告 ==========
    total_time = time.time() - total_start
    print(f"\n{'=' * 80}")
    print("CNN 架构对比分析 —— 总结报告")
    print(f"{'=' * 80}")
    print(f"{'模型':<20} {'测试准确率':>10} {'参数量':>12} {'训练时间':>12} {'推理速度':>14}")
    print("-" * 80)
    for model_name in models_to_train:
        if model_name in all_results:
            res = all_results[model_name]
            display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            print(
                f"{display:<20} "
                f"{res['test_acc']:>8.2f}% "
                f"{res['num_params']:>10,} "
                f"{res['training_time']:>8.0f}秒 "
                f"{res['inference_speed']:>10.0f} 样本/秒"
            )
    print("-" * 80)
    print(f"总实验耗时: {total_time:.0f}秒 ({total_time/60:.1f}分钟)")
    print(f"图表保存位置: {os.path.abspath(plot_dir)}")
    print(f"{'=' * 80}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="CNN 架构对比分析 —— 训练不同 CNN 模型并分析它们在 CIFAR-10 上的表现差异",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --model lenet                  # 训练单个 LeNet-5 模型
  python main.py --model resnet --epochs 30     # 训练 ResNet-18，30 个 epoch
  python main.py --compare                      # 训练所有模型并生成对比报告
  python main.py --compare --epochs 20 --lr 0.001  # 快速对比实验
  python main.py --resume results/checkpoints/lenet_final.pth   # 从检查点恢复训练
  python main.py --resume results/checkpoints/lenet_final.pth --epochs 50  # 恢复并延长训练
        """,
    )

    # 运行模式
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--model", "-m", type=str, default=None,
        choices=list_models(),
        help="要训练的单个模型名称",
    )
    mode_group.add_argument(
        "--compare", "-c", action="store_true",
        help="训练所有模型并生成对比分析报告",
    )

    # 超参数覆盖
    parser.add_argument("--epochs", "-e", type=int, default=None,
                        help="训练 epoch 数（覆盖配置文件中的值）")
    parser.add_argument("--batch_size", "-b", type=int, default=None,
                        help="批次大小（覆盖配置文件中的值）")
    parser.add_argument("--lr", type=float, default=None,
                        help="初始学习率（覆盖配置文件中的值）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，确保实验可复现（默认: 42）")

    # 断点续训
    parser.add_argument("--resume", type=str, default=None, metavar="PATH",
                        help="从指定检查点恢复训练（仅在单模型模式下有效）")

    # 系统选项
    parser.add_argument("--device", "-d", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="训练设备（默认: auto，自动选择可用的最佳设备）")
    parser.add_argument("--config_dir", type=str, default="./configs",
                        help="配置文件目录路径（默认: ./configs）")

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """根据参数获取训练设备。

    参数:
        device_arg: "auto"、"cpu" 或 "cuda"。

    返回:
        torch.device 对象。
    """
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("[警告] CUDA 不可用，将使用 CPU")
            return torch.device("cpu")
    elif device_arg == "cpu":
        return torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            print("[信息] 检测到 CUDA，将使用 GPU 训练")
            return torch.device("cuda")
        else:
            print("[信息] 未检测到 CUDA，将使用 CPU 训练")
            return torch.device("cpu")


def main() -> None:
    """主函数 —— 程序入口。"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # --resume 只能在单模型模式下使用
    if args.resume and args.compare:
        print("[错误] --resume 和 --compare 不能同时使用。"
              "断点续训仅支持单模型模式。")
        sys.exit(1)

    # 确定训练设备
    device = get_device(args.device)
    if platform.system() == "Windows":
        print(f"[系统] Windows 检测到，数据加载将使用单进程模式")

    # 确定要训练的模型列表
    if args.compare:
        models_to_train = list_models()
        print(f"\n[对比模式] 将依次训练所有 {len(models_to_train)} 个模型")
    elif args.model:
        models_to_train = [args.model]
    else:
        # 默认：训练 LeNet（最快的模型，适合快速测试）
        print("[提示] 未指定模型，默认训练 LeNet-5。使用 --help 查看更多选项。")
        models_to_train = ["lenet"]

    # 单模型模式：走简化流程
    if len(models_to_train) == 1 and not args.compare:
        model_name = models_to_train[0]

        # ========== 断点续训分支 ==========
        if args.resume:
            if not os.path.exists(args.resume):
                print(f"[错误] 检查点文件不存在: {args.resume}")
                sys.exit(1)

            # 1. 加载检查点，提取配置和模型名
            # weights_only=False：resume 需要读取 checkpoint 中的 config 等元数据
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            saved_config = checkpoint.get("config")
            if saved_config is None:
                print("[错误] 检查点中未找到配置信息，无法恢复训练。")
                sys.exit(1)

            saved_model_name = saved_config["model"]["name"]
            print(f"[断点续训] 从检查点恢复训练: {args.resume}")
            print(f"[模型] {saved_model_name}")

            # 2. 用 CLI 参数覆盖原配置
            config = apply_cli_overrides(saved_config, args)
            model_cfg = config["model"].copy()
            model_cfg.pop("name", None)

            # 3. 创建模型和训练器
            model = get_model(saved_model_name, **model_cfg)
            trainer = Trainer(model, device, config)

            # 4. 恢复训练状态，获取上次训练的 epoch
            saved_epoch = trainer.load_checkpoint(args.resume)
            print(f"[恢复] 将从 Epoch {saved_epoch + 1} 继续训练")

            # 5. 加载数据
            train_loader, val_loader, test_loader = get_cifar10_dataloaders(
                batch_size=config["training"]["batch_size"],
                num_workers=config["data"]["num_workers"],
                data_dir=config["data"]["data_dir"],
                val_split=config["data"]["val_split"],
            )

            # 6. 继续训练
            trainer.fit(train_loader, val_loader, start_epoch=saved_epoch + 1)

            # 7. 测试
            test_loss, test_acc, predictions, targets = trainer.test(test_loader)
            print(f"\n训练完成！测试准确率: {test_acc:.2f}%")
            return

        # ========== 正常单模型训练分支 ==========
        config_path = os.path.join(args.config_dir, f"{model_name}.yaml")

        if not os.path.exists(config_path):
            print(f"[错误] 找不到配置文件: {config_path}")
            sys.exit(1)

        config = load_config(config_path)
        config = apply_cli_overrides(config, args)

        result = train_single_model(model_name, config, device)

        if result is not None:
            print(f"\n训练完成！测试准确率: {result['test_acc']:.2f}%")
        else:
            print("\n[错误] 训练失败。")
            sys.exit(1)
    else:
        # 多模型对比模式
        run_comparison(models_to_train, args, device)

    print("\n程序执行完毕。")


if __name__ == "__main__":
    main()
