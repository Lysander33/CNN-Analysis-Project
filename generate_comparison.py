"""从已有 checkpoint 生成四模型对比图表和报告（不重新训练）"""
import json
import os
from typing import Dict

import torch

from models import get_model
from utils.data_loader import CIFAR10_CLASSES, get_cifar10_dataloaders
from utils.plot_utils import (
    MODEL_DISPLAY_NAMES,
    plot_confusion_matrices,
    plot_inference_speed,
    plot_model_comparison_bar,
    plot_per_class_accuracy,
    plot_training_curves,
)


def load_history(model_name: str) -> dict:
    """加载训练历史 JSON"""
    path = f"results/logs/{model_name}_history.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_model(model_name: str, device: torch.device):
    """从 best checkpoint 加载模型并评估，复用 Trainer.test() 和 evaluate_metrics()。"""
    import yaml

    from utils.train_eval import Trainer

    # 加载配置
    with open(f"configs/{model_name}.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 构建模型
    model_cfg = config["model"].copy()
    model_cfg.pop("name", None)
    model = get_model(model_name, **model_cfg)

    # 加载最佳权重（新格式仅存 state_dict；旧格式完整 dict 自动回退）
    best_path = f"results/checkpoints/{model_name}_best.pth"
    try:
        state_dict = torch.load(best_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"]
        print(f"[警告] {model_name} 使用旧格式 checkpoint，请重新训练")
    model.load_state_dict(state_dict)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 从 history JSON 读取最佳验证准确率
    history = load_history(model_name)
    print(f"\n{'='*60}")
    print(f"[评估] {MODEL_DISPLAY_NAMES.get(model_name, model_name)}")
    print(f"[参数量] {num_params:,}")
    print(f"[最佳验证准确率] {history.get('best_val_acc', 0):.2f}%")

    # 加载数据
    _, _, test_loader = get_cifar10_dataloaders(
        batch_size=config["training"]["batch_size"],
        num_workers=0,
        data_dir=config["data"]["data_dir"],
        val_split=config["data"]["val_split"],
    )

    # 复用 Trainer 的 evaluate_metrics（消除重复的测试循环和指标计算）
    trainer = Trainer(model, device, config)
    metrics = trainer.evaluate_metrics(test_loader)

    return {
        **metrics,
        "num_params": num_params,
        "training_time": 0,  # 由调用方根据 history 推算
        "history": history,
        "best_val_acc": history.get("best_val_acc", 0),
    }


def main():
    device = torch.device("cpu")
    print(f"[设备] {device}")

    model_names = ["lenet", "simple_cnn", "vgg", "resnet"]

    all_results: Dict[str, dict] = {}
    all_histories: Dict[str, dict] = {}

    for model_name in model_names:
        result = evaluate_model(model_name, device)
        all_results[model_name] = result
        all_histories[model_name] = result["history"]

    # 从 history JSON 获取实际 epoch 数，结合参数量推算训练时间
    # 基准：LeNet (~62K params) 在 CPU 上约 120 秒/epoch
    for name, res in all_results.items():
        epochs = len(all_histories[name].get("train_loss", []))
        params = res["num_params"]
        per_epoch_est = max(60, 120 * (params / 62000))
        res["training_time"] = epochs * per_epoch_est
        res["training_time_estimated"] = True

    # ========== 保存汇总 ==========
    output_dir = "results/plots"
    log_dir = "results/logs"
    os.makedirs(output_dir, exist_ok=True)

    # 保存 comparison_summary.json
    serializable = {}
    for name, res in all_results.items():
        serializable[name] = {
            "test_acc": res["test_acc"],
            "test_loss": res["test_loss"],
            "num_params": res["num_params"],
            "training_time": res["training_time"],
            "inference_speed": res["inference_speed"],
            "best_val_acc": res["best_val_acc"],
        }
    with open(f"{log_dir}/comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n[汇总已保存] {log_dir}/comparison_summary.json")

    # ========== 生成图表 ==========
    print(f"\n[生成对比图表...]")

    # 1. 训练曲线
    plot_training_curves(all_histories, f"{output_dir}/training_curves.png")

    # 2. 模型对比柱状图
    plot_model_comparison_bar(all_results, f"{output_dir}/model_comparison.png")

    # 3. 混淆矩阵
    all_cms = {name: res["confusion_matrix"] for name, res in all_results.items()}
    plot_confusion_matrices(all_cms, list(CIFAR10_CLASSES), f"{output_dir}/confusion_matrices.png")

    # 4. 各类别准确率
    all_per_class = {name: res["per_class_acc"] for name, res in all_results.items()}
    plot_per_class_accuracy(all_per_class, list(CIFAR10_CLASSES), f"{output_dir}/per_class_accuracy.png")

    # 5. 推理速度
    inference_speeds = {name: res["inference_speed"] for name, res in all_results.items()}
    plot_inference_speed(inference_speeds, f"{output_dir}/inference_speed.png")

    # ========== 控制台报告 ==========
    print(f"\n{'='*80}")
    epoch_count = {name: len(h.get("train_loss", [])) for name, h in all_histories.items()}
    print("CNN 架构对比分析 —— 总结报告")
    print(f"{'='*80}")
    print(f"{'模型':<20} {'测试准确率':>10} {'最佳验证':>10} {'参数量':>12} {'训练时间(估)':>10} {'推理速度':>14}")
    print("-" * 80)
    for model_name in model_names:
        res = all_results[model_name]
        display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        t = res["training_time"]
        time_str = f"{t/60:.0f}分钟" if t >= 60 else f"{t:.0f}秒"
        print(
            f"{display:<20} "
            f"{res['test_acc']:>8.2f}% "
            f"{res['best_val_acc']:>8.2f}% "
            f"{res['num_params']:>10,} "
            f"{time_str:>10} "
            f"{res['inference_speed']:>10.0f} 样本/秒"
        )
    print("-" * 80)
    print(f"※ 训练时间为基于参数量推算的估算值（epoch数从history获取，各模型: "
          f"{', '.join(f'{MODEL_DISPLAY_NAMES.get(n, n)}={epoch_count[n]}epoch' for n in model_names)}）")
    print(f"图表保存位置: {os.path.abspath(output_dir)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
