"""
训练与评估模块
==============
提供 Trainer 类，封装了 CNN 模型训练的完整生命周期：
- 训练循环（含进度条）
- 验证循环
- 学习率调度
- 模型保存与恢复
- 训练历史记录

设计原则:
- 将所有训练逻辑封装在一个类中，使 main.py 保持简洁
- 所有模型使用相同的 Trainer，保证对比实验的一致性
- 支持可复现的训练流程（固定种子、保存完整状态）
"""

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """CNN 模型训练器。

    封装了完整的训练流程:
    1. 训练一个 epoch：遍历数据集，前向传播 → 计算损失 → 反向传播 → 更新参数
    2. 验证：在不计算梯度的情况下评估模型表现
    3. 学习率调度：按预定策略降低学习率
    4. 模型保存：保存最佳模型和最终模型

    属性:
        history: 训练历史字典，包含 train_loss, train_acc, val_loss, val_acc 列表。
    """

    def __init__(self, model: nn.Module, device: torch.device, config: Dict[str, Any]):
        """初始化训练器。

        参数:
            model: 待训练的 PyTorch 模型。
            device: 训练设备（torch.device('cpu') 或 torch.device('cuda')）。
            config: 完整的配置字典（从 YAML 文件加载）。
        """
        self.model = model.to(device)
        self.device = device
        self.config = config

        training_cfg = config["training"]
        output_cfg = config["output"]

        # ========== 优化器 ==========
        # SGD + 动量（Momentum）是图像分类任务中经过充分验证的组合
        # 动量: 在参数更新时保留之前的更新方向，加速收敛、减少震荡
        # 权重衰减: L2 正则化，将权重推向更小的值，防止过拟合
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=training_cfg["learning_rate"],
            momentum=training_cfg["momentum"],
            weight_decay=training_cfg["weight_decay"],
        )

        # ========== 学习率调度器 ==========
        # StepLR: 每隔固定 epoch 将学习率乘以 gamma（如 0.1）
        # 目的: 训练后期降低学习率，帮助模型精细调整，找到更好的局部最优
        scheduler_cfg = training_cfg.get("scheduler", {})
        self.scheduler_type = scheduler_cfg.get("type", "step")
        if self.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_cfg.get("step_size", 25),
                gamma=scheduler_cfg.get("gamma", 0.1),
            )
        elif self.scheduler_type == "cosine":
            # CosineAnnealingLR: 余弦退火调度，学习率按余弦曲线平滑下降
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_cfg["epochs"],
            )
        else:
            # 不使用学习率调度器
            self.scheduler = None

        # ========== 损失函数 ==========
        # CrossEntropyLoss: 结合了 LogSoftmax + NLLLoss
        # 适用于多分类问题，输入为未归一化的 logits
        self.criterion = nn.CrossEntropyLoss()

        # ========== 输出路径 ==========
        self.checkpoint_dir = output_cfg["checkpoint_dir"]
        self.log_dir = output_cfg["log_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # ========== 训练状态 ==========
        # 记录每个 epoch 的训练和验证指标
        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.best_val_acc = 0.0  # 跟踪最佳验证准确率
        self.current_epoch = 0   # 当前 epoch（用于断点续训）
        self.model_name = config["model"]["name"]

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个 epoch。

        一个 epoch 意味着完整遍历一次训练集。
        训练循环是深度学习最核心的代码，每一步都有明确的目的。

        参数:
            train_loader: 训练数据加载器。

        返回:
            (epoch_loss, epoch_acc): 该 epoch 的平均损失和准确率。
        """
        self.model.train()  # 切换到训练模式（启用 Dropout、BatchNorm 等）

        running_loss = 0.0   # 累积损失，用于计算平均
        correct = 0          # 预测正确的样本数
        total = 0            # 总样本数

        # tqdm 进度条，显示实时损失和准确率
        pbar = tqdm(train_loader, desc="训练中", leave=False, ncols=100)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # 步骤 1: 清空上一轮的梯度
            # 为什么要清零？PyTorch 默认会累积梯度，不清零会导致梯度越来越大
            self.optimizer.zero_grad()

            # 步骤 2: 前向传播 —— 模型根据输入做出预测
            outputs = self.model(images)

            # 步骤 3: 计算损失 —— 衡量预测与真实标签的差距
            loss = self.criterion(outputs, labels)

            # 步骤 4: 反向传播 —— 计算损失对每个参数的梯度
            loss.backward()

            # 步骤 5: 更新参数 —— 沿梯度下降方向调整参数
            self.optimizer.step()

            # 统计指标
            running_loss += loss.item() * images.size(0)  # 乘以 batch_size 恢复总损失
            _, predicted = outputs.max(1)  # 取最大得分的类别作为预测
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条信息
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.2f}%",
            })

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()  # 禁用梯度计算，节省内存、加速推理
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """在给定数据集上评估模型（不更新参数）。

        此方法同时用于验证集评估和测试集评估。
        @torch.no_grad() 装饰器的作用:
        - 不计算梯度 → 节省显存/内存
        - 加速前向传播

        参数:
            loader: 数据加载器（验证集或测试集）。

        返回:
            (avg_loss, accuracy): 平均损失和准确率。
        """
        self.model.eval()  # 切换到评估模式（禁用 Dropout，BatchNorm 使用全局统计量）

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc="评估中", leave=False, ncols=100)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.2f}%",
            })

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            start_epoch: int = 1) -> Dict[str, list]:
        """执行完整的训练流程。

        对每个 epoch:
        1. 训练一个 epoch
        2. 在验证集上评估
        3. 调整学习率
        4. 如果验证准确率创新高 → 保存最佳模型
        5. 打印 epoch 统计信息

        参数:
            train_loader: 训练数据加载器。
            val_loader: 验证数据加载器。
            start_epoch: 起始 epoch 编号，用于断点续训。默认为 1。

        返回:
            history: 训练历史字典。
        """
        epochs = self.config["training"]["epochs"]
        start_time = time.time()

        if start_epoch > 1:
            print(f"\n{'='*60}")
            print(f"[断点续训] 模型: {self.model_name}")
            print(f"[从 Epoch {start_epoch} 继续训练] (共 {epochs} 个 epoch)")
            print(f"[设备] {self.device} | [批次大小] {train_loader.batch_size}")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"[开始训练] 模型: {self.model_name}")
            print(f"[设备] {self.device}")
            print(f"[Epoch 数] {epochs} | [批次大小] {train_loader.batch_size}")
            print(f"[初始学习率] {self.config['training']['learning_rate']}")
            print(f"{'='*60}")

        try:
            # 断点续训时补齐 scheduler 已跳过的 epoch，保持余弦退火等调度器的相位正确
            if start_epoch > 1 and self.scheduler is not None:
                for _ in range(1, start_epoch):
                    self.scheduler.step()

            for epoch in range(start_epoch, epochs + 1):
                self.current_epoch = epoch
                epoch_start = time.time()

                # 打印当前学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\n--- Epoch {epoch:3d}/{epochs} | LR: {current_lr:.6f} ---")

                # 训练阶段
                train_loss, train_acc = self.train_epoch(train_loader)

                # 验证阶段
                val_loss, val_acc = self.evaluate(val_loader)

                # 学习率调度（StepLR / CosineAnnealingLR）
                if self.scheduler is not None:
                    self.scheduler.step()

                # 记录历史
                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                # 打印 epoch 总结
                epoch_time = time.time() - epoch_start
                print(
                    f"训练 - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | "
                    f"验证 - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | "
                    f"耗时: {epoch_time:.1f}s"
                )

                # 如果当前验证准确率是最高的，保存最佳模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_checkpoint(is_best=True)
                    print(f"  ★ 新的最佳模型！验证准确率: {val_acc:.2f}%")

        except KeyboardInterrupt:
            print(f"\n\n[训练已中断] 用户按下了 Ctrl+C")
            print(f"[自动保存] 正在保存当前训练状态...")
            self.save_checkpoint(is_best=False)
            self._save_history()
            print(f"[已保存] 已训练 {self.current_epoch}/{epochs} 个 epoch")
            print(f"[恢复命令] python main.py --resume "
                  f"{os.path.join(self.checkpoint_dir, self.model_name + '_final.pth')}")
            return self.history

        # 训练结束，保存最终模型和训练历史
        total_time = time.time() - start_time
        self.save_checkpoint(is_best=False)
        self._save_history()

        print(f"\n[训练完成] 总耗时: {total_time:.1f}s ({total_time/60:.1f}分钟)")
        print(f"[最佳验证准确率] {self.best_val_acc:.2f}%")
        return self.history

    def test(self, test_loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """在测试集上评估最终模型。

        测试集是模型从未见过的数据，用于衡量模型的泛化能力。
        测试结果才是最终报告的性能指标。

        参数:
            test_loader: 测试数据加载器。

        返回:
            (test_loss, test_acc, all_predictions, all_targets):
                测试损失、准确率、所有预测值和真实标签。
        """
        print(f"\n{'='*60}")
        print(f"[测试模型] {self.model_name}")

        # 加载最佳模型权重进行测试
        # 新格式：pure state_dict → weights_only=True；旧格式：full dict → 回退
        best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth")
        if os.path.exists(best_path):
            print(f"[加载最佳模型] {best_path}")
            try:
                state_dict = torch.load(best_path, map_location=self.device, weights_only=True)
            except Exception:
                ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
                state_dict = ckpt["model_state_dict"]
                print("[警告] 旧格式 checkpoint 检测到，请重新训练以使用安全加载格式")
            self.model.load_state_dict(state_dict)

        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="测试中", ncols=100):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)

                all_predictions.append(predicted.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        test_loss = running_loss / len(test_loader.dataset)
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        test_acc = 100.0 * (all_predictions == all_targets).mean()

        print(f"[测试结果] Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
        print(f"{'='*60}\n")
        return test_loss, test_acc, all_predictions, all_targets

    @torch.no_grad()
    def evaluate_metrics(self, test_loader: DataLoader) -> Dict[str, Any]:
        """测试并计算全量评估指标。

        包含: 测试准确率、混淆矩阵、各类别准确率、推理速度。
        供 main.py train_single_model() 和 generate_comparison.py 共用。

        参数:
            test_loader: 测试数据加载器。

        返回:
            包含 test_acc, test_loss, confusion_matrix, per_class_acc, inference_speed 的字典。
        """
        test_loss, test_acc, predictions, targets = self.test(test_loader)

        # 混淆矩阵（向量化）
        confusion_mat = np.zeros((10, 10), dtype=np.int64)
        np.add.at(confusion_mat, (targets, predictions), 1)
        row_sums = confusion_mat.sum(axis=1, keepdims=True)
        confusion_mat_norm = np.divide(
            confusion_mat, row_sums,
            out=np.zeros_like(confusion_mat, dtype=np.float64),
            where=row_sums > 0,
        )
        per_class_acc = 100.0 * confusion_mat_norm.diagonal()

        # 推理速度
        self.model.eval()
        total_samples = 0
        start = time.time()
        for images, _ in test_loader:
            images = images.to(self.device)
            _ = self.model(images)
            total_samples += images.size(0)
        elapsed = time.time() - start
        inference_speed = total_samples / elapsed if elapsed > 0 else 0

        return {
            "test_acc": test_acc,
            "test_loss": test_loss,
            "confusion_matrix": confusion_mat_norm,
            "per_class_acc": per_class_acc,
            "inference_speed": inference_speed,
        }

    def save_checkpoint(self, is_best: bool = False) -> None:
        """保存模型检查点。

        best 检查点仅保存模型权重（state_dict），可安全地用 weights_only=True 加载。
        final 检查点保存完整训练状态（含优化器、调度器、历史、配置），用于断点续训。

        参数:
            is_best: 是否为最佳模型。最佳模型仅保存 state_dict。
        """
        filename = f"{self.model_name}_{'best' if is_best else 'final'}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)

        if is_best:
            torch.save(self.model.state_dict(), filepath)
        else:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "history": self.history,
                "best_val_acc": self.best_val_acc,
                "epoch": self.current_epoch,
                "config": self.config,
            }
            torch.save(checkpoint, filepath)

    def load_checkpoint(self, path: str) -> int:
        """从检查点恢复完整训练状态。

        恢复内容包括:
        - 模型权重
        - 优化器状态（动量缓冲区等）
        - 学习率调度器状态
        - 训练历史记录
        - 最佳验证准确率

        参数:
            path: 检查点文件路径。

        返回:
            int: 检查点保存时的 epoch 编号，用于确定从哪个 epoch 继续训练。
        """
        # weights_only=False：断点续训需要加载 dict 中的 config、history 等非 tensor 对象。
        # 仅用于加载自己训练产生的 checkpoint，不加载第三方来源。
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 恢复调度器状态
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # 恢复训练历史
        if "history" in checkpoint:
            self.history = checkpoint["history"]
            self.best_val_acc = checkpoint.get("best_val_acc", 0.0)

        saved_epoch = checkpoint.get("epoch", 0)
        print(f"[检查点已加载] {path} (已训练 {saved_epoch} 个 epoch, "
              f"最佳验证准确率: {self.best_val_acc:.2f}%)")
        return saved_epoch

    def _save_history(self) -> None:
        """将训练历史保存为 JSON 文件，便于后续分析和绘图。"""
        filepath = os.path.join(self.log_dir, f"{self.model_name}_history.json")
        # 将 numpy 数组转为列表以便 JSON 序列化
        history_dict = {
            k: [float(v) for v in vals] for k, vals in self.history.items()
        }
        history_dict["best_val_acc"] = float(self.best_val_acc)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history_dict, f, indent=2, ensure_ascii=False)
        print(f"[历史记录已保存] {filepath}")

    def get_num_params(self) -> int:
        """计算模型的可训练参数总数。

        返回:
            int: 参数数量。
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
