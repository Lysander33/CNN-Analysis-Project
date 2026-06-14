# CNN 架构对比分析项目

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

**一个面向深度学习初学者的 CNN 实验平台 —— 在 CIFAR-10 上对比 4 种经典架构**

</div>

---

## 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [四种 CNN 架构简介](#四种-cnn-架构简介)
- [使用方法](#使用方法)
- [实验结果](#实验结果)
- [图表说明](#图表说明)
- [如何扩展](#如何扩展)
- [常见问题](#常见问题)
- [参考文献](#参考文献)

---

## 项目概述

刚开始学深度学习的时候，你一定听说过 LeNet、VGG、ResNet 这些名字。但它们在同样的条件下到底差多少？为什么后来的网络比前面的好？这个项目就是我为回答这些问题而做的实验。

我在 **CIFAR-10**（10 类 32×32 彩色图像，60,000 张）上，用**完全相同的训练配置**跑了 4 种经典 CNN，对比它们的准确率、收敛速度、参数量和推理速度，然后用 5 张图表把结果直观地展示出来。

### 能做什么

- **训练 4 种经典 CNN**：LeNet-5 (1998) → SimpleCNN → VGG-11 (2014) → ResNet-18 (2015)，感受架构演进
- **一键生成对比图表**：训练曲线、性能柱状图、混淆矩阵、各类别准确率、推理速度 — 共 5 张
- **断点续训**：训练中断了？`Ctrl+C` 保存 checkpoint，下次 `--resume` 接着跑
- **配置驱动**：改超参数不用改代码，编辑 YAML 文件即可
- **实验可复现**：固定随机种子（Python/NumPy/PyTorch/CuDNN 四层），跨越 Windows/macOS/Linux

---

## 快速开始

```bash
# 1. 安装依赖（只需一次）
pip install -r requirements.txt

# 2. 快速验证环境（3 轮，~15 分钟）
python main.py --compare --epochs 3

# 3. 完整实验（使用配置文件默认的 50 轮，CPU 约 12-20 小时）
#    建议先用 15 轮跑一遍快速对比：
python main.py --compare --epochs 15
```

> **Windows 用户**：如果报 `OMP: Error #15`，先执行 `set KMP_DUPLICATE_LIB_OK=TRUE`（详见[常见问题](#常见问题)）。

首次运行会自动下载 CIFAR-10 数据集（约 170MB），请保持网络畅通。

---

## 四种 CNN 架构简介

这四种网络恰好代表了 CNN 架构的演进脉络：

```
LeNet-5 (1998)  →  SimpleCNN  →  VGG-11 (2014)  →  ResNet-18 (2015)
   基础范式           +现代训练技巧       深度堆叠             残差连接革命
   6.2万参数          65万参数           976万参数            1117万参数
```

### LeNet-5 — 一切的起点（1998，Yann LeCun）
最早的实用 CNN，用于手写数字识别。结构极简：**2 层卷积 + 3 层全连接**，仅 6.2 万参数。它确立了"卷积提取特征 → 池化降采样 → 全连接分类"的基本范式，后续所有 CNN 都是它的变体。

### SimpleCNN — 现代训练的入门
这是我在 LeNet 基础上设计的教学网络。核心改进只有两点：**BatchNorm**（批归一化，加速收敛）和 **Dropout**（随机丢弃神经元，防过拟合）。3 层卷积，通道数 32→64→128 逐步翻倍，共 65 万参数。它证明了：好的训练技巧比单纯堆参数更有效。

### VGG-11 — 深度就是力量（2014，牛津 VGG 组）
核心哲学：**全部用 3×3 小卷积核堆叠**。两个 3×3 卷积的感受野 = 一个 5×5 卷积，但参数量更少、非线性更强。8 层卷积 + 3 层全连接，共 976 万参数。在 ImageNet 上表现优异，但在 CIFAR-10 这种小数据集上……容易过拟合。

### ResNet-18 — 跳跃连接的革命（2015，何恺明等）
引入了**残差连接（Skip Connection）**：把输入直接加到输出上，网络只需学习"残差"（输出与输入之差）。这解决了深层网络的退化问题，梯度沿"高速公路"直达。18 层、1117 万参数，是本次实验中表现最好的模型。

---

## 使用方法

### 单模型训练

```bash
# 训练 LeNet-5（最快，~30 分钟/50 轮）
python main.py --model lenet

# 减少轮数快速实验
python main.py --model lenet --epochs 15

# 训练 ResNet-18（~7.5 小时/50 轮）
python main.py --model resnet

# 自定义超参数
python main.py --model simple_cnn --epochs 30 --batch_size 64 --lr 0.001

# 从 checkpoint 恢复中断的训练（仅支持 _final.pth）
python main.py --resume results/checkpoints/resnet_final.pth

# 恢复并延长训练（如从 15 轮延长到 30 轮）
python main.py --resume results/checkpoints/resnet_final.pth --epochs 30
```

### 全模型对比

```bash
# 完整对比实验（使用配置文件中的默认 50 轮）
python main.py --compare

# 快速对比（15 轮，CPU 约 5-7 小时）
python main.py --compare --epochs 15

# 从已有 checkpoint 直接生成图表（无需重新训练！）
python generate_comparison.py
```

### 命令行参数速查

| 参数 | 说明 | 默认值 |
|------|------|:---:|
| `--model` / `-m` | 单模型训练：lenet, simple_cnn, vgg, resnet | lenet |
| `--compare` / `-c` | 全模型对比模式 | — |
| `--resume` | 从 `_final.pth` checkpoint 恢复训练（不含 `_best.pth`） | — |
| `--epochs` / `-e` | 训练轮数（覆盖 YAML 配置） | YAML 配置 |
| `--batch_size` / `-b` | 批次大小（覆盖 YAML 配置） | YAML 配置 |
| `--lr` | 初始学习率（覆盖 YAML 配置） | YAML 配置 |
| `--device` / `-d` | 训练设备：auto, cpu, cuda | auto |

---

## 实验结果

### 15 Epoch 完整训练（CPU）

下面这张表是我用 15 轮训练跑出来的最终结果：

| 模型 | 测试准确率 | 最佳验证准确率 | 参数量 | 训练时间 | 推理速度 |
|------|:---:|:---:|:---:|:---:|:---:|
| **LeNet-5** | 64.02% | 64.68% | 6.2 万 | ~30 分钟 | 3,957 样本/秒 |
| **SimpleCNN** | 77.08% | 77.56% | 65.2 万 | ~45 分钟 | 2,481 样本/秒 |
| **VGG-11** | 81.74% | 83.20% | 975.6 万 | ~2 小时 | 656 样本/秒 |
| **ResNet-18** | **85.52%** | **86.56%** | 1,117.4 万 | ~2.25 小时 | 264 样本/秒 |

生成的对比图表保存在 `results/plots/` 下（详见[图表说明](#图表说明)）。

### 我从中得到的几点发现

**1. ResNet-18 全面领先，但不是没有代价的。** 85.52% 的准确率很漂亮，但推理速度只有 264 样本/秒 — 是 LeNet-5 的 1/15。如果要做实时应用，这一点必须考虑。

**2. SimpleCNN 是性价比之王。** 仅用 VGG-11 十五分之一的参数（65 万 vs 976 万），准确率只差不到 5 个百分点，推理速度快了近 4 倍。在资源受限的场景下，它可能是最明智的选择。

**3. VGG-11 有点尴尬。** 976 万参数，训练了 2 小时，准确率却比参数少得多的 SimpleCNN 只高了 4.66%。大量全连接层参数在 CIFAR-10 上成了负担。但换个角度 — 如果数据量更大（比如 ImageNet），VGG 的优势才能发挥出来。

**4. LeNet-5 虽垫底，但"单参数贡献率"最高。** 每 1,000 个参数贡献约 1% 准确率。而且 3,957 样本/秒的推理速度，在嵌入式设备上跑完全够用。

**5. 收敛速度差异巨大。** ResNet 在第 6 轮验证准确率就达到了 77.88%（比 LeNet 最终还高），而 LeNet 到了第 15 轮还在缓慢爬升。残差连接不仅提高了最终性能，更显著加快了学习速度。

### 快速验证数据（3 Epoch）

<details>
<summary>点击展开 3 轮快速测试结果（仅供参考）</summary>

| 模型 | 测试准确率 | 训练时间 | 推理速度 |
|------|:---:|:---:|:---:|
| LeNet-5 | 51.94% | 68 秒 | 3,581 样本/秒 |
| SimpleCNN | 66.60% | 146 秒 | 2,006 样本/秒 |
| VGG-11 | 57.86% | 726 秒 | 625 样本/秒 |
| ResNet-18 | 69.18% | 1,673 秒 | 265 样本/秒 |

> 注意：3 轮数据仅用于验证训练流程是否正常，模型远未收敛，**不具备参考意义**。请使用 15+ 轮获取可靠的对比结论。

</details>

---

## 图表说明

所有图表都在 `results/plots/` 目录下，150 DPI 的 PNG 格式。这是我用 `generate_comparison.py` 从训练好的 checkpoint 直接生成的。

### `training_curves.png` — 训练曲线
左右两张子图分别展示 Loss 和 Accuracy 随 epoch 的变化。实线 = 训练集，虚线 = 验证集。**两线分叉越大 → 过拟合越严重**（VGG 最明显）。ResNet 收敛最快，LeNet 还在慢慢爬。

### `model_comparison.png` — 综合性能对比
三张柱状图并排：准确率（越高越好）、参数量（对数坐标，越少越好）、训练时间（越短越好）。可以很直观地看出"高准确率 = 更多参数 + 更长时间"的权衡。

### `confusion_matrices.png` — 混淆矩阵
4 个子图，每个模型一张热力图。行 = 真实类别，列 = 预测类别，对角线越亮越好。常见的混淆对：猫↔狗、鸟↔鹿、卡车↔汽车。ResNet 在这些易混类别上表现明显更好。

### `per_class_accuracy.png` — 各类别准确率
分组柱状图，展示了每个模型在 10 个类别上的具体表现。ship、automobile 普遍容易识别；cat、bird、deer 相对困难。ResNet 在所有类别上全面领先。

### `inference_speed.png` — 推理速度
水平柱状图，反映部署效率。LeNet-5 快到飞起（3,957 样本/秒），是 ResNet-18 的 15 倍。在移动端或实时场景中，轻量模型的优势巨大。

---

## 如何扩展

### 添加你自己的模型（3 步）

**第 1 步**：在 `models/` 下新建 Python 文件，实现你的模型类

```python
# models/my_cnn.py
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = nn.functional.adaptive_avg_pool2d(x, (8, 8))
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

**第 2 步**：在 `models/__init__.py` 中注册

```python
from models.my_cnn import MyCNN
MODEL_REGISTRY["my_cnn"] = MyCNN
```

**第 3 步**：创建配置文件 `configs/my_cnn.yaml`

```yaml
model:
  name: my_cnn
  num_classes: 10
training:
  batch_size: 128
  epochs: 15
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0001
  scheduler:
    type: step
    step_size: 25
    gamma: 0.1
data:
  dataset: cifar10
  num_workers: 2
  val_split: 0.1
  data_dir: ./data
output:
  checkpoint_dir: ./results/checkpoints
  log_dir: ./results/logs
  plot_dir: ./results/plots
```

然后直接跑：`python main.py --model my_cnn`

### 调整训练策略

编辑 YAML 配置文件即可，不用碰代码：

```yaml
# 想减轻过拟合？试试余弦退火 + 更大的权重衰减
training:
  scheduler:
    type: cosine          # step → cosine
  weight_decay: 0.001     # 0.0001 → 0.001
```

---

## 常见问题

<details>
<summary><b>Q: Windows 上报 OMP: Error #15？</b></summary>

PyTorch 在 Windows 上的 OpenMP 库冲突问题。

```bash
set KMP_DUPLICATE_LIB_OK=TRUE
python main.py --compare --epochs 15
```

也可以把 `KMP_DUPLICATE_LIB_OK=TRUE` 加入系统环境变量永久生效。
</details>

<details>
<summary><b>Q: 训练太慢怎么办？</b></summary>

各模型 15 轮参考时间（CPU）：LeNet ~30 分钟、SimpleCNN ~45 分钟、VGG ~2 小时、ResNet ~2.25 小时。

加速方法：
```bash
# 减少轮数
python main.py --compare --epochs 10

# 增大 batch_size 利用 CPU 并行
python main.py --compare --batch_size 256

# 只训练轻量模型
python main.py --model lenet
python main.py --model simple_cnn

# 有 GPU 的话
python main.py --compare --device cuda
```
</details>

<details>
<summary><b>Q: 内存/显存不足？</b></summary>

```bash
python main.py --compare --batch_size 32
```
</details>

<details>
<summary><b>Q: 如何恢复中断的训练？</b></summary>

训练中 `Ctrl+C` 暂停会自动保存 checkpoint。恢复训练：

```bash
# 从 final checkpoint 恢复（包含完整训练状态：权重、优化器、配置、历史）
python main.py --resume results/checkpoints/resnet_final.pth

# 恢复并延长训练
python main.py --resume results/checkpoints/resnet_final.pth --epochs 30
```

> **注意**：`_best.pth` 仅保存模型权重（用于推理和评估），不包含训练状态，无法用于断点续训。断点续训必须使用 `_final.pth`。

</details>

<details>
<summary><b>Q: 已经有 checkpoint，如何直接生成对比图表？</b></summary>

```bash
python generate_comparison.py
```

前提是 `results/checkpoints/` 下有各模型的 `*_best.pth` 文件。
</details>

---

## 参考文献

| 序号 | 论文/文献 |
|:---:|---|
| 1 | LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition.* Proceedings of the IEEE. |
| 2 | Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR 2015. |
| 3 | He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. |
| 4 | Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.* |

---

<div align="center">

**从 LeNet-5 到 ResNet-18，一行行代码理解 CNN 架构演进的每一步。**

</div>
