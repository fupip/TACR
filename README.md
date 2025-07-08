# TACR: Transformer Actor-Critic with Regularization

一个结合Transformer和离线强化学习的自动化股票交易算法集成模型。

<p align="center">
  <img src="https://user-images.githubusercontent.com/104193216/214259965-0fbc1ac5-c3c4-4590-a267-b2a279239c40.PNG" width="700">
</p>

## 📋 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [使用指南](#使用指南)
- [策略系统](#策略系统)
- [训练与测试](#训练与测试)
- [参数配置](#参数配置)
- [数据格式](#数据格式)
- [扩展开发](#扩展开发)
- [许可证](#许可证)

## 📖 项目简介

TACR (Transformer Actor-Critic with Regularization) 是一个基于深度强化学习的股票交易系统，主要特点包括：

- **Transformer架构**: 使用GPT-2模型处理序列化的市场数据
- **Actor-Critic框架**: 实现了TACR、CQL、IQL等多种离线强化学习算法
- **多策略支持**: 内置均线策略、动量策略、随机策略等传统交易策略
- **灵活的数据处理**: 支持多种数据集格式和技术指标
- **完整的回测系统**: 提供策略对比和性能评估功能

## ✨ 功能特性

### 🤖 智能交易算法
- **TACR**: 基于Transformer的Actor-Critic算法
- **CQL**: Conservative Q-Learning离线强化学习
- **IQL**: Implicit Q-Learning算法

### 📊 传统交易策略
- **均线策略**: 基于5日、20日、60日移动平均线
- **动量策略**: 基于RSI指标的动量交易
- **随机策略**: 用于基线对比的随机交易策略

### 🔧 系统功能
- 自动化数据预处理和特征工程
- 轨迹生成和离线训练数据构建
- 多策略性能对比和评估
- 灵活的参数配置和超参数调优
- 完整的训练、验证和测试流程

## 🚀 环境配置

### 方法1：使用Conda (推荐)

```bash
# 创建环境
conda env create -f conda_stock.yaml

# 激活环境
conda activate stock
```

### 方法2：手动安装

```bash
# 创建Python环境
conda create -n stock python=3.9

# 激活环境
conda activate stock

# 安装依赖
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib scikit-learn
pip install transformers tqdm wandb
pip install gym stockstats yfinance
```

## 🎯 快速开始

### 基本使用流程

```bash
# 1. 数据准备
python create_data.py --dataset csi

# 2. 模型训练
python train.py --dataset csi --mode tacr

# 3. 模型测试
python test.py --dataset csi --test_strategy model

# 4. 策略对比
python run_strategy_comparison.py --dataset csi
```

### 单步测试示例

```bash
# 测试TACR模型
python test.py --dataset csi --test_strategy model

# 测试均线策略
python test.py --dataset csi --test_strategy ma --ma_strategy_id 5

# 测试随机策略
python test.py --dataset csi --test_strategy random
```

## 📁 项目结构

```
TACR/
├── README.md                    # 项目说明文档
├── conda_stock.yaml            # Conda环境配置
├── install.md                  # 安装说明
├── License                     # 许可证
│
├── datasets/                   # 数据集目录
│   ├── csi_train.csv          # 训练数据
│   ├── csi_test.csv           # 测试数据
│   └── ...
│
├── trajectory/                 # 轨迹数据目录
│   ├── csi_traj.pkl           # 生成的轨迹文件
│   └── ...
│
├── results/                    # 结果输出目录
│
├── 核心脚本/
├── create_data.py              # 数据预处理脚本
├── train.py                    # 模型训练脚本
├── test.py                     # 模型测试脚本
├── test_strategies.py          # 策略对比脚本
├── run_strategy_comparison.py  # 策略对比快速运行脚本
├── validate_cql_training.py    # CQL训练验证脚本
├── csi_data.py                 # CSI数据处理脚本
├── evaluate_episodes.py        # 回合评估脚本
│
├── tac/                        # 核心算法模块
│   ├── models/                 # 模型定义
│   │   ├── model.py           # 基础模型类
│   │   ├── transformer_actor.py # Transformer Actor模型
│   │   └── trajectory_gpt2.py  # GPT-2轨迹模型
│   ├── training/               # 训练模块
│   │   ├── trainer.py         # 基础训练器
│   │   ├── seq_trainer.py     # 序列训练器
│   │   ├── critic.py          # Critic网络
│   │   └── value_net.py       # 价值网络
│   └── evaluation/             # 评估模块
│
├── stock_env/                  # 交易环境模块
│   ├── allocation/             # 资产配置模块
│   │   └── env_portfolio.py   # 投资组合环境
│   └── apps/                   # 应用配置
│       └── config.py          # 配置文件
│
├── preprocessor/               # 数据预处理模块
│   ├── strategies/             # 策略模块
│   │   ├── base_strategy.py   # 基础策略类
│   │   ├── ma_strategy.py     # 均线策略
│   │   ├── momentum_strategy.py # 动量策略
│   │   └── random_strategy.py  # 随机策略
│   ├── preprocessors.py        # 数据预处理器
│   ├── process_traj.py         # 轨迹处理
│   ├── yahoodownloader.py      # Yahoo数据下载器
│   ├── strategy_example.py     # 策略使用示例
│   ├── test_strategies.py      # 策略测试
│   └── STRATEGY_README.md      # 策略系统文档
│
└── 文档/
    ├── STRATEGY_TESTING_README.md  # 策略测试指南
    └── readme.txt                  # 简要说明
```

## 📚 使用指南

### 数据准备

1. **数据下载**: 支持多种数据集 (CSI, DOW, KDD, HIGHTECH, NDX, MDAX)
2. **数据预处理**: 自动计算技术指标和特征工程
3. **轨迹生成**: 将历史数据转换为强化学习训练轨迹

```bash
# 准备CSI数据集
python create_data.py --dataset csi

# 准备其他数据集
python create_data.py --dataset dow
python create_data.py --dataset kdd
```

### 模型训练

支持多种训练模式：

```bash
# TACR模式训练
python train.py --dataset csi --mode tacr --max_iters 10 --num_steps_per_iter 1000

# CQL模式训练
python train.py --dataset csi --mode cql --max_iters 10 --num_steps_per_iter 1000

# IQL模式训练
python train.py --dataset csi --mode iql --max_iters 10 --num_steps_per_iter 1000
```

### 模型测试

```bash
# 测试训练好的模型
python test.py --dataset csi --test_strategy model

# 测试传统策略
python test.py --dataset csi --test_strategy ma --ma_strategy_id 5
python test.py --dataset csi --test_strategy random
```

## 🎛️ 策略系统

### 内置策略

详细的策略文档请参考 [`preprocessor/STRATEGY_README.md`](preprocessor/STRATEGY_README.md)

#### 1. 均线策略 (Moving Average)
- **信号生成**: 基于5日、20日、60日移动平均线交叉
- **参数调节**: `strategy_id` 控制策略激进程度
- **阈值设置**: `threshold_multiplier` 调整买卖阈值

#### 2. 动量策略 (Momentum)
- **技术指标**: 基于RSI指标
- **信号判断**: 超买超卖区间判断
- **参数控制**: 回看期间和动量阈值

#### 3. 随机策略 (Random)
- **基线对比**: 用于策略性能基线测试
- **随机种子**: 确保结果可重现

### 策略对比测试

详细的测试指南请参考 [`STRATEGY_TESTING_README.md`](STRATEGY_TESTING_README.md)

```bash
# 运行所有策略对比
python run_strategy_comparison.py --dataset csi

# 详细的策略测试
python test_strategies.py --dataset csi
```

## 🔬 训练与测试

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `csi` | 数据集选择 |
| `--mode` | `tacr` | 训练模式 (tacr/cql/iql) |
| `--u` | `60` | 序列长度 |
| `--embed_dim` | `128` | 嵌入维度 |
| `--n_layer` | `5` | Transformer层数 |
| `--n_head` | `4` | 注意力头数 |
| `--dropout` | `0.1` | Dropout概率 |
| `--learning_rate` | `1e-5` | 学习率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--batch_size` | `64` | 批次大小 |
| `--max_iters` | `10` | 最大迭代次数 |

### 测试参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--test_strategy` | `model` | 测试策略类型 |
| `--ma_strategy_id` | `1` | 均线策略强度 |
| `--ma_threshold` | `0.2` | 均线策略阈值 |
| `--seed` | `0` | 随机种子 |

## 📊 数据格式

### 输入数据格式

数据集应包含以下必要字段：

```csv
date,tic,open,high,low,close,volume,close_5_sma,close_20_sma,close_60_sma,rsi_14,macd
2020-01-01,STOCK1,100.0,102.0,99.0,101.0,1000000,100.5,99.8,98.2,55.2,0.5
...
```

### 技术指标

系统自动计算的技术指标包括：
- 移动平均线 (SMA 5, 20, 60日)
- 相对强弱指数 (RSI 14日)
- MACD指标
- 布林带
- 成交量相关指标

## 🛠️ 扩展开发

### 添加新策略

1. 继承 `BaseStrategy` 类
2. 实现 `calculate_position_and_action` 方法
3. 在工厂函数中注册新策略

```python
from preprocessor.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def calculate_position_and_action(self, data, last_day_data=None):
        # 实现策略逻辑
        if condition:
            return 1.0, np.array([0.0, 0.0, 1.0])  # 买入
        else:
            return -1.0, np.array([1.0, 0.0, 0.0])  # 卖出
```

### 添加新数据集

1. 准备符合格式的CSV文件
2. 在 `create_data.py` 中添加数据集配置
3. 运行数据预处理脚本

### 模型改进

1. 在 `tac/models/` 目录下添加新模型
2. 在 `tac/training/` 目录下实现对应的训练器
3. 在训练脚本中添加模型选项

## 🔍 常见问题

### 训练问题

**Q: bc_loss卡在0.55无法下降怎么办？**
A: 可能是防止过拟合参数设置过于保守，建议：
- 调整学习率: `--learning_rate 3e-5`
- 放宽梯度裁剪: 修改 `clip_grad_norm_` 参数
- 减少权重衰减: `--weight_decay 1e-5`

**Q: 训练时显存不足怎么办？**
A: 减少批次大小和序列长度：
- `--batch_size 32`
- `--u 40`

### 测试问题

**Q: 策略对比结果不一致怎么办？**
A: 确保使用相同的随机种子：
- `--seed 42`

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](License) 文件。

## 🙏 致谢

- 感谢 KDD 21 数据集的提供 ([Adv-ALSTM](https://github.com/fulifeng/Adv-ALSTM))
- 感谢 Hugging Face Transformers 库的支持
- 感谢开源社区的贡献

---

> 📧 如有问题，请提交 Issue 或参考相关文档。
> 
> 🔔 **重要提醒**: 本项目生成的交易信号仅供研究和学习使用，不构成投资建议。实际交易请谨慎考虑市场风险。
