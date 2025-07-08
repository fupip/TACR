# Informer模型实现

基于GPT2Model结构实现的Informer时间序列预测模型，专门用于长序列时间序列预测任务。

## 模型特点

### 1. ProbSparse自注意力机制
- **问题解决**: 传统Transformer在长序列上的O(L²)复杂度问题
- **解决方案**: 通过稀疏注意力将复杂度降低到O(L log L)
- **核心思想**: 只关注最重要的查询-键对，忽略冗余的注意力连接

### 2. 自注意力蒸馏
- **目的**: 减少网络参数和计算量
- **实现**: 在编码器中使用卷积层进行特征蒸馏
- **效果**: 提取注意力的主要模式，去除冗余信息

### 3. 生成式解码器
- **优势**: 一次性生成长序列预测，避免误差累积
- **机制**: 使用掩码多头注意力确保因果性
- **应用**: 特别适合长期时间序列预测

### 4. 多种时间嵌入
- **位置嵌入**: 学习序列中的位置信息
- **时间特征嵌入**: 处理年、月、日、小时等时间特征
- **值嵌入**: 通过卷积网络处理原始时间序列值

## 文件结构

```
tac/models/
├── informer.py              # Informer模型主要实现
├── informer_example.py      # 使用示例和演示
└── INFORMER_README.md       # 本说明文档
```

## 核心组件

### 1. InformerConfig
模型配置类，继承自GPT2Config，添加了时间序列预测相关的参数：
- `seq_len`: 输入序列长度
- `pred_len`: 预测序列长度  
- `d_model`: 模型维度
- `factor`: ProbSparse注意力采样因子

### 2. ProbAttention
ProbSparse注意力机制的核心实现：
```python
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        # 初始化ProbSparse注意力
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        # 计算稀疏性度量，选择重要的查询
        
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # 执行ProbSparse注意力计算
```

### 3. Encoder & Decoder
- **Encoder**: 多层编码器，可选择性地使用注意力蒸馏
- **Decoder**: 生成式解码器，支持长序列一次性预测

### 4. 嵌入层
- **DataEmbedding**: 组合值嵌入、位置嵌入和时间嵌入
- **TokenEmbedding**: 通过1D卷积处理输入值
- **PositionalEmbedding**: 标准位置编码
- **TemporalEmbedding**: 时间特征嵌入

## 使用方法

### 基本使用

```python
from informer import InformerConfig, InformerForPrediction

# 1. 创建配置
config = InformerConfig(
    seq_len=96,      # 输入序列长度
    pred_len=24,     # 预测长度
    d_model=512,     # 模型维度
    n_head=8,        # 注意力头数
    n_layer=6,       # 编码器层数
    n_embd=7,        # 特征数
    factor=5         # ProbSparse因子
)

# 2. 创建模型
model = InformerForPrediction(config)

# 3. 准备数据
# x_enc: [batch_size, seq_len, features] - 编码器输入
# x_mark_enc: [batch_size, seq_len, 4] - 编码器时间标记
# x_dec: [batch_size, label_len + pred_len, features] - 解码器输入
# x_mark_dec: [batch_size, label_len + pred_len, 4] - 解码器时间标记

# 4. 前向传播
outputs = model(
    x_enc=x_enc,
    x_mark_enc=x_mark_enc,
    x_dec=x_dec,
    x_mark_dec=x_mark_dec
)

# 5. 获取预测结果
prediction = outputs.prediction  # [batch_size, label_len + pred_len, features]
```

### 运行示例

```bash
cd tac/models
python informer_example.py
```

## 模型架构对比

| 模型 | 注意力复杂度 | 内存使用 | 长序列处理 | 预测方式 |
|------|-------------|----------|------------|----------|
| Transformer | O(L²) | 高 | 困难 | 自回归 |
| Informer | O(L log L) | 低 | 优秀 | 生成式 |

## 适用场景

1. **长期时间序列预测**: 预测长度 > 48个时间步
2. **多变量时间序列**: 支持多个特征的联合预测
3. **实时预测**: 低延迟的在线预测需求
4. **资源受限环境**: 相比标准Transformer节省计算和内存

## 与GPT2Model的关系

本实现参考了GPT2Model的架构设计模式：
- **模块化设计**: 清晰的组件分离
- **配置驱动**: 通过Config类管理所有超参数
- **预训练模型接口**: 兼容Transformers库的加载和保存机制
- **并行化支持**: 支持模型并行和设备映射

## 技术细节

### ProbSparse注意力计算流程
1. **稀疏性度量**: 计算查询-键的稀疏性分数
2. **Top-k选择**: 选择最重要的查询进行注意力计算
3. **上下文更新**: 基于选中的查询更新上下文向量

### 注意力蒸馏机制
```python
# 编码器中的蒸馏层
conv_layers = [ConvLayer(d_model) for _ in range(n_layer - 1)]
```

### 生成式解码
- 一次性生成整个预测序列
- 使用因果掩码确保时间顺序
- 避免自回归预测的误差累积

## 性能优化建议

1. **调整factor参数**: 控制注意力稀疏度，factor越小越稀疏
2. **启用蒸馏**: 设置`distil=True`减少计算量
3. **合理设置d_model**: 平衡模型容量和计算效率
4. **批量大小**: 根据GPU内存调整batch_size

## 扩展可能

1. **多尺度注意力**: 处理不同时间粒度的模式
2. **自适应稀疏**: 动态调整注意力稀疏度
3. **知识蒸馏**: 从大模型向小模型转移知识
4. **领域适应**: 针对特定领域优化模型结构

## 参考文献

- Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (AAAI 2021)
- Attention Is All You Need (NIPS 2017)
- GPT-2: Language Models are Unsupervised Multitask Learners 