# 动作预测损失函数修改：MSE → CrossEntropy

## 📋 修改概述

将动作预测任务从回归（MSE损失）改为分类（CrossEntropy损失），以提高动作预测的准确性和训练稳定性。

## 🔧 主要修改

### 1. simple_test.py 修改

#### 训练函数 (`train_simple_net`)
```python
# 原来：MSE损失
criterion = nn.MSELoss()

# 修改后：CrossEntropy损失
criterion = nn.CrossEntropyLoss()
```

#### 数据处理
```python
# 将动作概率分布转换为类别标签
if y_actions.shape[-1] > 1:  # 多类分类
    y_actions = torch.LongTensor(y_actions.argmax(axis=-1)).to(device)
```

#### 评估函数修改
```python
# 使用softmax + argmax进行预测
pred_probs = torch.softmax(pred_action, dim=-1)
pred_class = torch.argmax(pred_probs).item()
```

### 2. seq_trainer.py 确认

✅ 已经在使用CrossEntropy损失：
```python
bc_loss = F.cross_entropy(action_preds, action_sample.argmax(dim=-1))
```

### 3. 模型输出确认

✅ `SimpleTransformerActor` 输出logits（未经softmax），适配CrossEntropy损失：
```python
self.predict_action = nn.Linear(hidden_size, act_dim)  # 输出原始logits
```

## 📊 损失函数对比

| 特性 | MSE损失 | CrossEntropy损失 |
|------|---------|------------------|
| 任务类型 | 回归 | 分类 |
| 输出要求 | 概率分布 | Logits |
| 目标格式 | 连续值 | 类别索引 |
| 梯度特性 | 平滑 | 尖锐 |
| 适用场景 | 连续动作空间 | 离散动作选择 |

## 🎯 优势

1. **更清晰的分类信号**: CrossEntropy提供更强的分类梯度
2. **数值稳定性**: 避免概率分布的数值问题
3. **训练效率**: 分类任务通常收敛更快
4. **解释性**: 输出直接对应动作选择

## 📁 相关文件

- `simple_test.py`: 主要训练脚本
- `tac/training/seq_trainer.py`: 序列训练器
- `test_crossentropy.py`: 验证测试脚本
- `data_check.py`: 数据检查工具

## 🚀 使用方法

### 基本训练
```bash
python simple_test.py --dataset kdd --epochs 50 --batch_size 32
```

### 验证修改
```bash
python test_crossentropy.py
```

### 检查数据
```bash
python data_check.py
```

## ⚠️ 注意事项

1. **数据格式**: 确保动作数据是概率分布形式
2. **标签转换**: 使用argmax将概率转换为类别索引
3. **模型输出**: 确保模型输出logits而非概率
4. **评估指标**: 使用分类准确率而非MSE

## 🔍 验证清单

- [x] simple_test.py 使用 CrossEntropyLoss
- [x] 数据处理转换为分类标签
- [x] seq_trainer.py 确认使用 CrossEntropy
- [x] 模型输出 logits 格式
- [x] 评估函数使用 softmax + argmax
- [x] 创建测试验证脚本

## 📈 预期效果

- 提高动作预测准确率
- 加快训练收敛速度
- 增强模型的决策能力
- 改善梯度流动

---

*修改完成时间: 2024年*
*修改人: Claude Assistant*
