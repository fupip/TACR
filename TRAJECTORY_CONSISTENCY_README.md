# 轨迹一致性改进说明

## 🎯 问题背景

在原始实现中，`create_data.py` 和 `test.py` 使用了不同的标准化方式：

1. **`create_data.py`**: 使用滚动窗口标准化 (120天窗口)
2. **`test.py`**: 使用训练轨迹的全局统计信息进行额外标准化

这导致了**数据不一致**和**双重标准化**的问题。

## 🔧 解决方案

在 `create_data.py` 中同步生成训练和测试轨迹，确保：

- ✅ 使用完全相同的标准化参数
- ✅ 避免双重标准化
- ✅ 保障训练和测试数据的一致性

## 📁 文件结构变化

### 原始结构
```
trajectory/
├── csi_traj.pkl          # 仅包含训练轨迹
```

### 新结构
```
trajectory/
├── csi_train_traj.pkl    # 训练轨迹
└── csi_test_traj.pkl     # 测试轨迹
```

## 🚀 使用方法

### 1. 生成轨迹数据
```bash
python create_data.py --dataset csi
```

**输出**:
```
==================================================
Generating training trajectories...
==================================================
[2] total_reward: 0.1234
total training paths: 1

==================================================
Generating testing trajectories...
==================================================
[2] total_reward: 0.0987
total testing paths: 1

==================================================
Created training trajectories: 1
Created testing trajectories: 1
==================================================
```

### 2. 使用新测试脚本
```bash
python test_with_trajectories.py --dataset csi --mode tacr
```

## 🔄 标准化流程对比

### 原始流程（不一致）
```
训练数据 → 滚动窗口标准化 → 轨迹生成 → 模型训练
测试数据 → 滚动窗口标准化 → 环境交互 → 额外标准化 → 模型测试
```

### 新流程（一致）
```
完整数据 → 滚动窗口标准化 → 数据分割
    ↓
训练数据 → 轨迹生成 → 模型训练
测试数据 → 轨迹生成 → 模型测试
```

## 📊 优势分析

| 方面 | 原始方法 | 新方法 |
|------|----------|--------|
| **标准化一致性** | ❌ 不一致 | ✅ 完全一致 |
| **双重标准化** | ❌ 存在 | ✅ 避免 |
| **数据分布** | ❌ 可能偏移 | ✅ 保持一致 |
| **测试可靠性** | ❌ 较低 | ✅ 更高 |
| **维护性** | ❌ 复杂 | ✅ 简单 |

## 🎯 核心改进点

### 1. 数据预处理一致性
```python
# 确保训练和测试数据使用相同的标准化参数
train = train.iloc[120:].reset_index(drop=True)
trade = trade.iloc[120:].reset_index(drop=True)  # 同样跳过前120行
```

### 2. 轨迹生成同步
```python
# 创建训练环境
train_env = trajectory(df=train, dataset=variant['dataset'], **env_kwargs)

# 创建测试环境
trade_env = trajectory(df=trade, dataset=variant['dataset'], **env_kwargs)
```

### 3. 文件命名规范
```python
# 保存训练轨迹
train_name = f'trajectory/{variant["dataset"]}_train_traj'
# 保存测试轨迹
test_name = f'trajectory/{variant["dataset"]}_test_traj'
```

## 🔍 测试验证

### 1. 标准化参数验证
```python
# 检查训练和测试轨迹的统计特性
train_states = np.concatenate([traj['observations'] for traj in train_trajectories], axis=0)
test_states = np.concatenate([traj['observations'] for traj in test_trajectories], axis=0)

print(f"Train states mean: {np.mean(train_states, axis=0)[:5]}")
print(f"Test states mean: {np.mean(test_states, axis=0)[:5]}")
```

### 2. 数据分布一致性
```python
# 比较训练和测试数据的分布
from scipy import stats
ks_stat, p_value = stats.ks_2samp(
    train_states.flatten(), 
    test_states.flatten()
)
print(f"KS test p-value: {p_value:.6f}")
```

## 🚨 注意事项

1. **数据分割**: 确保训练和测试数据的分割边界一致
2. **标准化窗口**: 使用相同的滚动窗口大小 (120天)
3. **策略参数**: 训练和测试使用相同的策略参数
4. **文件路径**: 更新相关脚本中的轨迹文件路径

## 📈 预期效果

- **模型性能提升**: 消除数据分布偏移的影响
- **测试可靠性**: 更准确的泛化能力评估
- **代码维护**: 简化的标准化流程
- **结果一致性**: 训练和测试使用相同的数据处理标准

## 🔮 未来扩展

1. **多策略支持**: 为不同策略生成对应的测试轨迹
2. **动态标准化**: 根据市场变化调整标准化参数
3. **交叉验证**: 支持时间序列交叉验证的轨迹生成
4. **实时更新**: 支持增量式的轨迹更新机制
