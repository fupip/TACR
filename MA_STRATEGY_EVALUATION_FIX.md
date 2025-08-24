# MA策略评估函数修复说明

## 🔍 发现的问题

在 `simple_test.py` 中的 `evaluate_ma_strategy_prediction` 函数存在一个参数未使用的问题：

```python
def evaluate_ma_strategy_prediction(model, trajectories, ma_strategy, device='cpu', sequence_length=10):
    # ma_strategy 参数被定义但从未使用
```

## 📋 问题分析

### 原始问题
- 函数接收 `ma_strategy` 参数但没有使用
- 直接使用轨迹中的 `actions` 作为 ground truth
- 这意味着评估的是模型对轨迹动作的拟合能力，而不是对MA策略的预测能力

### 根本原因
- 轨迹数据中的观测值 (`observations`) 是经过标准化的
- MA策略需要原始的价格和技术指标数据（如 `close`, `close_5_sma`, `close_20_sma` 等）
- 标准化后的数据无法直接用于策略计算

## 🔧 修复方案

### 1. 短期修复（已实现）
```python
def evaluate_ma_strategy_prediction(model, trajectories, ma_strategy=None, device='cpu', sequence_length=10):
    """评估模型的动作预测能力"""
    if ma_strategy is not None:
        print(f"使用策略生成ground truth: {ma_strategy.get_strategy_info()}")
        # 警告用户当前限制
        print("警告: MA策略需要原始价格数据，当前使用轨迹动作作为替代")
    else:
        print("使用轨迹中的动作作为ground truth")
    
    # 继续使用轨迹中的动作作为ground truth
    true_class = np.argmax(actions[i])
```

### 2. 长期解决方案（建议）
创建新的评估函数，使用原始数据：

```python
def evaluate_with_strategy_ground_truth(model, raw_data_df, ma_strategy, device='cpu', sequence_length=10):
    """使用原始数据和MA策略评估模型"""
    # 1. 从原始数据生成策略标签
    strategy_actions = []
    for i in range(len(raw_data_df)):
        current_data = raw_data_df.iloc[i]
        last_data = raw_data_df.iloc[i-1] if i > 0 else None
        position, action = ma_strategy.calculate_position_and_action(current_data, last_data)
        strategy_actions.append(action)
    
    # 2. 将原始数据转换为模型输入格式
    # 3. 进行模型预测和比较
```

## 📊 数据流分析

### 当前数据流
```
原始数据 → 特征工程 → 标准化 → 轨迹生成 → 模型训练
                                    ↓
                              动作存储在轨迹中
                                    ↓
                              评估时使用轨迹动作
```

### 理想数据流
```
原始数据 → 特征工程 → 标准化 → 轨迹生成 → 模型训练
    ↓                                        ↓
MA策略计算 → 策略标签                    模型预测
    ↓                                        ↓
    └─────────────→ 评估比较 ←─────────────┘
```

## 🎯 建议的完整解决方案

### 1. 数据结构改进
在轨迹生成时保存原始数据索引：
```python
traj = {
    "observations": obs,
    "rewards": rews, 
    "dones": term,
    "actions": acs,
    "data_indices": data_indices  # 新增：原始数据索引
}
```

### 2. 评估函数重构
```python
def evaluate_ma_strategy_prediction(model, trajectories, raw_data_df, ma_strategy, device='cpu'):
    """使用MA策略作为ground truth评估模型"""
    for traj in trajectories:
        data_indices = traj['data_indices']
        
        for i, data_idx in enumerate(data_indices):
            # 模型预测
            pred_action = model.predict(...)
            
            # 策略计算
            current_data = raw_data_df.iloc[data_idx]
            last_data = raw_data_df.iloc[data_idx-1] if data_idx > 0 else None
            _, strategy_action = ma_strategy.calculate_position_and_action(current_data, last_data)
            
            # 比较预测和策略
            pred_class = np.argmax(pred_action)
            true_class = np.argmax(strategy_action)
```

## ⚠️ 当前限制

1. **数据格式不匹配**: 轨迹数据是标准化的，策略需要原始数据
2. **缺少数据映射**: 无法将轨迹中的观测映射回原始数据行
3. **策略接口**: 需要确保策略接口与数据格式兼容

## 🚀 实施建议

1. **立即**: 使用修复后的函数，明确说明当前使用轨迹动作作为ground truth
2. **短期**: 修改数据生成流程，保存原始数据索引
3. **长期**: 重构评估系统，支持多种ground truth来源

---

*修复时间: 2024年*
*问题发现: 用户反馈*
