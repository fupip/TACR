#!/usr/bin/env python3
"""
Simple Test Script for SimpleNet with MA Strategy
使用SimpleNet网络测试单均线MA策略的预测能力
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
import pandas as pd
import argparse
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

from stock_env.apps import config
from stock_env.allocation.env_portfolio import StockPortfolioEnv
from tac.models.simple_net import SimpleNet
from preprocessor.strategies.ma_strategy import MovingAverageStrategy
import torch.backends.cudnn as cudnn


class SimpleTransformerActor(nn.Module):
    """
    简化版的TransformerActor,使用SimpleNet替代GPT2Model
    """
    
    def __init__(self, state_dim, act_dim, hidden_size=64, max_length=20, max_ep_len=4096):
        super().__init__()
        
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # 创建SimpleNet的配置
        class SimpleConfig:
            def __init__(self):
                self.n_embd = hidden_size
                self.num_layers = 3
        
        config = SimpleConfig()
        
        # 使用SimpleNet作为主干网络
        self.transformer = SimpleNet(config)
        
        # 嵌入层
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # 预测头
        self.predict_action = nn.Linear(hidden_size, act_dim)
        
    def forward(self, states, actions, rewards, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)
            
        # 嵌入各个模态
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(rewards)
        time_embeddings = self.embed_timestep(timesteps)
        
        # 添加时间嵌入
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        
        # 堆叠输入序列 (reward, state, action)
        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1
                                    ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # 堆叠attention mask
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        
        # 通过SimpleNet处理
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        
        # 重新整形
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        
        # 预测动作（基于状态）
        action_preds = self.predict_action(x[:,1])  # 使用状态位置的输出
        
        return action_preds
    
    def predict_next_action(self, states, actions, rewards, timesteps):
        """预测下一步动作"""
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        
        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            rewards = rewards[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]
            
            # 创建attention mask
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), 
                                    torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            
            # 填充序列
            states = torch.cat([torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), 
                                        device=states.device), states], dim=1).to(dtype=torch.float32)
            actions = torch.cat([torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                                        device=actions.device), actions], dim=1).to(dtype=torch.float32)
            rewards = torch.cat([torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1), 
                                        device=rewards.device), rewards], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat([torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), 
                                        device=timesteps.device), timesteps], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None
            
        # 前向传播
        with torch.no_grad():
            action_preds = self.forward(states, actions, rewards, timesteps, attention_mask=attention_mask)
            
        return action_preds[0, -1]  # 返回最后一个时间步的预测


def load_trajectory_data(dataset_path):
    """加载轨迹数据"""
    print(f"正在加载轨迹数据: {dataset_path}")
    try:
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        print(f"成功加载 {len(trajectories)} 条轨迹")
        return trajectories
    except FileNotFoundError:
        print(f"错误: 找不到轨迹文件 {dataset_path}")
        return None
    except Exception as e:
        print(f"加载轨迹数据时出错: {e}")
        return None


def load_test_data(test_data_path):
    """加载测试数据集 (CSV格式)"""
    try:
        import pandas as pd
        test_df = pd.read_csv(test_data_path)
        print(f"成功加载测试数据: {test_data_path}")
        print(f"测试数据形状: {test_df.shape}")
        print(f"测试数据列: {test_df.columns.tolist()}")
        return test_df
    except FileNotFoundError:
        print(f"测试数据文件不存在: {test_data_path}")
        return None
    except Exception as e:
        print(f"加载测试数据时出错: {str(e)}")
        return None


def preprocess_test_data(test_df, tech_features=['close_60_sma_z', 'close_ma60_diff']):
    """预处理测试数据，转换为模型输入格式"""
    print("正在预处理测试数据...")
    
    # 确保数据按日期和股票代码排序
    if 'date' in test_df.columns and 'tic' in test_df.columns:
        test_df = test_df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    # 检查必要的列是否存在
    required_cols = ['open_z', 'high_z', 'low_z', 'close_z'] + tech_features
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    
    if missing_cols:
        print(f"警告: 测试数据缺少以下列: {missing_cols}")
        # 如果缺少标准化列，尝试使用原始列进行标准化
        if 'close_z' not in test_df.columns and 'close' in test_df.columns:
            print("尝试对价格数据进行标准化...")
            close_mean = test_df['close'].rolling(window=120).mean()
            close_std = test_df['close'].rolling(window=120).std()
            test_df['open_z'] = (test_df['open'] - close_mean) / close_std
            test_df['high_z'] = (test_df['high'] - close_mean) / close_std
            test_df['low_z'] = (test_df['low'] - close_mean) / close_std
            test_df['close_z'] = (test_df['close'] - close_mean) / close_std
            
            # 计算技术指标
            if 'close_60_sma_z' not in test_df.columns and 'close_60_sma' in test_df.columns:
                test_df['close_60_sma_z'] = (test_df['close_60_sma'] - close_mean) / close_std
            if 'close_ma60_diff' not in test_df.columns and 'close_60_sma' in test_df.columns:
                test_df['close_ma60_diff'] = (test_df['close'] - test_df['close_60_sma']) / test_df['close_60_sma']
    
    # 移除包含NaN的行
    test_df = test_df.dropna().reset_index(drop=True)
    print(f"预处理后数据形状: {test_df.shape}")
    
    return test_df


def prepare_training_data(trajectories, sequence_length=10):
    """准备训练数据，从轨迹中提取序列"""
    X_states, X_actions, X_rewards, X_timesteps = [], [], [], []
    y_actions = []
    
    print("正在准备训练数据...")
    for traj in trajectories:
        obs = traj['observations']
        actions = traj['actions']
        rewards = traj['rewards']
        
        # 从每条轨迹中提取多个序列
        for i in range(len(obs) - sequence_length):
            # 输入序列
            X_states.append(obs[i:i+sequence_length])
            X_actions.append(actions[i:i+sequence_length])
            X_rewards.append(rewards[i:i+sequence_length])
            X_timesteps.append(np.arange(i, i+sequence_length))
            
            # 目标动作（下一时刻的动作）
            y_actions.append(actions[i+sequence_length])
    
    # 转换为numpy数组
    X_states = np.array(X_states)
    X_actions = np.array(X_actions)
    X_rewards = np.array(X_rewards).reshape(-1, sequence_length, 1)
    X_timesteps = np.array(X_timesteps)
    y_actions = np.array(y_actions)
    
    print(f"准备了 {len(X_states)} 个训练样本")
    print(f"输入状态形状: {X_states.shape}")
    print(f"输入动作形状: {X_actions.shape}")
    print(f"输入奖励形状: {X_rewards.shape}")
    print(f"目标动作形状: {y_actions.shape}")
    
    # 对于交叉熵损失，我们需要将连续动作转换为离散类别
    # 这里我们使用argmax将概率分布转换为类别索引
    print("正在将连续动作转换为分类标签...")
    if y_actions.shape[-1] > 1:  # 多维动作
        # 将动作概率分布转换为类别标签
        print(f"动作维度: {y_actions.shape[-1]}，使用argmax转换为分类标签")
        print(f"样例原始动作: {y_actions[0][:5]}...")  # 显示前5个元素
        print(f"样例动作标签: {np.argmax(y_actions[0])}")
    
    return X_states, X_actions, X_rewards, X_timesteps, y_actions


def train_simple_net(model, train_data, epochs=50, batch_size=32, learning_rate=1e-3, device='cpu'):
    """训练SimpleNet模型"""
    X_states, X_actions, X_rewards, X_timesteps, y_actions = train_data
    
    # 转换为tensor
    X_states = torch.FloatTensor(X_states).to(device)
    X_actions = torch.FloatTensor(X_actions).to(device)
    X_rewards = torch.FloatTensor(X_rewards).to(device)
    X_timesteps = torch.LongTensor(X_timesteps).to(device)
    
    # 对于交叉熵损失，需要将动作转换为类别标签
    # 假设y_actions是概率分布，我们需要转换为类别索引
    if y_actions.shape[-1] > 1:  # 多类分类
        y_actions = torch.LongTensor(y_actions.argmax(axis=-1)).to(device)  # 转换为类别索引
    else:
        y_actions = torch.FloatTensor(y_actions).to(device)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 使用交叉熵损失进行动作分类预测
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    train_losses = []
    
    print("开始训练SimpleNet模型...")
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        num_batches = 0
        
        # 随机打乱数据
        indices = torch.randperm(len(X_states))
        
        for i in range(0, len(X_states), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            batch_states = X_states[batch_indices]
            batch_actions = X_actions[batch_indices]
            batch_rewards = X_rewards[batch_indices]
            batch_timesteps = X_timesteps[batch_indices]
            batch_y = y_actions[batch_indices]
            
            # 前向传播
            optimizer.zero_grad()
            predictions = model(batch_states, batch_actions, batch_rewards, batch_timesteps)
            
            # 计算损失（使用最后一个时间步的预测）
            # 对于交叉熵损失，predictions应该是logits，batch_y应该是类别索引
            loss = criterion(predictions[:, -1, :], batch_y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return train_losses


def evaluate_ma_strategy_prediction(model, trajectories, ma_strategy=None, device='cpu', sequence_length=10):
    """评估模型对MA策略的预测能力
    
    Args:
        model: 训练好的模型
        trajectories: 轨迹数据
        ma_strategy: MA策略实例（可选，如果为None则使用轨迹中的动作作为ground truth）
        device: 计算设备
        sequence_length: 序列长度
    """
    print("正在评估模型的动作预测能力...")
    
    if ma_strategy is not None:
        print(f"使用策略生成ground truth: {ma_strategy.get_strategy_info()}")
    else:
        print("使用轨迹中的动作作为ground truth")
    
    model.eval()
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for traj in trajectories:  # 使用前5条轨迹进行测试
            obs = traj['observations']
            actions = traj['actions']
            rewards = traj['rewards']
            print(f"obs: {obs.shape}")
            print(f"actions: {actions.shape}")
            print(f"rewards: {rewards.shape}")
            
            # 对每条轨迹进行预测
            for i in range(sequence_length, len(obs) - 1):
                # 准备输入序列
                input_states = torch.FloatTensor(obs[i-sequence_length:i]).unsqueeze(0).to(device)
                input_actions = torch.FloatTensor(actions[i-sequence_length:i]).unsqueeze(0).to(device)
                input_rewards = torch.FloatTensor(rewards[i-sequence_length:i]).unsqueeze(0).unsqueeze(-1).to(device)
                input_timesteps = torch.LongTensor(np.arange(i-sequence_length, i)).unsqueeze(0).to(device)
                
                # 模型预测
                pred_action = model.predict_next_action(input_states.squeeze(0), 
                                                        input_actions.squeeze(0),
                                                        input_rewards.squeeze(0).squeeze(-1),
                                                        input_timesteps.squeeze(0))
                
                # 将预测转换为动作类别（使用softmax + argmax）
                pred_probs = torch.softmax(pred_action, dim=-1)
                pred_class = torch.argmax(pred_probs).item()
                
                # 生成ground truth
                if ma_strategy is not None:
                    # TODO: 使用MA策略生成真实标签
                    # 这需要原始的价格和技术指标数据，而不是标准化后的观测数据
                    # 由于轨迹数据中的obs是标准化的，暂时使用轨迹中的动作
                    # print("警告: MA策略需要原始价格数据，当前使用轨迹动作作为替代")
                    true_class = np.argmax(actions[i])
                else:
                    # 直接使用轨迹中的动作作为ground truth
                    true_class = np.argmax(actions[i])
                
                predictions.append(pred_class)
                ground_truth.append(true_class)
    
    # 计算准确率
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"ground_truth: {ground_truth}")
    print(f"predictions: {predictions}")
    print(f"动作预测准确率: {accuracy:.4f}")
    
    # 打印详细分类报告
    class_names = ['Sell', 'Buy']
    print("\n分类报告:")
    print(classification_report(ground_truth, predictions, target_names=class_names))
    
    return accuracy, predictions, ground_truth


def compare_train_test_performance(train_results, test_results, dataset_name):
    """比较训练集和测试集的性能差异"""
    print("\n" + "=" * 60)
    print(f"数据集: {dataset_name} - 性能对比分析")
    print("=" * 60)
    
    train_acc = train_results[0]
    test_acc = test_results[0]
    
    # 计算性能差异
    acc_diff = test_acc - train_acc
    acc_ratio = test_acc / train_acc if train_acc > 0 else 0
    
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"准确率差异: {acc_diff:+.4f}")
    print(f"准确率比率: {acc_ratio:.4f}")
    
    # 泛化能力评估
    if acc_ratio >= 0.9:
        generalization = "优秀"
    elif acc_ratio >= 0.8:
        generalization = "良好"
    elif acc_ratio >= 0.7:
        generalization = "一般"
    else:
        generalization = "需要改进"
    
    print(f"泛化能力评估: {generalization}")
    
    # 过拟合检测
    if acc_diff < -0.1:
        print("⚠️  检测到可能的过拟合现象")
        print("   建议: 减少模型复杂度、增加正则化、使用更多训练数据")
    elif acc_diff > 0.05:
        print("ℹ️  测试集性能优于训练集，可能是数据分布差异或随机性")
    else:
        print("✅ 模型泛化性能正常")
    
    # 预测分布分析
    train_preds = train_results[1]
    test_preds = test_results[1]
    
    print(f"\n预测分布分析:")
    print(f"训练集预测分布: {np.bincount(train_preds)}")
    print(f"测试集预测分布: {np.bincount(test_preds)}")
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'accuracy_difference': acc_diff,
        'accuracy_ratio': acc_ratio,
        'generalization': generalization,
        'overfitting_detected': acc_diff < -0.1
    }


def generate_detailed_evaluation_report(train_results, test_results, dataset_name):
    """生成详细的评估报告，包括混淆矩阵和分类报告"""
    print("\n" + "=" * 60)
    print(f"数据集: {dataset_name} - 详细评估报告")
    print("=" * 60)
    
    train_acc, train_preds, train_truth = train_results
    test_acc, test_preds, test_truth = test_results
    
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
        
        # 训练集详细报告
        print("\n📊 训练集详细报告:")
        print("-" * 40)
        print("分类报告:")
        class_names = ['Sell', 'Hold', 'Buy'] if len(set(train_preds + train_truth)) > 2 else ['Sell', 'Buy']
        print(classification_report(train_truth, train_preds, target_names=class_names, zero_division=0))
        
        print("混淆矩阵:")
        cm_train = confusion_matrix(train_truth, train_preds)
        print(cm_train)
        
        # 测试集详细报告
        print("\n📊 测试集详细报告:")
        print("-" * 40)
        print("分类报告:")
        print(classification_report(test_truth, test_preds, target_names=class_names, zero_division=0))
        
        print("混淆矩阵:")
        cm_test = confusion_matrix(test_truth, test_preds)
        print(cm_test)
        
        # 计算每个类别的性能
        print("\n📈 各类别性能分析:")
        print("-" * 40)
        
        for i, class_name in enumerate(class_names):
            if i < len(cm_train) and i < len(cm_test):
                # 训练集该类别的准确率
                train_class_acc = cm_train[i, i] / cm_train[i, :].sum() if cm_train[i, :].sum() > 0 else 0
                # 测试集该类别的准确率
                test_class_acc = cm_test[i, i] / cm_test[i, :].sum() if cm_test[i, :].sum() > 0 else 0
                
                print(f"{class_name}:")
                print(f"  训练集准确率: {train_class_acc:.4f}")
                print(f"  测试集准确率: {test_class_acc:.4f}")
                print(f"  性能差异: {test_class_acc - train_class_acc:+.4f}")
        
        return {
            'train_confusion_matrix': cm_train.tolist(),
            'test_confusion_matrix': cm_test.tolist(),
            'class_names': class_names
        }
        
    except ImportError:
        print("⚠️  scikit-learn未安装，无法生成详细报告")
        print("   建议安装: pip install scikit-learn")
        return None
    except Exception as e:
        print(f"⚠️  生成详细报告时出错: {str(e)}")
        return None

def visualize_results(train_losses, accuracy, predictions, ground_truth):
    """可视化训练结果"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练损失
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # 准确率显示
    ax2.bar(['Accuracy'], [accuracy])
    ax2.set_title(f'Prediction Accuracy: {accuracy:.4f}')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    
    # 预测vs真实值分布
    unique, counts_pred = np.unique(predictions, return_counts=True)
    unique_true, counts_true = np.unique(ground_truth, return_counts=True)
    
    x = np.arange(3)
    width = 0.35
    
    ax3.bar(x - width/2, [counts_pred[i] if i in unique else 0 for i in range(3)], 
            width, label='Predicted', alpha=0.7)
    ax3.bar(x + width/2, [counts_true[i] if i in unique_true else 0 for i in range(3)], 
            width, label='Ground Truth', alpha=0.7)
    ax3.set_xlabel('Action Class')
    ax3.set_ylabel('Count')
    ax3.set_title('Action Distribution Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Sell', 'Hold', 'Buy'])
    ax3.legend()
    
    # 预测准确性时间序列（前100个预测）
    if len(predictions) > 100:
        sample_preds = predictions[:100]
        sample_true = ground_truth[:100]
        correct = [1 if p == t else 0 for p, t in zip(sample_preds, sample_true)]
        ax4.plot(correct, 'o-', alpha=0.7)
        ax4.set_title('Prediction Correctness (First 100 samples)')
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel('Correct (1) / Incorrect (0)')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='SimpleNet MA Strategy Test')
    parser.add_argument('--dataset', type=str, default='csi', 
                        choices=['kdd', 'csi', 'dow', 'hightech', 'ndx', 'mdax'],
                        help='Dataset to use')
    parser.add_argument('--device', type=str, default='cuda', 
                        choices=['cpu', 'cuda'],
                        help='Device to use for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Input sequence length')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size for SimpleNet')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    
    print("=" * 80)
    print("SimpleNet MA Strategy Prediction Test")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Hidden Size: {args.hidden_size}")
    print(f"Random Seed: {args.seed}")
    print("=" * 80)
    
    # 检查设备可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用,切换到CPU")
        args.device = 'cpu'
    
    # 加载训练轨迹数据
    train_dataset_path = f'trajectory/{args.dataset}_train_traj.pkl'
    train_trajectories = load_trajectory_data(train_dataset_path)
    
    # 加载测试轨迹数据
    test_dataset_path = f'trajectory/{args.dataset}_test_traj.pkl'
    test_trajectories = load_trajectory_data(test_dataset_path)
    
    # 如果测试轨迹不存在，尝试加载旧格式的轨迹文件
    if test_trajectories is None:
        print("测试轨迹数据不存在，尝试加载旧格式轨迹文件...")
        old_dataset_path = f'trajectory/{args.dataset}_traj.pkl'
        old_trajectories = load_trajectory_data(old_dataset_path)
        if old_trajectories is not None:
            print("使用旧格式轨迹文件作为训练数据")
            train_trajectories = old_trajectories
            test_trajectories = None
        else:
            print("无法加载任何轨迹数据，程序退出")
            return
    
    if train_trajectories is None:
        print("无法加载训练轨迹数据，程序退出")
        return
    
    # 获取数据维度信息
    sample_traj = train_trajectories[0]
    state_dim = sample_traj['observations'].shape[1]
    act_dim = sample_traj['actions'].shape[1]
    
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {act_dim}")
    
    # 准备训练数据
    train_data = prepare_training_data(train_trajectories, args.sequence_length)
    
    if len(train_data[0]) == 0:
        print("没有足够的数据进行训练，程序退出")
        return
    
    # 创建模型
    model = SimpleTransformerActor(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=args.hidden_size,
        max_length=args.sequence_length
    ).to(args.device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # # 训练模型
    train_losses = train_simple_net(
        model, train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # 创建MA策略进行比较
    ma_strategy = MovingAverageStrategy(strategy_id=2)
    
    # 1. 评估训练数据集（轨迹数据）
    print("\n" + "=" * 60)
    print("在训练数据集上评估模型")
    print("=" * 60)
    train_accuracy, train_predictions, train_ground_truth = evaluate_ma_strategy_prediction(
        model, train_trajectories, ma_strategy, args.device, args.sequence_length
    )
    print(f"训练数据集准确率: {train_accuracy:.4f}")
    
    # 2. 评估测试数据集（轨迹数据）
    if test_trajectories is not None:
        print("\n" + "=" * 60)
        print("在测试数据集上评估模型")
        print("=" * 60)
        print(f"test_trajectories: {test_trajectories}")
        test_accuracy, test_predictions, test_ground_truth = evaluate_ma_strategy_prediction(
            model, test_trajectories, ma_strategy, args.device, args.sequence_length
        )
        print(f"测试数据集准确率: {test_accuracy:.4f}")
        
        # 使用新的性能比较函数
        comparison_results = compare_train_test_performance(
            (train_accuracy, train_predictions, train_ground_truth),
            (test_accuracy, test_predictions, test_ground_truth),
            args.dataset
        )
        
        # 生成详细评估报告
        detailed_report = generate_detailed_evaluation_report(
            (train_accuracy, train_predictions, train_ground_truth),
            (test_accuracy, test_predictions, test_ground_truth),
            args.dataset
        )
        
        # 保存评估结果
        results = {
            'dataset': args.dataset,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'train_ground_truth': train_ground_truth,
            'test_ground_truth': test_ground_truth,
            'comparison_analysis': comparison_results,
            'detailed_report': detailed_report
        }
        
        # 保存结果到文件
        import json
        results_file = f'results_{args.dataset}_evaluation.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"评估结果已保存到: {results_file}")
        
    else:
        print("\n⚠️  没有测试轨迹数据，跳过测试集评估")
        print("建议运行 create_data.py 生成测试轨迹数据")
        
        # 只保存训练结果
        results = {
            'dataset': args.dataset,
            'train_accuracy': train_accuracy,
            'train_predictions': train_predictions,
            'train_ground_truth': train_ground_truth,
            'note': 'No test trajectories available'
        }
        
        import json
        results_file = f'results_{args.dataset}_train_only.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"训练结果已保存到: {results_file}")
    
    # for p,g in zip(predictions, ground_truth):
    #     print(f"predictions: {p}, ground_truth: {g}")
    
    # # 可视化结果
    # print("\n正在生成结果图表...")
    # fig = visualize_results(train_losses, accuracy, predictions, ground_truth)
    
    # # 保存模型
    # model_path = f'simple_net_{args.dataset}_model.pt'
    # torch.save(model.state_dict(), model_path)
    # print(f"\n模型已保存至: {model_path}")
    
    # print("\n" + "=" * 80)
    # print("测试完成!")
    # print(f"最终预测准确率: {accuracy:.4f}")
    # print("结果图表已保存为: simple_test_results.png")
    # print("=" * 80)


if __name__ == '__main__':
    main()
