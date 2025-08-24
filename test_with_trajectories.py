import torch
import argparse
import pandas as pd
import random
import numpy as np
import pickle
from stock_env.apps import config
from stock_env.allocation.env_portfolio import StockPortfolioEnv
from tac.models.transformer_actor import TransformerActor
import torch.backends.cudnn as cudnn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def experiment(variant):
    """
    使用预生成的测试轨迹来评估模型
    这样可以确保训练和测试使用完全相同的标准化参数
    """
    mode = variant.get('mode', 'tacr')
    device = variant.get('device', 'cuda')
    
    env_name, dataset = variant['env'], variant['dataset']
    group_name = f'{env_name}-{dataset}-{mode}'

    # 加载训练和测试轨迹
    train_traj_path = f'trajectory/{dataset}_train_traj.pkl'
    test_traj_path = f'trajectory/{dataset}_test_traj.pkl'
    
    print(f"Loading training trajectories from: {train_traj_path}")
    print(f"Loading testing trajectories from: {test_traj_path}")
    
    with open(train_traj_path, 'rb') as f:
        train_trajectories = pickle.load(f)
    
    with open(test_traj_path, 'rb') as f:
        test_trajectories = pickle.load(f)
    
    print(f"Loaded {len(train_trajectories)} training trajectories")
    print(f"Loaded {len(test_trajectories)} testing trajectories")
    
    # 从训练轨迹获取状态空间信息
    state_space = train_trajectories[0]['observations'].shape[1]
    max_ep_len = train_trajectories[0]['observations'].shape[0]
    
    # 计算训练轨迹的标准化参数（用于模型输入）
    train_states = []
    for traj in train_trajectories:
        train_states.append(traj['observations'])
    train_states = np.concatenate(train_states, axis=0)
    state_mean, state_std = np.mean(train_states, axis=0), np.std(train_states, axis=0) + 1e-6
    
    print(f"State space: {state_space}")
    print(f"Max episode length: {max_ep_len}")
    print(f"State mean shape: {state_mean.shape}")
    print(f"State std shape: {state_std.shape}")
    
    # 从训练数据获取股票维度信息
    train_df = pd.read_csv(f"datasets/{dataset}_train.csv", index_col=[0])
    stock_dimension = len(train_df.tic.unique())
    
    print(f"Stock Dimension: {stock_dimension}")
    
    # 创建测试环境（用于实时交互）
    tech_features = ["close_60_sma_z","close_ma60_diff"]
    env_kwargs = {
        "dataset": dataset,
        "initial_amount": 1000000,
        "transaction_cost": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_features,
        "action_space": 3,
        "mode": "test",
        "turbulence_threshold": None,
    }
    
    # 加载测试数据
    trade_df = pd.read_csv(f"datasets/{dataset}_trade.csv", index_col=[0])
    trade_df = trade_df.iloc[120:].reset_index(drop=True)  # 与轨迹生成时保持一致
    
    env = StockPortfolioEnv(df=trade_df, **env_kwargs)
    
    # 设置随机种子
    seed = variant['seed']
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    print(f"Loading model: {group_name}.pt")
    
    # 加载训练好的模型
    u = variant['u']
    model = TransformerActor(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=u,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        train_mode=False,
        resid_pdrop=0.0,
        attn_pdrop=0.0
    )
    
    model.load_state_dict(torch.load(group_name+'.pt'))
    print(f"Model loaded successfully from {group_name}.pt")
    
    # 评估模型在测试轨迹上的性能
    print("\n" + "="*60)
    print("Evaluating model on pre-generated test trajectories")
    print("="*60)
    
    model.eval()
    model.to(device=device)
    
    # 使用测试轨迹评估模型
    trajectory_returns = []
    trajectory_lengths = []
    
    with torch.no_grad():
        for i, test_traj in enumerate(test_trajectories):
            print(f"\nEvaluating test trajectory {i+1}/{len(test_trajectories)}")
            
            # 获取轨迹中的观测值
            traj_obs = test_traj['observations']
            traj_actions = test_traj['actions']
            traj_rewards = test_traj['rewards']
            
            print(f"Trajectory length: {len(traj_obs)}")
            print(f"Trajectory total reward: {np.sum(traj_rewards):.4f}")
            
            # 计算模型预测的准确性
            correct_predictions = 0
            total_predictions = 0
            
            for t in range(len(traj_obs)):
                # 准备模型输入
                if t < u:  # 如果时间步小于序列长度，需要padding
                    # 创建padding
                    padding_length = u - t - 1
                    padded_states = np.zeros((u, state_dim))
                    padded_actions = np.zeros((u, act_dim))
                    padded_rewards = np.zeros(u)
                    padded_timesteps = np.arange(u)
                    
                    # 填充实际数据
                    padded_states[padding_length:] = traj_obs[:t+1]
                    if t > 0:
                        padded_actions[padding_length:-1] = traj_actions[:t]
                        padded_rewards[padding_length:-1] = traj_rewards[:t]
                else:
                    # 使用完整的序列
                    padded_states = traj_obs[t-u+1:t+1]
                    padded_actions = traj_actions[t-u:t]
                    padded_rewards = traj_rewards[t-u:t]
                    padded_timesteps = np.arange(u)
                
                # 转换为tensor
                states = torch.from_numpy(padded_states).to(device=device, dtype=torch.float32).unsqueeze(0)
                actions = torch.from_numpy(padded_actions).to(device=device, dtype=torch.float32).unsqueeze(0)
                rewards = torch.from_numpy(padded_rewards).to(device=device, dtype=torch.float32).unsqueeze(0)
                timesteps = torch.from_numpy(padded_timesteps).to(device=device, dtype=torch.long).unsqueeze(0)
                
                # 模型预测
                predicted_action = model.get_action(states, actions, rewards, timesteps)
                predicted_action = predicted_action.detach().cpu().numpy()
                
                # 获取真实动作
                true_action = traj_actions[t]
                
                # 比较预测和真实动作
                if np.argmax(predicted_action) == np.argmax(true_action):
                    correct_predictions += 1
                total_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            print(f"Trajectory {i+1} accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
            
            trajectory_returns.append(np.sum(traj_rewards))
            trajectory_lengths.append(len(traj_obs))
    
    # 输出总体评估结果
    print("\n" + "="*60)
    print("OVERALL EVALUATION RESULTS")
    print("="*60)
    print(f"Number of test trajectories: {len(test_trajectories)}")
    print(f"Average trajectory length: {np.mean(trajectory_lengths):.2f}")
    print(f"Average trajectory return: {np.mean(trajectory_returns):.4f}")
    print(f"Total trajectory return: {np.sum(trajectory_returns):.4f}")
    
    # 保存评估结果
    results = {
        'dataset': dataset,
        'model': group_name,
        'num_trajectories': len(test_trajectories),
        'avg_length': np.mean(trajectory_lengths),
        'avg_return': np.mean(trajectory_returns),
        'total_return': np.sum(trajectory_returns),
        'trajectory_returns': trajectory_returns,
        'trajectory_lengths': trajectory_lengths
    }
    
    if not os.path.exists("results"):
        os.makedirs("results")
    
    results_file = f"results/{dataset}_trajectory_evaluation.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Evaluation results saved to: {results_file}")
    
    return np.mean(trajectory_returns), np.mean(trajectory_lengths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='csi')
    parser.add_argument('--env', type=str, default='stock')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--u', type=int, default=60)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=5)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='tacr')
    
    args = parser.parse_args()
    experiment(variant=vars(args))
