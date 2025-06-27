import torch
import argparse
import pandas as pd
import random
import numpy as np
import pickle
from stock_env.apps import config
from stock_env.allocation.env_portfolio import StockPortfolioEnv
from tac.evaluation.evaluate_episodes import eval_test
from tac.models.transformer_actor import TransformerActor
from preprocessor.strategies.ma_strategy import MovingAverageStrategy
from preprocessor.strategies.random_strategy import RandomStrategy
import torch.backends.cudnn as cudnn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def eval_strategy(env, strategy, max_ep_len, state_dim=None):
    """
    评估策略在环境中的表现
    """
    state = env.reset()
    episode_return, episode_length = 0, 0
    
    for t in range(max_ep_len):
        # 获取当前交易日的数据
        current_data = env.data
        last_data = None
        if t > 0:
            # 获取前一天的数据（如果可用）
            try:
                last_data = env.df.loc[env.day - 1, :]
            except:
                last_data = None
        
        # 使用策略计算动作
        position, action = strategy.calculate_position_and_action(current_data, last_data)
        
        # 执行动作
        state, reward, done, _ = env.step(action)
        
        episode_return += reward
        episode_length += 1
        
        if done:
            break
    
    return episode_return, episode_length

def experiment(variant):
    mode = variant.get('mode', 'tacr')
    device = variant.get('device', 'cuda')
    test_strategy = variant.get('test_strategy', 'model')  # 新增：选择测试类型

    env_name, dataset = variant['env'], variant['dataset']
    group_name = f'{env_name}-{dataset}-{mode}'

    train = pd.read_csv("datasets/" + dataset+"_train.csv", index_col=[0])
    trade = pd.read_csv("datasets/" + dataset + "_trade.csv", index_col=[0])
    max_ep_len = train.index[-1]

    # 对于策略测试，可能不需要轨迹数据
    if test_strategy == 'model':
        dataset_path = f'{"trajectory/" + variant["dataset"] + "_traj.pkl"}'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        state_space = trajectories[0]['observations'].shape[1]
    else:
        # 对于策略测试，从训练数据中推断状态空间
        state_space = 4 + len(config.TECHNICAL_INDICATORS_LIST)  # open, high, low, close + 技术指标
        
    stock_dimension = len(train.tic.unique())

    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    turbulence_threshold = 100 if dataset == "dow" else None
    env_kwargs = {
        "dataset": dataset,
        "initial_amount": 1000000,
        "transaction_cost": 0.002,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": 3,
        "mode": "test",
        "turbulence_threshold": turbulence_threshold,
    }

    env = StockPortfolioEnv(df=trade, **env_kwargs)

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

    if test_strategy == 'model':
        # 测试模型
        print("测试 TACR 模型...")
        
        states = []
        for path in trajectories:
            states.append(path['observations'])

        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

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
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'])

        model.load_state_dict(torch.load(group_name+'.pt'))

        episode_return, episode_length = eval_test(
            env,
            state_dim,
            act_dim,
            model,
            max_ep_len=max_ep_len,
            state_mean=state_mean,
            state_std=state_std,
            device=device
        )
        
        print(f"TACR 模型回报: {episode_return:.4f}, 交易天数: {episode_length}")
        
    elif test_strategy == 'ma':
        # 测试均线策略
        print("测试均线策略...")
        
        ma_strategy = MovingAverageStrategy(
            strategy_id=variant.get('ma_strategy_id', 1),
            threshold_multiplier=variant.get('ma_threshold', 0.2)
        )
        
        episode_return, episode_length = eval_strategy(
            env,
            ma_strategy,
            max_ep_len=max_ep_len,
            state_dim=state_dim
        )
        
        print(f"均线策略回报: {episode_return:.4f}, 交易天数: {episode_length}")
        
    elif test_strategy == 'random':
        # 测试随机策略
        print("测试随机策略...")
        
        random_strategy = RandomStrategy(strategy_id=0)
        
        episode_return, episode_length = eval_strategy(
            env,
            random_strategy,
            max_ep_len=max_ep_len,
            state_dim=state_dim
        )
        
        print(f"随机策略回报: {episode_return:.4f}, 交易天数: {episode_length}")
    
    else:
        raise ValueError(f"不支持的测试策略: {test_strategy}")

    return episode_return, episode_length

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='csi')  # kdd, hightech, dow,  ndx, mdax, csi
    parser.add_argument('--env', type=str, default='stock')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--u', type=int, default=60)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=5)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='tacr')
    
    # 新增：测试策略选择
    parser.add_argument('--test_strategy', type=str, default='model', 
                       choices=['model', 'ma', 'random'],
                       help='选择测试策略: model(TACR模型), ma(均线策略), random(随机策略)')
    
    # 均线策略参数
    parser.add_argument('--ma_strategy_id', type=int, default=1,
                       help='均线策略强度参数')
    parser.add_argument('--ma_threshold', type=float, default=0.2,
                       help='均线策略阈值倍数')

    args = parser.parse_args()
    experiment(variant=vars(args))
