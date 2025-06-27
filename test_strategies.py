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
    
    Args:
        env: 交易环境
        strategy: 策略对象
        max_ep_len: 最大回合长度
        state_dim: 状态维度（策略不需要，但保持接口一致）
    
    Returns:
        episode_return, episode_length: 回合回报和长度
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


def experiment_comparison(variant):
    """
    比较实验：模型 vs 均线策略
    """
    mode = variant.get('mode', 'tacr')
    device = variant.get('device', 'cuda')

    env_name, dataset = variant['env'], variant['dataset']
    group_name = f'{env_name}-{dataset}-{mode}'

    # 加载数据
    train = pd.read_csv("datasets/" + dataset+"_train.csv", index_col=[0])
    trade = pd.read_csv("datasets/" + dataset + "_trade.csv", index_col=[0])
    max_ep_len = train.index[-1]

    # 加载轨迹数据（用于模型测试）
    dataset_path = f'{"trajectory/" + variant["dataset"] + "_traj.pkl"}'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    state_space = trajectories[0]['observations'].shape[1]
    stock_dimension = len(train.tic.unique())

    print(f"股票维度: {stock_dimension}, 状态空间: {state_space}")

    # 环境配置
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

    # 设置随机种子
    seed = variant['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

    print("=" * 60)
    print("开始策略对比测试")
    print("=" * 60)

    # 1. 测试 TACR 模型
    print("\n1. 测试 TACR 模型...")
    env_model = StockPortfolioEnv(df=trade, **env_kwargs)
    env_model.seed(seed)
    env_model.action_space.seed(seed)

    state_dim = env_model.observation_space.shape[0]
    act_dim = env_model.action_space.shape[0]

    # 计算状态归一化参数
    states = []
    for path in trajectories:
        states.append(path['observations'])
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    # 加载模型
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

    model_return, model_length = eval_test(
        env_model,
        state_dim,
        act_dim,
        model,
        max_ep_len=max_ep_len,
        state_mean=state_mean,
        state_std=state_std,
        device=device
    )

    print(f"TACR 模型回报: {model_return:.4f}, 交易天数: {model_length}")

    # 2. 测试均线策略
    print("\n2. 测试均线策略...")
    env_ma = StockPortfolioEnv(df=trade, **env_kwargs)
    env_ma.seed(seed)
    env_ma.action_space.seed(seed)

    # 创建均线策略实例
    ma_strategy = MovingAverageStrategy(
        strategy_id=variant.get('ma_strategy_id', 1),
        threshold_multiplier=variant.get('ma_threshold', 0.2)
    )

    ma_return, ma_length = eval_strategy(
        env_ma,
        ma_strategy,
        max_ep_len=max_ep_len,
        state_dim=state_dim
    )

    print(f"均线策略回报: {ma_return:.4f}, 交易天数: {ma_length}")

    # 3. 测试随机策略（作为基线）
    print("\n3. 测试随机策略...")
    env_random = StockPortfolioEnv(df=trade, **env_kwargs)
    env_random.seed(seed)
    env_random.action_space.seed(seed)

    # 创建随机策略实例
    random_strategy = RandomStrategy(strategy_id=0)

    random_return, random_length = eval_strategy(
        env_random,
        random_strategy,
        max_ep_len=max_ep_len,
        state_dim=state_dim
    )

    print(f"随机策略回报: {random_return:.4f}, 交易天数: {random_length}")

    # 4. 结果对比
    print("\n" + "=" * 60)
    print("策略对比结果")
    print("=" * 60)
    print(f"TACR 模型:  {model_return:>10.4f}")
    print(f"均线策略:  {ma_return:>10.4f}")
    print(f"随机策略:  {random_return:>10.4f}")
    print("=" * 60)

    # 计算相对表现
    if random_return != 0:
        model_vs_random = (model_return - random_return) / abs(random_return) * 100
        ma_vs_random = (ma_return - random_return) / abs(random_return) * 100
        print(f"TACR vs 随机: {model_vs_random:>+8.2f}%")
        print(f"均线 vs 随机: {ma_vs_random:>+8.2f}%")

    if ma_return != 0:
        model_vs_ma = (model_return - ma_return) / abs(ma_return) * 100
        print(f"TACR vs 均线: {model_vs_ma:>+8.2f}%")

    return {
        'model_return': model_return,
        'ma_return': ma_return,
        'random_return': random_return,
        'model_length': model_length,
        'ma_length': ma_length,
        'random_length': random_length
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='策略对比测试')
    parser.add_argument('--dataset', type=str, default='csi', 
                       help='数据集选择: kdd, hightech, dow, ndx, mdax, csi')
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
    
    # 均线策略参数
    parser.add_argument('--ma_strategy_id', type=int, default=1,
                       help='均线策略强度参数')
    parser.add_argument('--ma_threshold', type=float, default=0.2,
                       help='均线策略阈值倍数')

    args = parser.parse_args()
    results = experiment_comparison(variant=vars(args)) 