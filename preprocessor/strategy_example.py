"""
策略系统使用示例
展示如何使用不同的交易策略生成轨迹
"""

import pandas as pd
import numpy as np
from strategies import create_strategy, MovingAverageStrategy, MomentumStrategy, RandomStrategy
from process_traj import trajectory


def strategy_demo():
    """策略系统演示"""
    
    # 模拟一些数据 (实际使用时从CSV文件加载)
    dates = pd.date_range('2023-01-01', periods=100)
    data = {
        'date': dates,
        'tic': ['AAPL'] * 100,
        'open': np.random.uniform(150, 160, 100),
        'high': np.random.uniform(160, 170, 100),
        'low': np.random.uniform(140, 150, 100),
        'close': np.random.uniform(150, 160, 100),
        'close_5_sma': np.random.uniform(150, 160, 100),
        'close_20_sma': np.random.uniform(150, 160, 100),
        'close_60_sma': np.random.uniform(150, 160, 100),
        'rsi_14': np.random.uniform(30, 70, 100),
    }
    df = pd.DataFrame(data)
    df.index = range(len(df))
    
    print("=== 策略系统演示 ===\n")
    
    # 1. 演示不同策略的创建
    print("1. 创建不同类型的策略:")
    
    # 移动平均线策略
    ma_strategy = create_strategy('moving_average', strategy_id=5)
    print(f"移动平均线策略: {ma_strategy.get_strategy_info()}")
    
    # 动量策略
    momentum_strategy = create_strategy('momentum', strategy_id=3, momentum_threshold=0.05)
    print(f"动量策略: {momentum_strategy.get_strategy_info()}")
    
    # 随机策略
    random_strategy = create_strategy('random', strategy_id=1, random_seed=123)
    print(f"随机策略: {random_strategy.get_strategy_info()}")
    
    print("\n" + "="*50 + "\n")
    
    # 2. 演示在轨迹生成中使用不同策略
    print("2. 在轨迹生成中使用不同策略:")
    
    tech_indicators = ['close_5_sma', 'close_20_sma', 'close_60_sma', 'rsi_14']
    
    # 使用移动平均线策略的轨迹
    traj_ma = trajectory(
        dataset='demo',
        df=df,
        stock_dim=1,
        state_space=len(tech_indicators) + 4,  # OHLC + 技术指标
        action_space=3,
        tech_indicator_list=tech_indicators,
        strategy_name='moving_average'
    )
    
    # 使用动量策略的轨迹
    traj_momentum = trajectory(
        dataset='demo',
        df=df,
        stock_dim=1,
        state_space=len(tech_indicators) + 4,
        action_space=3,
        tech_indicator_list=tech_indicators,
        strategy_name='momentum',
        strategy_kwargs={'momentum_threshold': 0.03}
    )
    
    # 使用随机策略的轨迹
    traj_random = trajectory(
        dataset='demo',
        df=df,
        stock_dim=1,
        state_space=len(tech_indicators) + 4,
        action_space=3,
        tech_indicator_list=tech_indicators,
        strategy_name='random'
    )
    
    # 3. 比较不同策略在相同数据上的表现
    print("3. 比较不同策略的交易信号:")
    
    strategies_info = [
        ('移动平均线策略', traj_ma),
        ('动量策略', traj_momentum),
        ('随机策略', traj_random)
    ]
    
    # 在前10天测试不同策略
    for day in range(min(10, len(df)-1)):
        print(f"\n第 {day+1} 天:")
        
        for strategy_name, traj_obj in strategies_info:
            # 重置轨迹到指定天数
            traj_obj.day = day
            traj_obj.data = df.loc[day, :]
            if day > 0:
                traj_obj.last_day_memory = df.loc[day-1, :]
            
            # 使用策略强度参数 i=5 生成信号
            state, reward, terminal, action = traj_obj.step(i=5)
            
            # 解释动作
            if np.argmax(action) == 0:
                action_desc = "卖出"
            elif np.argmax(action) == 1:
                action_desc = "持有"
            else:
                action_desc = "买入"
                
            print(f"  {strategy_name}: {action_desc} {action}")
    
    print("\n" + "="*50 + "\n")
    
    # 4. 展示如何自定义策略参数
    print("4. 自定义策略参数:")
    
    # 创建具有不同敏感度的移动平均线策略
    conservative_ma = MovingAverageStrategy(strategy_id=0, threshold_multiplier=0.1)
    aggressive_ma = MovingAverageStrategy(strategy_id=10, threshold_multiplier=0.5)
    
    print(f"保守的移动平均线策略: {conservative_ma.get_strategy_info()}")
    print(f"激进的移动平均线策略: {aggressive_ma.get_strategy_info()}")
    
    # 测试同一天数据下的不同反应
    test_data = df.loc[50, :]  # 选择第50天的数据
    
    conservative_pos, conservative_action = conservative_ma.calculate_position_and_action(test_data)
    aggressive_pos, aggressive_action = aggressive_ma.calculate_position_and_action(test_data)
    
    print(f"\n在相同市场条件下:")
    print(f"保守策略: 持仓={conservative_pos}, 动作={conservative_action}")
    print(f"激进策略: 持仓={aggressive_pos}, 动作={aggressive_action}")


if __name__ == "__main__":
    strategy_demo() 