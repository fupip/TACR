"""
轨迹数据检查脚本
用于检查生成的轨迹数据文件的结构、shape和内容

使用方法:
1. 检查所有轨迹文件:
   python data_check.py

2. 在代码中使用:
   from data_check import check_csi_data, main
   
   # 检查CSI数据
   check_csi_data()
   
   # 检查特定模式的文件
   main("csi*.pkl")      # 检查所有csi开头的文件
   main("*_traj.pkl")    # 检查所有_traj.pkl结尾的文件
   main("specific.pkl")  # 检查特定文件
"""

import pickle
import numpy as np
import pandas as pd
import os
from pathlib import Path


def check_trajectory_data(traj_file):
    """
    检查单个轨迹文件的数据结构
    
    Args:
        traj_file: 轨迹文件路径
    """
    print(f"\n{'='*60}")
    print(f"检查轨迹文件: {traj_file}")
    print(f"{'='*60}")
    
    # 检查文件是否存在
    if not os.path.exists(traj_file):
        print(f"❌ 文件不存在: {traj_file}")
        return
    
    try:
        # 加载轨迹数据
        with open(traj_file, 'rb') as f:
            trajectories = pickle.load(f)
        
        print(f"✅ 成功加载轨迹数据")
        print(f"📊 轨迹数量: {len(trajectories)}")
        
        if len(trajectories) == 0:
            print("⚠️  轨迹数据为空")
            return
        
        # 检查第一个轨迹的结构
        first_traj = trajectories[0]
        print(f"\n📋 轨迹数据结构:")
        print(f"   数据键: {list(first_traj.keys())}")
        print(f"   轨迹类型: {type(first_traj)}")
        
        # 检查每个组件的详细信息
        print(f"\n📊 各组件详细信息:")
        for key, value in first_traj.items():
            if isinstance(value, np.ndarray):
                print(f"   {key:12} - Shape: {str(value.shape):15} | Dtype: {value.dtype}")
                if value.ndim == 1 and len(value) > 0:
                    print(f"   {' '*12}   范围: [{value.min():.6f}, {value.max():.6f}]")
                elif value.ndim == 2 and len(value) > 0:
                    print(f"   {' '*12}   每步维度: {value.shape[1]} | 总步数: {value.shape[0]}")
                    if value.shape[1] <= 10:  # 如果维度不太大，显示一些样本
                        print(f"   {' '*12}   前3步数据:")
                        for i in range(min(3, len(value))):
                            print(f"   {' '*12}     步骤{i}: {value[i]}")
            else:
                print(f"   {key:12} - Type: {type(value)} | Value: {value}")
        
        # 检查所有轨迹的一致性
        print(f"\n🔍 轨迹一致性检查:")
        shapes_consistent = True
        keys_consistent = True
        
        for i, traj in enumerate(trajectories):
            # 检查键的一致性
            if set(traj.keys()) != set(first_traj.keys()):
                print(f"   ❌ 轨迹 {i} 的键与第一个轨迹不一致")
                keys_consistent = False
            
            # 检查shape的一致性
            for key in first_traj.keys():
                if isinstance(first_traj[key], np.ndarray) and isinstance(traj[key], np.ndarray):
                    if first_traj[key].shape[1:] != traj[key].shape[1:]:  # 除了时间维度外的shape
                        print(f"   ❌ 轨迹 {i} 的 {key} shape 不一致: {traj[key].shape} vs {first_traj[key].shape}")
                        shapes_consistent = False
        
        if shapes_consistent:
            print("   ✅ 所有轨迹的数据shape一致")
        if keys_consistent:
            print("   ✅ 所有轨迹的数据键一致")
        
        # 统计信息
        print(f"\n📈 统计信息:")
        traj_lengths = [len(traj['observations']) for traj in trajectories]
        total_rewards = [np.sum(traj['rewards']) for traj in trajectories]
        
        print(f"   轨迹长度 - 平均: {np.mean(traj_lengths):.1f} | 最小: {np.min(traj_lengths)} | 最大: {np.max(traj_lengths)}")
        print(f"   总奖励   - 平均: {np.mean(total_rewards):.6f} | 最小: {np.min(total_rewards):.6f} | 最大: {np.max(total_rewards):.6f}")
        
        # 检查动作分布
        if 'actions' in first_traj:
            all_actions = np.concatenate([traj['actions'] for traj in trajectories])
            print(f"\n🎯 动作分布:")
            print(f"   动作维度: {all_actions.shape[1]}")
            print(f"   总动作数: {len(all_actions)}")
            print(f"   动作范围: [{all_actions.min():.6f}, {all_actions.max():.6f}]")
            
            # 假设动作是 [sell, hold, buy] 的概率分布
            if all_actions.shape[1] == 3:
                action_names = ['Sell', 'Hold', 'Buy']
                for i, name in enumerate(action_names):
                    action_ratio = np.mean(all_actions[:, i])
                    print(f"   {name:4} 平均比例: {action_ratio:.4f}")
            else:
                # 对于其他维度的动作，显示统计信息
                print(f"   动作统计 - 均值: {all_actions.mean():.6f} | 标准差: {all_actions.std():.6f}")
        
        # 检查观测数据
        if 'observations' in first_traj:
            all_obs = np.concatenate([traj['observations'] for traj in trajectories])
            print(f"\n👁️  观测数据:")
            print(f"   观测维度: {all_obs.shape[1]}")
            print(f"   总观测数: {len(all_obs)}")
            print(f"   数值范围: [{all_obs.min():.6f}, {all_obs.max():.6f}]")
            
            # 检查是否有异常值
            nan_count = np.isnan(all_obs).sum()
            inf_count = np.isinf(all_obs).sum()
            if nan_count > 0:
                print(f"   ⚠️  发现 {nan_count} 个 NaN 值")
            if inf_count > 0:
                print(f"   ⚠️  发现 {inf_count} 个 Inf 值")
            if nan_count == 0 and inf_count == 0:
                print(f"   ✅ 无异常值 (NaN/Inf)")
        
    except Exception as e:
        print(f"❌ 加载轨迹数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()


def main(file_pattern="*.pkl"):
    """
    主函数：检查轨迹数据文件
    
    Args:
        file_pattern: 文件匹配模式，支持通配符
                     例如: "*.pkl", "csi*.pkl", "kdd_traj.pkl"
    """
    print("🔍 轨迹数据检查工具")
    print("="*60)
    
    # 轨迹文件目录
    traj_dir = Path("trajectory")
    
    if not traj_dir.exists():
        print(f"❌ 轨迹目录不存在: {traj_dir}")
        return
    
    # 查找匹配的 .pkl 文件
    traj_files = list(traj_dir.glob(file_pattern))
    
    if not traj_files:
        print(f"❌ 在 {traj_dir} 目录下未找到匹配 '{file_pattern}' 的轨迹文件")
        return
    
    print(f"📁 找到 {len(traj_files)} 个匹配 '{file_pattern}' 的轨迹文件:")
    for f in traj_files:
        print(f"   - {f}")
    
    # 逐个检查每个文件
    for traj_file in traj_files:
        check_trajectory_data(traj_file)
    
    print(f"\n{'='*60}")
    print("✅ 数据检查完成!")


def check_csi_data():
    """专门检查CSI数据集的轨迹文件"""
    print("🏦 检查CSI数据集轨迹文件")
    main("csi*.pkl")


def check_all_data():
    """检查所有轨迹文件"""
    print("📊 检查所有轨迹文件")
    main("*.pkl")


if __name__ == "__main__":

    main("csi_traj.pkl")  # 检查特定文件

