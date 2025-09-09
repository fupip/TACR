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
import os
from pathlib import Path


def check_trajectory_data(traj_file):
    """
    检查单个轨迹文件的数据结构（精简输出）
    
    Args:
        traj_file: 轨迹文件路径
    """
    print(f"\n== {traj_file}")
    
    if not os.path.exists(traj_file):
        print(f"ERROR: File not found")
        return
    
    try:
        with open(traj_file, 'rb') as f:
            trajectories = pickle.load(f)
        
        num_traj = len(trajectories) if isinstance(trajectories, list) else 0
        print(f"count={num_traj}")
        if num_traj == 0:
            print("empty")
            return
        
        first_traj = trajectories[0]
        keys = list(first_traj.keys())
        print(f"keys={', '.join(keys)}")
        
        # 简要 shape 信息
        def safe_shape(d, k):
            v = d.get(k)
            return tuple(v.shape) if isinstance(v, np.ndarray) else None
        obs_shape = safe_shape(first_traj, 'observations')
        act_shape = safe_shape(first_traj, 'actions')
        rew_shape = safe_shape(first_traj, 'rewards')
        done_shape = safe_shape(first_traj, 'dones')
        print(f"shapes: obs={obs_shape} actions={act_shape} rewards={rew_shape} dones={done_shape}")
        
        # 一致性检查（仅输出结论）
        shapes_consistent = True
        keys_consistent = True
        for traj in trajectories:
            if set(traj.keys()) != set(first_traj.keys()):
                keys_consistent = False
                break
        if obs_shape is not None:
            for traj in trajectories:
                arr = traj.get('observations')
                if isinstance(arr, np.ndarray) and arr.ndim == len(obs_shape):
                    if arr.shape[1:] != obs_shape[1:]:
                        shapes_consistent = False
                        break
        print(f"consistent: keys={keys_consistent} shapes={shapes_consistent}")
        
        # 统计
        traj_lengths = [len(traj['observations']) for traj in trajectories if isinstance(traj.get('observations'), np.ndarray)]
        total_rewards = [np.sum(traj['rewards']) for traj in trajectories if isinstance(traj.get('rewards'), np.ndarray)]
        if traj_lengths:
            print(f"len(steps): mean={np.mean(traj_lengths):.1f} min={np.min(traj_lengths)} max={np.max(traj_lengths)}")
        if total_rewards:
            print(f"reward(sum): mean={np.mean(total_rewards):.6f} min={np.min(total_rewards):.6f} max={np.max(total_rewards):.6f}")
        
        # 观测摘要
        if 'observations' in first_traj and isinstance(first_traj['observations'], np.ndarray):
            all_obs = np.concatenate([traj['observations'] for traj in trajectories if isinstance(traj.get('observations'), np.ndarray)])
            obs_dim = all_obs.shape[1] if all_obs.ndim == 2 else None
            nan_count = int(np.isnan(all_obs).sum())
            inf_count = int(np.isinf(all_obs).sum())
            vmin = float(np.nanmin(all_obs)) if all_obs.size else float('nan')
            vmax = float(np.nanmax(all_obs)) if all_obs.size else float('nan')
            print(f"obs: dim={obs_dim} total={len(all_obs)} range=[{vmin:.6f},{vmax:.6f}] nan={nan_count} inf={inf_count}")
        
        # 动作摘要
        if 'actions' in first_traj and isinstance(first_traj['actions'], np.ndarray):
            all_actions = np.concatenate([traj['actions'] for traj in trajectories if isinstance(traj.get('actions'), np.ndarray)])
            adim = all_actions.shape[1] if all_actions.ndim == 2 else None
            if adim == 3:
                means = np.mean(all_actions, axis=0)
                # 基于 argmax 的 sell/hold/buy 频次比例
                preds = np.argmax(all_actions, axis=1)
                counts = np.bincount(preds, minlength=3).astype(float)
                ratios = counts / len(all_actions)
                print(f"actions: dim=3 mean=[{means[0]:.4f},{means[1]:.4f},{means[2]:.4f}] ratio=[{ratios[0]:.4f},{ratios[1]:.4f},{ratios[2]:.4f}] n={len(all_actions)}")
            else:
                print(f"actions: dim={adim} mean={all_actions.mean():.6f} std={all_actions.std():.6f} n={len(all_actions)}")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def main(file_pattern="*.pkl"):
    """
    主函数：检查轨迹数据文件（精简输出）
    
    Args:
        file_pattern: 文件匹配模式，支持通配符，如 "*.pkl", "csi*.pkl"
    """
    traj_dir = Path("trajectory")
    if not traj_dir.exists():
        print(f"ERROR: Dir not found -> {traj_dir}")
        return
    
    traj_files = list(traj_dir.glob(file_pattern))
    if not traj_files:
        print(f"No files matched '{file_pattern}' in {traj_dir}")
        return
    
    print(f"DataCheck pattern='{file_pattern}' files={len(traj_files)}")
    for f in traj_files:
        print(f" - {f}")
    
    for traj_file in traj_files:
        check_trajectory_data(traj_file)
    
    print("\nDone.")


def check_csi_data():
    """专门检查CSI数据集的轨迹文件"""
    print("🏦 检查CSI数据集轨迹文件")
    main("csi*.pkl")


def check_all_data():
    """检查所有轨迹文件"""
    print("📊 检查所有轨迹文件")
    main("*.pkl")


if __name__ == "__main__":

    main("csi_train_traj.pkl")  # 检查特定文件

