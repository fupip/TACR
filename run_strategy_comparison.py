#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略对比运行脚本

使用示例:
python run_strategy_comparison.py --dataset csi
"""

import subprocess
import sys
import time

def run_test(dataset='csi', test_strategy='model', seed=0, ma_strategy_id=1, ma_threshold=0.2):
    """
    运行单个测试
    """
    cmd = [
        sys.executable, 'test.py',
        '--dataset', dataset,
        '--test_strategy', test_strategy,
        '--seed', str(seed),
        '--ma_strategy_id', str(ma_strategy_id),
        '--ma_threshold', str(ma_threshold)
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return None
    
    return result.stdout

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='运行策略对比测试')
    parser.add_argument('--dataset', type=str, default='csi', 
                       help='数据集: kdd, hightech, dow, ndx, mdax, csi')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--ma_strategy_id', type=int, default=1, help='均线策略强度')
    parser.add_argument('--ma_threshold', type=float, default=0.2, help='均线策略阈值')
    
    args = parser.parse_args()
    
    strategies = ['model', 'ma', 'random']
    results = {}
    
    print("=" * 80)
    print(f"开始对比测试 - 数据集: {args.dataset}")
    print("=" * 80)
    
    for strategy in strategies:
        print(f"\n正在测试: {strategy}")
        print("-" * 40)
        
        output = run_test(
            dataset=args.dataset,
            test_strategy=strategy,
            seed=args.seed,
            ma_strategy_id=args.ma_strategy_id,
            ma_threshold=args.ma_threshold
        )
        
        if output:
            print(output)
            results[strategy] = output
        else:
            print(f"测试 {strategy} 失败")
            
        time.sleep(1)  # 给系统一点时间
    
    print("\n" + "=" * 80)
    print("对比测试完成")
    print("=" * 80)
    
    # 如果您想要解析结果并做更详细的对比，可以在这里添加代码
    
if __name__ == '__main__':
    main() 