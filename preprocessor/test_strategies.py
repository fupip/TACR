"""
Simple test script to verify the new strategy system structure
"""

import numpy as np
import pandas as pd

def test_strategy_import():
    """Test importing strategies from the new package structure"""
    try:
        from strategies import create_strategy, MovingAverageStrategy, MomentumStrategy, RandomStrategy
        from strategies.base_strategy import BaseStrategy
        print("✅ All strategy imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_strategy_creation():
    """Test creating different strategy instances"""
    try:
        from strategies import create_strategy
        
        # Test creating different strategies
        ma_strategy = create_strategy('moving_average', strategy_id=5)
        momentum_strategy = create_strategy('momentum', strategy_id=3)
        random_strategy = create_strategy('random', strategy_id=1)
        
        print("✅ Strategy creation successful")
        print(f"  - Moving Average: {ma_strategy.name}")
        print(f"  - Momentum: {momentum_strategy.name}")
        print(f"  - Random: {random_strategy.name}")
        return True
    except Exception as e:
        print(f"❌ Strategy creation failed: {e}")
        return False

def test_strategy_calculation():
    """Test strategy calculation with sample data"""
    try:
        from strategies import create_strategy
        
        # Create sample data
        sample_data = pd.Series({
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'close_5_sma': 101.0,
            'close_20_sma': 100.0,
            'close_60_sma': 99.0,
            'rsi_14': 55.0
        })
        
        # Test moving average strategy
        ma_strategy = create_strategy('moving_average', strategy_id=5)
        position, action = ma_strategy.calculate_position_and_action(sample_data)
        
        print("✅ Strategy calculation successful")
        print(f"  - Position: {position}")
        print(f"  - Action: {action}")
        print(f"  - Strategy info: {ma_strategy.get_strategy_info()}")
        return True
    except Exception as e:
        print(f"❌ Strategy calculation failed: {e}")
        return False

def test_trajectory_integration():
    """Test integration with trajectory class"""
    try:
        from process_traj import trajectory
        
        # Create minimal sample data
        dates = pd.date_range('2023-01-01', periods=10)
        data = {
            'date': dates,
            'tic': ['AAPL'] * 10,
            'open': np.random.uniform(150, 160, 10),
            'high': np.random.uniform(160, 170, 10),
            'low': np.random.uniform(140, 150, 10),
            'close': np.random.uniform(150, 160, 10),
            'close_5_sma': np.random.uniform(150, 160, 10),
            'close_20_sma': np.random.uniform(150, 160, 10),
            'close_60_sma': np.random.uniform(150, 160, 10),
        }
        df = pd.DataFrame(data)
        df.index = range(len(df))
        
        # Test trajectory with moving average strategy
        traj = trajectory(
            dataset='test',
            df=df,
            stock_dim=1,
            state_space=7,  # 4 OHLC + 3 technical indicators
            action_space=3,
            tech_indicator_list=['close_5_sma', 'close_20_sma', 'close_60_sma'],
            strategy_name='moving_average'
        )
        
        print("✅ Trajectory integration successful")
        print(f"  - Strategy: {traj.strategy_name}")
        return True
    except Exception as e:
        print(f"❌ Trajectory integration failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing New Strategy System Structure")
    print("=" * 50)
    
    tests = [
        test_strategy_import,
        test_strategy_creation,
        test_strategy_calculation,
        test_trajectory_integration
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        results.append(test())
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} tests passed!")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    main() 