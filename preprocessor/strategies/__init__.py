"""
Trading Strategy Package
Provides a flexible framework for implementing different trading strategies
"""

from .base_strategy import BaseStrategy
from .ma_strategy import MovingAverageStrategy
from .momentum_strategy import MomentumStrategy
from .random_strategy import RandomStrategy
from .huric_strategy import HuricStrategy
from .nano_strategy import NanoStrategy


# Strategy factory function for easy strategy creation
def create_strategy(strategy_name, strategy_id=0, **kwargs):
    """
    Strategy factory function
    
    Args:
        strategy_name: Name of the strategy
        strategy_id: Strategy intensity parameter
        **kwargs: Other strategy-specific parameters
        
    Returns:
        BaseStrategy: Strategy instance
    """
    strategies = {
        'moving_average': MovingAverageStrategy,
        'momentum': MomentumStrategy,
        'random': RandomStrategy,
        'huric': HuricStrategy,
        'nano': NanoStrategy,
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name](strategy_id=strategy_id, **kwargs)


# Export all strategy classes and factory function
__all__ = [
    'BaseStrategy',
    'MovingAverageStrategy', 
    'MomentumStrategy',
    'RandomStrategy',
    'HuricStrategy',
    'NanoStrategy',
    'create_strategy'
]