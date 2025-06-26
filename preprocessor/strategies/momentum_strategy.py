import numpy as np
from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy
    Based on price momentum trading strategy
    """
    
    def __init__(self, strategy_id=0, lookback_period=5, momentum_threshold=0.02):
        """
        Initialize Momentum Strategy
        
        Args:
            strategy_id: Strategy intensity parameter
            lookback_period: Lookback period for momentum calculation
            momentum_threshold: Momentum threshold for signal generation
        """
        super().__init__(strategy_id)
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
    
    def calculate_position_and_action(self, data, last_day_data=None):
        """
        Calculate position and action based on momentum
        
        Strategy Logic:
        - Calculate short-term momentum, follow trend if momentum is strong enough
        - Momentum = (current_price - N_days_ago_price) / N_days_ago_price
        """
        # Simplified implementation, actual usage needs more historical data
        current_close = data['close']
        
        # Use RSI indicator as momentum proxy if available
        if 'rsi_14' in data:
            rsi = data['rsi_14']
            adjusted_threshold = self.momentum_threshold * (1 + self.strategy_id * 0.1)
            
            if rsi > 70 + adjusted_threshold * 100:  # Overbought, prepare to sell
                position = -1.0
                action = np.array([1.0, 0.0, 0.0])
            elif rsi < 30 - adjusted_threshold * 100:  # Oversold, prepare to buy
                position = 1.0
                action = np.array([0.0, 0.0, 1.0])
            else:
                position = 0.0
                action = np.array([0.0, 1.0, 0.0])
        else:
            # If no RSI available, default to hold
            position = 0.0
            action = np.array([0.0, 1.0, 0.0])
            
        return position, action
    
    def get_strategy_info(self):
        """Get detailed strategy information"""
        info = super().get_strategy_info()
        info.update({
            'lookback_period': self.lookback_period,
            'momentum_threshold': self.momentum_threshold,
            'description': 'Momentum Strategy based on RSI'
        })
        return info 