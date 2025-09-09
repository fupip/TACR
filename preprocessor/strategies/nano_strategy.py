import numpy as np
from collections import deque
from .base_strategy import BaseStrategy


class NanoStrategy(BaseStrategy):
    """
    Nano Strategy
    Based on nano trading strategy
    """
    
    def __init__(self, strategy_id=0, m = 0.002):
        super().__init__(strategy_id)
        self.m = m
        
    def calculate_position_and_action(self, data, last_day_data=None):
        """
        Calculate position and action based on nano trading strategy
        """
        
        close = float(data['close'])
        ma_60 = float(data['close_60_sma'])
        position = 0
        action = np.array([0.0, 1.0, 0.0])
        
        distance_from_ma60 = close - ma_60
        if distance_from_ma60 > self.m:
            position = 1
            action = np.array([0.0, 0.0, 1.0])
        elif distance_from_ma60 < -self.m:
            position = -1
            action = np.array([1.0, 0.0, 0.0])
        else:
            position = 0
            action = np.array([0.0, 1.0, 0.0])
        
        
        
        return position, action
    
    
    
    
    def get_strategy_info(self):
        return {
            'description': 'Nano Strategy',
            'm': self.m
        }