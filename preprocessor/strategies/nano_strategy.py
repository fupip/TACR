import numpy as np
from collections import deque
from .base_strategy import BaseStrategy


class NanoStrategy(BaseStrategy):
    """
    Nano Strategy
    Based on nano trading strategy
    """
    
    def __init__(self, strategy_id=0, m = 0.001):
        super().__init__(strategy_id)
        self.m = m
        self.pos_state = 0
        self.pos_to_action = {
            -1: np.array([1.0, 0.0, 0.0]),
             0: np.array([0.0, 1.0, 0.0]),
             1: np.array([0.0, 0.0, 1.0]),
        }
        
    def calculate_position_and_action(self, data, last_day_data=None):
        """
        Calculate position and action based on nano trading strategy
        """
        
        close = float(data['close'])
        ma_60 = float(data['close_60_sma'])
        position = self.pos_state
        action = self.pos_to_action[self.pos_state]
        
        delta_from_ma60 = (close - ma_60)/ma_60
        eps = 1e-10
        # print("delta_from_ma60: ", delta_from_ma60)
        if delta_from_ma60 > self.m + eps:
            self.pos_state = 1
            position = self.pos_state
            action = self.pos_to_action[self.pos_state]
        elif delta_from_ma60 < -self.m - eps:
            self.pos_state = -1
            position = self.pos_state
            action = self.pos_to_action[self.pos_state]
        else:
            position = self.pos_state
            action = self.pos_to_action[self.pos_state]
        
        
        
        return position, action
    
    
    
    
    def get_strategy_info(self):
        return {
            'description': 'Nano Strategy',
            'm': self.m
        }