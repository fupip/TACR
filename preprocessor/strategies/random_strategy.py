import numpy as np
from .base_strategy import BaseStrategy


class RandomStrategy(BaseStrategy):
    """
    Random Strategy
    Used for testing and baseline comparison
    """
    
    def __init__(self, strategy_id=0, random_seed=42):
        """
        Initialize Random Strategy
        
        Args:
            strategy_id: Strategy intensity parameter
            random_seed: Random seed for reproducibility
        """
        super().__init__(strategy_id)
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def calculate_position_and_action(self, data, last_day_data=None):
        """
        Randomly generate position and action
        """
        # Strategy intensity affects randomness: higher strategy_id tends toward extreme positions
        bias = self.strategy_id * 0.1
        
        rand_val = np.random.random()
        
        if rand_val < 0.33 - bias:
            position = -1.0
            action = np.array([1.0, 0.0, 0.0])  # Sell
        elif rand_val > 0.67 + bias:
            position = 1.0
            action = np.array([0.0, 0.0, 1.0])  # Buy
        else:
            position = 0.0
            action = np.array([0.0, 1.0, 0.0])  # Hold
            
        return position, action
    
    def get_strategy_info(self):
        """Get detailed strategy information"""
        info = super().get_strategy_info()
        info.update({
            'random_seed': self.random_seed,
            'description': 'Random Strategy for baseline comparison'
        })
        return info 