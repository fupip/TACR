from abc import ABC, abstractmethod
import numpy as np


class BaseStrategy(ABC):
    """
    Base class for trading strategies
    All trading strategies should inherit from this class and implement the required methods
    """
    
    def __init__(self, strategy_id=0):
        """
        Initialize strategy
        
        Args:
            strategy_id: Strategy intensity parameter to adjust aggressiveness
        """
        self.strategy_id = strategy_id
        self.name = self.__class__.__name__
    
    @abstractmethod
    def calculate_position_and_action(self, data, last_day_data=None):
        """
        Calculate position and action based on current data
        
        Args:
            data: Current trading day data (pandas Series)
            last_day_data: Previous trading day data (pandas Series, optional)
            
        Returns:
            tuple: (position, action)
                - position: Position direction (-1: short, 0: neutral, 1: long)
                - action: Action vector numpy array([sell_ratio, hold_ratio, buy_ratio])
        """
        pass
    
    def get_strategy_info(self):
        """
        Get strategy information
        
        Returns:
            dict: Dictionary containing strategy name and parameters
        """
        return {
            'name': self.name,
            'strategy_id': self.strategy_id
        } 