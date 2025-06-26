import numpy as np
from .base_strategy import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    """
    Moving Average Strategy
    Based on 5-day, 20-day, and 60-day moving average crossover signals
    """
    
    def __init__(self, strategy_id=0, threshold_multiplier=0.2):
        """
        Initialize Moving Average Strategy
        
        Args:
            strategy_id: Strategy intensity parameter to adjust buy/sell thresholds
            threshold_multiplier: Threshold multiplier to control strategy sensitivity
        """
        super().__init__(strategy_id)
        self.threshold_multiplier = threshold_multiplier
    
    def calculate_position_and_action(self, data, last_day_data=None):
        """
        Calculate position and action based on moving averages
        
        Strategy Logic:
        - When 5-day MA > 20-day MA * (1 + threshold) and close > 60-day MA: go long
        - When 5-day MA < 20-day MA * (1 - threshold) and close < 60-day MA: go short  
        - Otherwise: hold cash
        """
        # Get moving average data
        close_5_sma = data['close_5_sma']
        close_20_sma = data['close_20_sma']
        close_60_sma = data['close_60_sma']
        current_close = data['close']
        
        # Calculate dynamic thresholds
        buy_threshold = close_20_sma * (100 + self.strategy_id * self.threshold_multiplier) / 100.0
        sell_threshold = close_20_sma * (100 - self.strategy_id * self.threshold_multiplier) / 100.0
        
        # Long signal: 5-day MA crosses above 20-day MA and close above 60-day MA
        if close_5_sma > buy_threshold and current_close > close_60_sma:
            position = 1.0
            action = np.array([0.0, 0.0, 1.0])  # Buy
            
        # Short signal: 5-day MA crosses below 20-day MA and close below 60-day MA
        elif close_5_sma < sell_threshold and current_close < close_60_sma:
            position = -1.0
            action = np.array([1.0, 0.0, 0.0])  # Sell
            
        # Otherwise: hold cash
        else:
            position = 0.0
            action = np.array([0.0, 1.0, 0.0])  # Hold
            
        return position, action
    
    def get_strategy_info(self):
        """Get detailed strategy information"""
        info = super().get_strategy_info()
        info.update({
            'threshold_multiplier': self.threshold_multiplier,
            'description': 'Moving Average Crossover Strategy (5/20/60 SMA)'
        })
        return info 