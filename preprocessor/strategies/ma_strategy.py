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
        ma_types = ["1_5", "1_20", "1_60", "1_120",
                    "2_5_20","2_5_60","2_5_120","2_20_60","2_60_120",
                    "3_5_20_60","3_5_20_120","3_20_60_120"
                    ]
        self.ma_types = ma_types
        self.ma_type = self.ma_types[strategy_id]

    def calculate_position_and_action(self, data, last_day_data=None):
        """
        Calculate position and action based on moving averages
        
        Strategy Logic:
        - When 5-day MA > 20-day MA * (1 + threshold) and close > 60-day MA: go long
        - When 5-day MA < 20-day MA * (1 - threshold) and close < 60-day MA: go short  
        - Otherwise: hold cash
        """
        ma_fs = self.ma_type.split("_")
        ma_count = len(ma_fs)
        
        line_type = ma_fs[0]
        fast_ma = ma_fs[1]
        fast_ma_value = data[f'close_{fast_ma}_sma']
        if ma_count == 2:
            slow_ma = ma_fs[2]
            slow_ma_value = data[f'close_{slow_ma}_sma']
        elif ma_count == 3:
            slow_ma = ma_fs[2]
            limit_ma = ma_fs[3]
            limit_ma_value = data[f'close_{limit_ma}_sma']
        
        
        
        # Get moving average data
        
        
        
        
        current_close = data['close']
        
        # Calculate dynamic thresholds
        # buy_threshold = close_20_sma * (100 + self.strategy_id * self.threshold_multiplier) / 100.0
        # sell_threshold = close_20_sma * (100 - self.strategy_id * self.threshold_multiplier) / 100.0
        
        if line_type == "1":
            
            if current_close > fast_ma_data:
                position = 1.0
                action = np.array([0.0, 0.0, 1.0])  # Buy
            elif current_close < fast_ma_data:
                position = -1.0
                action = np.array([1.0, 0.0, 0.0])  # Sell
            else:
                position = 0.0
                action = np.array([0.0, 1.0, 0.0])  # Hold
        
        elif line_type == "2":
            if fast_ma_value > slow_ma_value:
                position = 1.0
                action = np.array([0.0, 0.0, 1.0])  # Buy
            elif fast_ma_value < slow_ma_value:
                position = -1.0
                action = np.array([1.0, 0.0, 0.0])  # Sell
            else:
                position = 0.0
                action = np.array([0.0, 1.0, 0.0])  # Hold
        
        elif line_type == "3":
            if fast_ma_value > slow_ma_value and current_close > limit_ma_value:
                position = 1.0
                action = np.array([0.0, 0.0, 1.0])  # Buy
            elif fast_ma_value < slow_ma_value and current_close < limit_ma_value:
                position = -1.0
                action = np.array([1.0, 0.0, 0.0])  # Sell
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