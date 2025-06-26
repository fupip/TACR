# Trading Strategy System

This module provides a flexible trading strategy framework for generating different types of trading trajectories.

## File Structure

```
preprocessor/
├── strategies/           # Strategy package
│   ├── __init__.py      # Package initialization and factory function
│   ├── base_strategy.py # Base strategy class
│   ├── ma_strategy.py   # Moving average strategy
│   ├── momentum_strategy.py  # Momentum strategy
│   └── random_strategy.py    # Random strategy
├── process_traj.py       # Modified trajectory generation class
├── strategy_example.py   # Usage examples
└── STRATEGY_README.md    # This document
```

## Core Concepts

### 1. Base Strategy Class (BaseStrategy)

All trading strategies inherit from the `BaseStrategy` abstract base class and must implement:

- `calculate_position_and_action(data, last_day_data=None)`: Calculate position and action
- `get_strategy_info()`: Return strategy information

### 2. Strategy Intensity Parameter (strategy_id)

- Controls the aggressiveness of the strategy
- Higher values make the strategy more aggressive
- Has different specific meanings in different strategies

### 3. Action Space

Each strategy returns an action as a 3-dimensional vector:
- `[1.0, 0.0, 0.0]`: Sell
- `[0.0, 1.0, 0.0]`: Hold
- `[0.0, 0.0, 1.0]`: Buy

## Built-in Strategies

### 1. Moving Average Strategy (MovingAverageStrategy)

Based on 5-day, 20-day, and 60-day moving average crossover signals.

**Parameters:**
- `strategy_id`: Strategy intensity (affects buy/sell thresholds)
- `threshold_multiplier`: Threshold multiplier (default 0.2)

**Logic:**
- When 5-day MA > 20-day MA × (1 + threshold) and close > 60-day MA: buy
- When 5-day MA < 20-day MA × (1 - threshold) and close < 60-day MA: sell
- Otherwise: hold cash

### 2. Momentum Strategy (MomentumStrategy)

Based on RSI indicator momentum strategy.

**Parameters:**
- `strategy_id`: Strategy intensity
- `lookback_period`: Lookback period (default 5)
- `momentum_threshold`: Momentum threshold (default 0.02)

**Logic:**
- When RSI > 70 + adjusted threshold: sell (overbought)
- When RSI < 30 - adjusted threshold: buy (oversold)
- Otherwise: hold

### 3. Random Strategy (RandomStrategy)

Random strategy for baseline testing.

**Parameters:**
- `strategy_id`: Affects random bias
- `random_seed`: Random seed (default 42)

## Usage

### 1. Basic Usage

```python
from strategies import create_strategy

# Create moving average strategy
strategy = create_strategy('moving_average', strategy_id=5)

# Calculate trading signals
position, action = strategy.calculate_position_and_action(data)
```

### 2. Usage in Trajectory Generation

```python
from process_traj import trajectory

# Create trajectory using specific strategy
traj = trajectory(
    dataset='csi',
    df=df,
    stock_dim=30,
    state_space=360,
    action_space=3,
    tech_indicator_list=['close_5_sma', 'close_20_sma', 'close_60_sma'],
    strategy_name='moving_average',  # Specify strategy
    strategy_kwargs={'threshold_multiplier': 0.3}  # Strategy parameters
)
```

### 3. Creating Custom Strategies

```python
from strategies.base_strategy import BaseStrategy
import numpy as np

class MyCustomStrategy(BaseStrategy):
    def __init__(self, strategy_id=0, custom_param=1.0):
        super().__init__(strategy_id)
        self.custom_param = custom_param
    
    def calculate_position_and_action(self, data, last_day_data=None):
        # Implement your strategy logic
        if data['close'] > data['close_20_sma']:
            return 1.0, np.array([0.0, 0.0, 1.0])  # Buy
        else:
            return -1.0, np.array([1.0, 0.0, 0.0])  # Sell
```

### 4. Batch Testing Different Strategies

```python
strategies = ['moving_average', 'momentum', 'random']
results = {}

for strategy_name in strategies:
    traj = trajectory(
        dataset='csi',
        df=df,
        stock_dim=30,
        state_space=360,
        action_space=3,
        tech_indicator_list=tech_indicators,
        strategy_name=strategy_name
    )
    
    # Run trajectory generation and collect results
    results[strategy_name] = run_trajectory(traj)
```

## Extension Guide

### Adding New Strategies

1. Inherit from `BaseStrategy` class
2. Implement `calculate_position_and_action` method
3. Override `get_strategy_info` method (optional)
4. Register new strategy in `create_strategy` function

### Strategy Parameter Tuning

- Test strategy intensity using different `strategy_id` values
- Adjust strategy-specific parameters (e.g., `threshold_multiplier`)
- Conduct backtesting and performance evaluation

### Data Requirements

Technical indicators required by strategies should be pre-calculated in the data:
- Moving averages: `close_5_sma`, `close_20_sma`, `close_60_sma`
- Momentum indicators: `rsi_14`, `macd`, etc.
- Basic data: `open`, `high`, `low`, `close`

## Running Examples

```bash
cd preprocessor
python strategy_example.py
```

This will demonstrate:
- Creating and configuring different strategies
- Comparing strategy performance on the same data
- Effects of custom strategy parameters

## Best Practices

1. **Strategy Testing**: Conduct thorough backtesting before using new strategies
2. **Parameter Optimization**: Use grid search and other methods to optimize strategy parameters
3. **Strategy Combination**: Consider combining multiple strategies
4. **Risk Management**: Include stop-loss and position management logic in strategies
5. **Performance Monitoring**: Regularly evaluate actual strategy performance

## Important Notes

- Signals generated by strategies are for research and training purposes only, not investment advice
- Consider trading costs, slippage, and other factors in actual applications
- Recommend testing strategy robustness under various market conditions 