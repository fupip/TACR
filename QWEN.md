# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TACR (Transformer Actor-Critic with Regularization) is a deep reinforcement learning stock trading system that combines Transformer architecture with offline reinforcement learning algorithms. The system implements multiple trading strategies including TACR, CQL (Conservative Q-Learning), and IQL (Implicit Q-Learning) algorithms.

## Key Commands

### Environment Setup
```bash
# Create conda environment
conda env create -f conda_stock.yaml

# Activate environment
conda activate stock
```

### Data Preparation
```bash
# Prepare CSI dataset (other options: dow, kdd, hightech, ndx, mdax)
python create_data.py --dataset csi
```

### Model Training
```bash
# Train with TACR algorithm
python train.py --dataset csi --mode tacr --max_iters 10 --num_steps_per_iter 1000

# Train with CQL algorithm
python train.py --dataset csi --mode cql --max_iters 10 --num_steps_per_iter 1000

# Train with IQL algorithm
python train.py --dataset csi --mode iql --max_iters 10 --num_steps_per_iter 1000
```

### Model Testing
```bash
# Test trained model
python test.py --dataset csi --test_strategy model

# Test moving average strategy
python test.py --dataset csi --test_strategy ma --ma_strategy_id 5

# Test random strategy
python test.py --dataset csi --test_strategy random
```

### Strategy Comparison
```bash
# Run all strategy comparisons
python run_strategy_comparison.py --dataset csi
```

## Code Architecture

### Core Components

1. **Data Processing** (`create_data.py`, `preprocessor/`)
   - Downloads and processes stock market data
   - Computes technical indicators (SMA, RSI, MACD, etc.)
   - Generates trading trajectories for RL training

2. **Models** (`tac/models/`)
   - `transformer_actor.py`: Main Transformer-based actor model
   - `trajectory_gpt2.py`: Modified GPT-2 for trajectory modeling
   - Uses Hugging Face Transformers library

3. **Training** (`tac/training/`)
   - `seq_trainer.py`: Implements TACR, CQL, and IQL training algorithms
   - Contains separate training methods for each algorithm
   - Handles critic networks and target networks

4. **Environment** (`stock_env/`)
   - `env_portfolio.py`: Portfolio allocation trading environment
   - Implements realistic trading constraints and transaction costs

5. **Strategies** (`preprocessor/strategies/`)
   - `ma_strategy.py`: Moving average crossover strategies
   - `random_strategy.py`: Baseline random trading strategy
   - Extensible strategy framework

### Key Data Flow

1. **Data Preparation**: Raw stock data → Technical indicators → Trajectories
2. **Training**: Trajectories → Transformer Actor-Critic → Trained model
3. **Testing**: Trained model + Environment → Trading performance metrics

### Model Architecture

The core model uses a Transformer architecture where:
- States, actions, and rewards are embedded and concatenated
- Time embeddings are added to each modality
- GPT-2 based transformer processes the sequence
- Action predictions are made using linear layers
- Three training modes: TACR (behavior cloning + Q-learning), CQL (conservative Q-learning), IQL (implicit Q-learning)

## Development Notes

- The system supports multiple datasets (CSI, DOW, KDD, etc.)
- Training uses Weights & Biases for experiment tracking
- Implements proper learning rate scheduling and gradient clipping
- Uses deterministic seeds for reproducible results