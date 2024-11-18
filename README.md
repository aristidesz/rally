# SP500 Trading Environment

This project provides a custom trading environment for the S&P 500 index using the OpenAI Gym API. The environment simulates a simple **Dollar-Cost Averaging (DCA)** strategy where an agent invests £10 daily in the S&P 500. The project is designed for experimenting with reinforcement learning (RL) strategies to optimize investment decisions over time.

## Overview

The `SP500TradingEnv` class is an implementation of a custom gym environment for simulating investment in the S&P 500 index. It allows you to test different strategies in an investment environment that tracks balance, net worth, and investment value as the agent interacts with historical S&P 500 data.

### Features:
- **Observation Space**: The environment tracks the balance, net worth, investment amount, and current step.
- **Action Space**: There are two possible actions:
  - **0**: Hold (no investment)
  - **1**: Invest £10 in the index
- **Rewards**: The reward is the change in net worth after each step (investment and index price fluctuation).
- **Reset**: The environment can be reset to the starting conditions with an initial balance and net worth.
- **Rendering**: The environment can render the agent's current state in the terminal.
