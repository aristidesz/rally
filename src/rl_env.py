import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

register(
    id="SP500TradingEnv-v0",  # Unique identifier for the environment
    entry_point="src.rl_env:SP500TradingEnv",  # Module and class name
)


class SP500TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, balance=10000, render_mode=None):
        self.df = df
        total_steps = self.df.shape[0]
        self.render_mode = render_mode
        # Define observation space
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([total_steps, 100000, 10000, 100], dtype=np.float32),
            dtype=np.float32,
        )

        # Define action space (0: hold, 1: invest £10)
        self.action_space = spaces.Discrete(2)

        # Initialize variables
        self.initial_balance = balance
        self.balance = self.initial_balance  # Starting cash balance
        self.investment = 0  # Amount invested in the index
        # Total net worth (balance + value of investments)
        self.net_worth = self.initial_balance
        self.current_step = 0  # Current step
        self.max_steps = len(self.df)  # Maximum steps in the environment

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Get the current price of the index
        current_price = self.df.loc[self.current_step, "Close"]
        # Process the action
        if action == 1 and self.balance >= 10:
            # Invest £10 into the index
            self.balance -= 10
            # Units of the index purchased
            self.investment += 10 / current_price

        # Update investment value based on the new index price
        next_price = (
            self.df.loc[self.current_step + 1, "Close"]
            if self.current_step + 1 < self.max_steps
            else current_price
        )
        self.net_worth = self.balance + self.investment * next_price

        # Advance the step
        self.current_step += 1

        # Check if the episode is done
        done = self.current_step >= self.max_steps

        # Calculate reward (change in net worth)
        reward = self.net_worth - (
            self.balance + self.investment * current_price
        )

        # Generate observation
        obs = self._next_observation()

        # Return observation, reward, done, truncated, and info
        truncated = False
        info = {}
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)

        # Reset environment state
        self.balance = self.initial_balance
        self.investment = 0
        self.net_worth = self.initial_balance
        self.current_step = 0
        return self._next_observation(), {}

    def render(self, mode="human", close=False):
        if self.render_mode == mode:
            print(
                f"Step: {self.current_step}, "
                f"Net Worth: {self.net_worth}, "
                f"Balance: {self.balance}, "
                f"Investment: {self.investment}"
            )

    def _next_observation(self):
        # Return observation of current environment state
        return np.array(
            [self.current_step, self.net_worth, self.balance, self.investment],
            dtype=np.float32,
        )
