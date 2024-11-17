import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register

register(
    id='SP500TradingEnv-v0',  # Unique identifier for the environment
    entry_point='src.rl_env:SP500TradingEnv',  # Module and class name
)

# class SP500TradingEnv(gym.Env):
#     metadata = {'render.modes': ['human']}

#     def __init__(self, df):
#         # load config values

#         # read S&P 500 data from a CSV file
#         self.df = df
#         total_steps = self.df.shape[0]
#         # define the observation space
#         self.observation_space = spaces.Box(
#             low=np.array([0, 0, 0, 0], dtype=np.float32),
#             high=np.array([100, 10000, total_steps, 10], dtype=np.float32),
#             shape=(4,),
#             dtype=np.float32)
#         # define the action space (0: sell, 1: buy)
#         self.action_space = spaces.Discrete(2)
#         # set the initial balance to $10,000
#         self.balance = 100
#         # Amount invested in the index
#         self.investment = 0
#         # set the initial net worth to $10,000
#         self.net_worth = self.balance
#         # set the initial step to 0
#         self.current_step = 0
#         # set the maximum number of steps to the number of data points
#         self.max_steps = total_steps

#     def step(self, action):
#         assert self.action_space.contains(action)
#         reward = 0
#         done = False
#         # sell
#         if action == 0:
#             if self.buy_price > 0:
#                 reward = self.df.loc[self.current_step, 'Close'] / \
#                     self.buy_price - 1
#                 self.balance += self.net_worth * reward
#                 self.buy_price = 0
#         # buy
#         elif action == 1:
#             if self.balance > 0:
#                 self.buy_price = self.df.loc[self.current_step, 'Close']
#                 self.balance -= self.buy_price
#         self.net_worth = self.balance + self.df.loc[
#           self.current_step, 'Close'
#         ]
#         self.current_step += 1

#         if self.current_step == self.max_steps:
#             done = True

#         obs = self._next_observation()

#         truncated = False
#         info = {}

#         # return obs, reward, done, {}
#         return obs, reward, done, truncated, info

#     def reset(self, seed=None, options={}):
#         # Seed the environment
#         super().reset(seed=seed)

#         self.balance = 10000
#         self.net_worth = self.balance
#         self.current_step = 0
#         self.buy_price = 0
#         return self._next_observation(), {}

#     def render(self, mode='human', close=False):
#         print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')

#     def _next_observation(self):
#         if self.current_step < 1:
#             obs = np.array([0, 0, 0, 0], dtype=np.float32)
#         else:
#             obs = np.array([
#                 self.balance,
#                 self.net_worth,
#                 self.buy_price,
#                 self.current_step
#             ], dtype=np.float32)

#         return obs


class SP500TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, render_mode=None):
        self.df = df
        total_steps = self.df.shape[0]
        self.render_mode = render_mode      
        # Define observation space
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([10000, 10000, total_steps, 10], dtype=np.float32),
            dtype=np.float32
        )

        # Define action space (0: hold, 1: invest £10)
        self.action_space = spaces.Discrete(2)

        # Initialize variables
        self.balance = 10000  # Starting cash balance
        self.investment = 0  # Amount invested in the index
        # Total net worth (balance + value of investments)
        self.net_worth = 10000
        self.current_step = 0  # Current step
        self.max_steps = len(self.df)  # Maximum steps in the environment

    def step(self, action):
        assert self.action_space.contains(action)

        # Get the current price of the index
        current_price = self.df.loc[self.current_step, 'Close']

        # Process the action
        if action == 1 and self.balance >= 10:
            # Invest £10 into the index
            self.balance -= 10
            # Units of the index purchased
            self.investment += 10 / current_price

        # Update investment value based on the new index price
        next_price = self.df.loc[
            self.current_step + 1, 'Close'
        ] if self.current_step + 1 < self.max_steps else current_price
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
        self.balance = 10000
        self.investment = 0
        self.net_worth = 10000
        self.current_step = 0

        return self._next_observation(), {}

    def render(self, mode='human', close=False):
        if self.render_mode == mode:
            print(
                f'Step: {self.current_step}, \
                Net Worth: {self.net_worth}, \
                Balance: {self.balance}, \
                Investment: {self.investment}'
            )

    def _next_observation(self):
        # Return observation of current environment state
        return np.array([
            self.balance,
            self.net_worth,
            self.investment,
            self.current_step
        ], dtype=np.float32)
