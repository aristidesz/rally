# from src.rl_env import SP500TradingEnv
import logging

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO

from src import rl_env

# Basic configuration
logging.basicConfig(
    level=logging.INFO,  # Log messages of INFO level and higher
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[logging.StreamHandler()],  # Output to console by default
)


def load_config(name="config.yaml"):
    # Define the path to your YAML file
    file_path = name

    # Load the YAML file
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def prepare_snp_df(df: pd.DataFrame) -> pd.DataFrame:
    # Keep desired columns
    cols = ["Date", "Close", "Open", "High", "Low"]
    df = df[cols]
    # Sort by date
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return df.sort_values("Date").reset_index(drop=True)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Impute missing values and 0s
    df = df.replace(0, np.nan)
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df


def get_snp_data(path):
    df = pd.read_csv(path)
    df = prepare_snp_df(df)
    df = validate_dataset(df)
    return df


def check_custom_env(my_custom_env):
    # Check if the environment is implemented correctly
    try:
        check_env(my_custom_env)
        print("The environment is correctly implemented!")
    except Exception as e:
        print(f"Environment check failed: {e}")


def dca_approach(
    df: pd.DataFrame, initial_balance: float = 10000
) -> np.float32:
    # Create the environment using gym.make
    env = gym.make(
        "SP500TradingEnv-v0",
        df=df,
        balance=initial_balance,
        render_mode="human",
    )
    # Test the environment
    _, _ = env.reset()
    env.render()
    for _ in range(df.shape[0]):
        env.step(1)
    env.render()
    # Access the original SP500TradingEnv (unwrapped version)
    original_env = env.unwrapped
    net_worth_dca = getattr(original_env, "net_worth", None)
    return round(net_worth_dca, 2)


def rl_approach(df: pd.DataFrame, initial_balance: float = 10000):
    env = gym.make(
        "SP500TradingEnv-v0",
        df=df,
        balance=initial_balance,
        render_mode="human",
    )
    # Test the environment
    _, _ = env.reset()
    env.render()
    # Instantiate RL model
    model = PPO("MlpPolicy", env, verbose=1)

    logging.info("Training the model...")
    model.learn(total_timesteps=10000)

    obs, _ = env.reset()
    for _ in range(df.shape[0]):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

    # Access the original SP500TradingEnv (unwrapped version)
    original_env = env.unwrapped
    net_worth_rl = getattr(original_env, "net_worth", None)
    return round(net_worth_rl, 2)


if __name__ == "__main__":

    config = load_config()
    # Load dataset
    df = get_snp_data(config["data_path"])

    # Ignore import not used
    rl_env.SP500TradingEnv

    # Create the environment using gym.make
    env = gym.make("SP500TradingEnv-v0", df=df, render_mode="human")

    # Initialise the environment
    check_custom_env(env.unwrapped)

    # Test the environment
    obs, info = env.reset()
    env.render()

    # Perform a baseline investment strategy based on dca
    net_worth_dca = dca_approach(df)
    logging.info(
        f"Net worth after Dollar " f"Cost Average (DCA): {net_worth_dca}"
    )

    # Perform a RL investment strategy based PPO
    net_worth_rl = rl_approach(df)
    logging.info(f"Net worth after RL strategy: {net_worth_rl}")
