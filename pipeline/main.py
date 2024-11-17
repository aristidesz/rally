# from src.rl_env import SP500TradingEnv
import pandas as pd
from gymnasium.utils.env_checker import check_env
import gymnasium as gym
from src import rl_env
import numpy as np


def prepare_snp_df(df: pd.DataFrame) -> pd.DataFrame:
    # Keep desired columns
    cols = ["Date", "Close", "Open", "High", "Low"]
    df = df[cols]
    # Sort by date
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return df.sort_values("Date")


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Impute missing values and 0s
    df = df.replace(0, np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df


def check_custom_env(my_custom_env):
    # Check if the environment is implemented correctly
    try:
        check_env(my_custom_env)
        print("The environment is correctly implemented!")
    except Exception as e:
        print(f"Environment check failed: {e}")


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("./data/sp500.csv")
    df = prepare_snp_df(df)
    df = validate_dataset(df)
    # Create the environment using gym.make
    env = gym.make('SP500TradingEnv-v0',
                   df=df,
                   render_mode="human")

    # Initialise the environment
    check_custom_env(env)

    # Test the environment
    obs, info = env.reset()
    env.render()

    _, reward, done, truncated, info = env.step(0)
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(0)
    env.render()
