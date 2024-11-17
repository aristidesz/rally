# from src.rl_env import SP500TradingEnv
import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from gymnasium.utils.env_checker import check_env

from src import rl_env


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


def dca_approach(env: gym.Env, df: pd.DataFrame) -> np.float32:

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


if __name__ == "__main__":

    config = load_config()
    # Load dataset
    df = get_snp_data(config["data_path"])

    # Ignore import not used
    rl_env.SP500TradingEnv

    # Create the environment using gym.make
    env = gym.make("SP500TradingEnv-v0", df=df, render_mode="human")

    # Initialise the environment
    check_custom_env(env)

    # Test the environment
    obs, info = env.reset()
    env.render()

    # Perform a baseline investment strategy based on dca
    net_worth_dca = dca_approach(env, df)
    print(f"Net worth after Dollar " f"Cost Average (DCA): {net_worth_dca}")
