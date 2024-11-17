import gymnasium as gym
import numpy as np
import pytest

from pipeline.main import dca_approach, get_snp_data, load_config


@pytest.fixture
def snp_data():
    config = load_config()
    df = get_snp_data(config["data_path"])
    return df


def test_dca_approach(snp_data):
    df = snp_data.copy()

    # Starting conditions
    initial_balance = 10000
    net_worth_expected = initial_balance
    investment_per_day = 10
    shares_owned = 0  # Start with no shares owned

    days_invested = 0

    # Loop through each day in the data
    for index, row in df.iterrows():
        if initial_balance < investment_per_day:
            # If not enought funds buy 0 shares
            shares_bought_today = 0
        else:
            # Calculate how many shares you
            # can buy with your balance on that day
            shares_bought_today = investment_per_day / row["Close"]

        # Update the total number of shares owned
        shares_owned += shares_bought_today

        # Calculate the net worth by
        # multiplying the shares by the closing price on the current day
        net_worth_expected = shares_owned * row["Close"]

        # Track the number of days invested (to
        # simulate 10 years or however many days)
        days_invested += 1

        initial_balance -= investment_per_day

    net_worth_expected = round(net_worth_expected, 2)

    # Create the environment using gym.make
    env = gym.make("SP500TradingEnv-v0", df=df, render_mode="human")

    net_worth = dca_approach(env, df)
    assert np.isclose(
        net_worth_expected, net_worth
    ), f"Expected {net_worth_expected}, but got {net_worth}"
