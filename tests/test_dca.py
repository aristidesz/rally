import numpy as np
import pandas as pd
import pytest

from pipeline.main import dca_approach, load_config


def test_snp_data():
    config = load_config()
    assert config["data_path"] == "./data/sp500.csv"


@pytest.fixture
def sample_snp_df():
    data = {
        "Date": [
            "2012-02-06",
            "2012-02-07",
            "2012-02-08",
            "2012-02-09",
            "2012-02-10",
        ],
        "Close": [1344.33, 1347.05, 1349.96, 1351.95, 1342.64],
        "Open": [1344.32, 1344.33, 1347.04, 1349.97, 1351.21],
        "High": [1344.36, 1349.24, 1351.00, 1354.32, 1351.21],
        "Low": [1337.52, 1335.92, 1341.95, 1344.63, 1337.35],
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Convert the Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def test_dca_approach(sample_snp_df):
    # Starting conditions
    initial_balance = 50
    net_worth_expected = initial_balance
    investment_per_day = 10
    shares_owned = 0  # Start with no shares owned

    days_invested = 0

    # Loop through each day in the data
    for index, row in sample_snp_df.iterrows():
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

    net_worth = dca_approach(sample_snp_df, initial_balance=50)

    assert np.isclose(
        net_worth_expected, net_worth
    ), f"Expected {net_worth_expected}, but got {net_worth}"
