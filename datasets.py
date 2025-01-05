import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class RegressionData:
    platform: str
    dataframe: pd.DataFrame


def generate_dataset():
    np.random.seed(42)
    rng = np.random.default_rng()
    data = {
        "channel": np.random.choice(
            ["Google", "Facebook", "Instagram", "LinkedIn"], size=100
        ),
        "ad_spend": rng.integers(100, 1000, size=100),
        "impressions": rng.integers(1000, 10000, size=100),
        "clicks": rng.integers(10, 500, size=100),
        "leads": rng.integers(5, 200, size=100),
        "conversions": rng.integers(0, 50, size=100),
        "roi": rng.integers(200, 1000, size=100),
    }
    df = pd.DataFrame(data)
    df.to_csv("sample.csv")
    return df


def load_data():
    df = pd.read_csv("sample.csv")
    if df.empty:
        df = generate_dataset()
    return df


def get_regression_data(df: pd.DataFrame) -> list[RegressionData]:
    reg_data = []
    platforms = df["channel"].unique()
    for platform in platforms:
        platform_data = df[df["channel"] == platform]
        if isinstance(platform_data, pd.DataFrame):
            reg_data.append(RegressionData(platform=platform, dataframe=platform_data))
        else:
            print(f"unable to read data for platform: {platform}")

    return reg_data
