import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class RegressionData:
    platform: str
    dataframe: pd.DataFrame


@dataclass
class TrainTestData:
    full_df: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame


def generate_dataset(num_weeks=1000):
    np.random.seed(42)
    rng = np.random.default_rng()

    channels = ["Google", "Facebook", "Instagram", "LinkedIn"]
    channel_params = {
        "Google": {"trend": 20, "seasonality_amplitude": 0.005},
        "Facebook": {"trend": 10, "seasonality_amplitude": 0.007},
        "Instagram": {"trend": 15, "seasonality_amplitude": 0.004},
        "LinkedIn": {"trend": 12, "seasonality_amplitude": 0.006},
    }
    data = []

    for channel in channels:
        ad_spend = rng.integers(500, 1000)
        impressions = rng.integers(1000, 5000)
        clicks = rng.integers(50, 300)
        leads = rng.integers(20, 150)
        conversions = rng.integers(5, 50)
        roi = rng.integers(200, 500)
        # add type
        # ROI by channel
        # roi overall campaign
        # SUbcampains are initiates
        # A/B testing - sometimes they will test 50 or 100 different ads
        # Timing regarding certain times of data
        # To a marketer, it's more anecdotal
        # Time of day
        # Return is traceable
        for week in range(num_weeks):
            trend_increase = channel_params[channel]["trend"]
            seasonality = 1 + channel_params[channel]["seasonality_amplitude"] * np.sin(
                2 * np.pi * (week % 52) / 52
            )  # Annual seasonality

            # Ad spend
            ad_spend += rng.normal(50, 5) + trend_increase
            ad_spend = max(ad_spend, 100)  # Ensure non-negative spend
            ad_spend = min(ad_spend, 100000)

            # Impressions
            impressions = int(ad_spend * rng.uniform(8, 12) + rng.normal(0, 50))
            impressions = max(impressions, 0)

            # Clicks
            ctr = rng.uniform(0.02, 0.05)
            clicks = int(impressions * ctr + rng.normal(0, 2))
            clicks = max(clicks, 0)

            # Leads
            lead_rate = rng.uniform(0.1, 0.3)
            leads = int(clicks * lead_rate + rng.normal(0, 1))
            leads = max(leads, 0)

            # Conversion
            conversion_rate = rng.uniform(0.05, 0.2)
            conversions = int(leads * conversion_rate + rng.normal(0, 0.5))
            conversions = max(conversions, 0)

            # ROI
            # This is where lifetime value of customer would be implemented
            revenue_per_conversion = rng.uniform(300, 1000)
            roi = (conversions * revenue_per_conversion) - ad_spend
            roi = max(roi, ad_spend * -0.3)
            roi = min(roi, 100000)

            # Timestamp
            timestamp = pd.Timestamp("2023-01-01") + pd.Timedelta(weeks=week)

            data.append(
                {
                    "channel": channel,
                    "timestamp": timestamp,
                    "adspend": round(ad_spend, 2),
                    "impressions": int(impressions),
                    "clicks": clicks,
                    "leads": leads,
                    "conversions": conversions,
                    "roi": round(roi, 2),
                }
            )

    df = pd.DataFrame(data)
    # Temporal features
    df["week_number"] = df["timestamp"].dt.isocalendar().week
    df["month"] = df["timestamp"].dt.month
    df["quarter"] = df["timestamp"].dt.quarter
    # Cyclical encoding
    df["week_sin"] = np.sin(2 * np.pi * df["week_number"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_number"] / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)

    # Lag features
    df["lag_roi_1"] = df.groupby("channel")["roi"].shift(1)
    df["lag_roi_2"] = df.groupby("channel")["roi"].shift(2)
    df = df.dropna()
    # Interaction features
    df["spend_clicks"] = df["adspend"] * df["clicks"]
    df["impression_leads"] = df["impressions"] * df["leads"]
    # Log roi (target) to stabilize variance
    df["log_roi"] = np.log1p(df["roi"] + abs(df["roi"].min()) + 1)

    df.to_csv("sample.csv", index=False)
    return df


def load_data():
    df = pd.read_csv("sample.csv")
    if df.empty:
        df = generate_dataset()
    return df


def load_test_train_data(platform: str, train_percent=0.9) -> TrainTestData:
    p_df = load_data()

    p_df = p_df.sort_values(by="timestamp")

    p_df = p_df[p_df["channel"] == platform]
    train_size = int(len(p_df) * train_percent)
    train_df = p_df.iloc[:train_size].copy()
    test_df = p_df.iloc[train_size:].copy()

    X_train = train_df[
        [
            "adspend",
            "impressions",
            "clicks",
            "leads",
            "conversions",
            "lag_roi_1",
            "lag_roi_2",
            "spend_clicks",
            "impression_leads",
            "week_sin",
            "week_cos",
            "month_sin",
            "month_cos",
            "quarter_sin",
            "quarter_cos",
        ]
    ]
    y_train = train_df["log_roi"]

    X_test = test_df[
        [
            "adspend",
            "impressions",
            "clicks",
            "leads",
            "conversions",
            "lag_roi_1",
            "lag_roi_2",
            "spend_clicks",
            "impression_leads",
            "week_sin",
            "week_cos",
            "month_sin",
            "month_cos",
            "quarter_sin",
            "quarter_cos",
        ]
    ]
    y_test = test_df["log_roi"]

    return TrainTestData(p_df, X_train, y_train, X_test, y_test)


def get_platforms(df: pd.DataFrame) -> list[str]:
    return list(df["channel"].unique())


# TODO: Deprecate
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
