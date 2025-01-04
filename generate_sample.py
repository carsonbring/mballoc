import numpy as np
import pandas as pd

# Generate synthetic ad spend data
np.random.seed(42)
rng = np.random.default_rng()
# Lifetime value of customer
data = {
    "channel": np.random.choice(
        ["Google", "Facebook", "Instagram", "LinkedIn"], size=100
    ),
    "ad_spend": rng.integers(100, 1000, size=100),
    "impressions": rng.integers(1000, 10000, size=100),
    "clicks": rng.integers(10, 500, size=100),
    "leads": rng.integers(5, 200, size=100),
    "conversions": rng.integers(0, 50, size=100),
}
df = pd.DataFrame(data)
df.to_csv("sample.csv")
