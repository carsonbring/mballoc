import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import parse_version
import datasets


def grad_boost_regression():
    all_data = datasets.load_data()
    platform_data = datasets.get_regression_data(all_data)
    for p_df in platform_data:
        X = p_df[["adspend", "impressions", "clicks", "leads", "conversions"]]
        y = p_df["roi"]
