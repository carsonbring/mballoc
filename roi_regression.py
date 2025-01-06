import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import parse_version
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import datasets


def grad_boost_regression():
    all_data = datasets.load_data()
    platform_data = datasets.get_regression_data(all_data)
    all_figures = []
    results = []
    for reg_data in platform_data:
        p_df = reg_data.dataframe
        X = p_df[
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
        y = p_df["log_roi"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=13
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.1],
            "min_samples_split": [2, 5, 10],
        }
        grid_search = GridSearchCV(
            estimator=ensemble.GradientBoostingRegressor(
                loss="squared_error", random_state=42
            ),
            param_grid=param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid_search.fit(X_train_scaled, y_train)
        best_reg = grid_search.best_estimator_
        y_pred = best_reg.predict(X_test_scaled)
        y_pred_original = np.expm1(y_pred) - abs(p_df["roi"].min()) - 1
        y_test_original = np.expm1(y_test) - abs(p_df["roi"].min()) - 1
        mse_original = mean_squared_error(y_test_original, y_pred_original)
        rmse_original = np.sqrt(mse_original)
        mae_original = mean_absolute_error(y_test_original, y_pred_original)
        r2_original = r2_score(y_test_original, y_pred_original)

        results.append(
            {
                "platform": reg_data.platform,
                "mse_log_scale": mean_squared_error(y_test, y_pred),
                "mse_original_scale": mse_original,
                "rmse_original_scale": rmse_original,
                "mae_original_scale": mae_original,
                "r2_original_scale": r2_original,
            }
        )

        test_score = np.zeros((best_reg.n_estimators,), dtype=np.float64)

        for i, y_pred in enumerate(best_reg.staged_predict(X_test_scaled)):
            test_score[i] = mean_squared_error(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"{reg_data.platform}: Deviance")
        ax.plot(
            np.arange(best_reg.n_estimators) + 1,
            best_reg.train_score_,
            "b-",
            label="Training Set Deviance",
        )
        ax.plot(
            np.arange(best_reg.n_estimators) + 1,
            test_score,
            "r-",
            label="Test Set Deviance",
        )
        ax.legend(loc="upper right")
        ax.set_xlabel("Boosting Iterations")
        ax.set_ylabel("Deviance")
        fig.tight_layout()

        all_figures.append(fig)

    for fig in all_figures:
        fig.show()

    for result in results:
        print(f"Platform: {result['platform']}")
        print(f"  MSE (Log Scale): {result['mse_log_scale']:.4f}")
        print(f"  MSE (Original Scale): {result['mse_original_scale']:.4f}")
        print(f"  RMSE (Original Scale): {result['rmse_original_scale']:.4f}")
        print(f"  MAE (Original Scale): {result['mae_original_scale']:.4f}")
        print(f"  R² (Original Scale): {result['r2_original_scale']:.4f}\n")
