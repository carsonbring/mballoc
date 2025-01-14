import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    accuracy_score,
)

from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import parse_version
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from skopt import BayesSearchCV
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import joblib
import data
import xgboost as xgb


def xgboost_regression():
    all_data = data.load_data()
    platform_data = data.get_regression_data(all_data)
    results = []
    all_figures = []
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
        z_scores = np.abs(stats.zscore(X_train))
        train_mask = (z_scores < 3).all(axis=1)
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask]

        z_scores_test = np.abs(stats.zscore(X_test))
        test_mask = (z_scores_test < 3).all(axis=1)
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask]

        xgb_train = xgb.DMatrix(X_train_clean, y_train_clean, enable_categorical=True)
        xgb_test = xgb.DMatrix(X_test_clean, y_test_clean, enable_categorical=True)

        param_grid = [
            {
                "max_depth": 3,
                "eta": 0.1,
                "subsample": 1.0,
            },
            {"max_depth": 4, "eta": 0.1, "subsample": 1.0},
            {"max_depth": 3, "eta": 0.01, "subsample": 0.8},
        ]

        best_score = float("inf")
        best_params = None
        best_iteration = 0

        for params in param_grid:
            params.update(
                {
                    "objective": "reg:squarederror",
                    "tree_method": "hist",
                    "eval_metric": "rmse",
                    "device": "cuda",
                }
            )

            cv_results = xgb.cv(
                params=params,
                dtrain=xgb_train,
                nfold=3,  # 3-fold cross validation
                num_boost_round=200,  # max number of boosting rounds
                early_stopping_rounds=10,
                seed=13,
            )
            mean_rmse = cv_results["test-rmse-mean"].min()
            best_round = cv_results["test-rmse-mean"].argmin()

            print(f"Params: {params} => CV RMSE: {mean_rmse:.4f} at round {best_round}")

            if mean_rmse < best_score:
                best_score = mean_rmse
                best_params = params.copy()
                best_iteration = best_round

        print("-" * 60)
        print(f"Best parameters found: {best_params}")
        print(f"Best RMSE: {best_score:.6f} (round {best_iteration})")

        evals_result = {}
        best_model = xgb.train(
            params=best_params,
            dtrain=xgb_train,
            num_boost_round=best_iteration + 1,
            evals=[(xgb_train, "train"), (xgb_test, "eval")],
            evals_result=evals_result,
            verbose_eval=False,
        )

        epochs = len(evals_result["train"]["rmse"])
        x_axis = range(0, epochs)
        plt.figure()
        plt.plot(x_axis, evals_result["train"]["rmse"], label="Train")
        plt.plot(x_axis, evals_result["eval"]["rmse"], label="Test")
        plt.legend()
        plt.xlabel("Boosting Round")
        plt.ylabel("RMSE")
        plt.title(f"Training vs Test RMSE for {reg_data.platform}")
        plt.show(block=True)

        preds = best_model.predict(xgb_test)
        mse = mean_squared_error(y_test_clean, preds)
        r2 = r2_score(y_test_clean, preds)

        y_pred_original = np.expm1(preds) - abs(p_df["roi"].min()) - 1
        y_test_original = np.expm1(y_test_clean) - abs(p_df["roi"].min()) - 1

        mse_original = mean_squared_error(y_test_original, y_pred_original)
        rmse_original = np.sqrt(mse_original)
        mae_original = mean_absolute_error(y_test_original, y_pred_original)
        mape_original = mean_absolute_percentage_error(y_test_original, y_pred_original)
        ev_original = explained_variance_score(y_test_original, y_pred_original)
        r2_original = r2_score(y_test_original, y_pred_original)

        results.append(
            {
                "platform": reg_data.platform,
                "r2": r2,
                "mse_log_scale": mse,
                "mse_original_scale": mse_original,
                "rmse_original_scale": rmse_original,
                "mae_original_scale": mae_original,
                "mape_original_scale": mape_original,
                "explained_variance_original_scale": ev_original,
                "r2_original_scale": r2_original,
            }
        )
        joblib.dump(best_model, f"best_model_{reg_data.platform}.joblib")

    for result in results:
        print(f"Platform: {result['platform']}")
        print(f"  MSE (Log Scale): {result['mse_log_scale']:.4f}")
        print(f"  MSE (Original Scale): {result['mse_original_scale']:.4f}")
        print(f"  RMSE (Original Scale): {result['rmse_original_scale']:.4f}")
        print(f"  MAE (Original Scale): {result['mae_original_scale']:.4f}")
        print(f"  MAPE (Original Scale): {result['mape_original_scale']:.4f}")
        print(
            f"  Explained Variance (Original Scale): {result['explained_variance_original_scale']:.4f}"
        )
        print(f"  R² (Original Scale): {result['r2_original_scale']:.4f}")
        print(f"  R²: {result['r2']:.4f}\n")


def grad_boost_regression():
    all_data = data.load_data()
    platform_data = data.get_regression_data(all_data)
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
        z_scores = np.abs(stats.zscore(X_train))
        train_mask = (z_scores < 3).all(axis=1)
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask]

        z_scores_test = np.abs(stats.zscore(X_test))
        test_mask = (z_scores_test < 3).all(axis=1)
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask]

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),  # Handle missing values
                ("scaler", StandardScaler()),
                (
                    "feature_selection",
                    SelectFromModel(
                        Lasso(alpha=0.1),
                        threshold="median",
                    ),
                ),
                (
                    "regressor",
                    ensemble.GradientBoostingRegressor(
                        loss="squared_error", random_state=42
                    ),
                ),
            ]
        )

        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train_clean)
        # X_test_scaled = scaler.transform(X_test_clean)
        #
        param_grid = {
            "regressor__n_estimators": [100, 200, 300, 400, 500],
            "regressor__max_depth": [3, 4, 5, 6, 7],
            "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "regressor__min_samples_split": [2, 5, 10, 15],
            "regressor__subsample": [0.6, 0.8, 1.0],
            "regressor__max_features": ["sqrt", "log2", None],
            # 'feature_selection__threshold': ['median', 'mean', 1.0, 0.5]  # Optional
        }

        # param_grid = {
        #     "n_estimators": [100, 200, 300],
        #     "max_depth": [3, 4, 5],
        #     "learning_rate": [0.05, 0.1],
        #     "min_samples_split": [2, 5, 10],
        # }
        bayes_search = BayesSearchCV(
            estimator=pipeline,
            n_iter=50,
            search_spaces=param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            random_state=42,
        )
        bayes_search.fit(X_train_clean, y_train_clean)

        best_reg = bayes_search.best_estimator_

        # selector = SelectFromModel(best_reg, prefit=True)
        # X_train_selected = selector.transform(X_train_scaled)
        # X_test_selected = selector.transform(X_test_scaled)

        # grid_search.fit(X_train_selected, y_train)

        y_pred = best_reg.predict(X_test_clean)

        y_pred_original = np.expm1(y_pred) - abs(p_df["roi"].min()) - 1
        y_test_original = np.expm1(y_test_clean) - abs(p_df["roi"].min()) - 1

        mse_original = mean_squared_error(y_test_original, y_pred_original)
        rmse_original = np.sqrt(mse_original)
        mae_original = mean_absolute_error(y_test_original, y_pred_original)
        mape_original = mean_absolute_percentage_error(y_test_original, y_pred_original)
        ev_original = explained_variance_score(y_test_original, y_pred_original)
        r2_original = r2_score(y_test_original, y_pred_original)

        results.append(
            {
                "platform": reg_data.platform,
                "mse_log_scale": mean_squared_error(y_test_clean, y_pred),
                "mse_original_scale": mse_original,
                "rmse_original_scale": rmse_original,
                "mae_original_scale": mae_original,
                "mape_original_scale": mape_original,
                "explained_variance_original_scale": ev_original,
                "r2_original_scale": r2_original,
            }
        )

        regressor = best_reg.named_steps["regressor"]
        X_test_transformed = best_reg.named_steps["feature_selection"].transform(
            best_reg.named_steps["scaler"].transform(
                best_reg.named_steps["imputer"].transform(X_test_clean)
            )
        )

        test_score = np.zeros((regressor.n_estimators,), dtype=np.float64)

        for i, y_pred_iter in enumerate(regressor.staged_predict(X_test_transformed)):
            test_score[i] = mean_squared_error(y_test_clean, y_pred_iter)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"{reg_data.platform}: Deviance")
        ax.plot(
            np.arange(regressor.n_estimators) + 1,
            regressor.train_score_,
            "b-",
            label="Training Set Deviance",
        )
        ax.plot(
            np.arange(regressor.n_estimators) + 1,
            test_score,
            "r-",
            label="Test Set Deviance",
        )
        ax.legend(loc="upper right")
        ax.set_xlabel("Boosting Iterations")
        ax.set_ylabel("Deviance")
        ax.grid(True)
        fig.tight_layout()

        all_figures.append(fig)
        joblib.dump(best_reg, f"best_model_{reg_data.platform}.joblib")

    for result in results:
        print(f"Platform: {result['platform']}")
        print(f"  MSE (Log Scale): {result['mse_log_scale']:.4f}")
        print(f"  MSE (Original Scale): {result['mse_original_scale']:.4f}")
        print(f"  RMSE (Original Scale): {result['rmse_original_scale']:.4f}")
        print(f"  MAE (Original Scale): {result['mae_original_scale']:.4f}")
        print(f"  MAPE (Original Scale): {result['mape_original_scale']:.4f}")
        print(
            f"  Explained Variance (Original Scale): {result['explained_variance_original_scale']:.4f}"
        )
        print(f"  R² (Original Scale): {result['r2_original_scale']:.4f}\n")

    plt.show()
