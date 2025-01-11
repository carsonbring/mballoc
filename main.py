import data
import roi_regression


def main():
    data.generate_dataset()
    roi_regression.xgboost_regression()


if __name__ == "__main__":
    main()
