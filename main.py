import datasets
import roi_regression


def main():
    datasets.generate_dataset()
    roi_regression.grad_boost_regression()


if __name__ == "__main__":
    main()
