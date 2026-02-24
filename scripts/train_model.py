import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models import MultiOutputModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    X = pd.read_csv(args.features)
    Y = pd.read_csv(args.targets)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_size, random_state=42
    )

    model = MultiOutputModel()
    model.fit(X_train, Y_train)
    metrics = model.evaluate(X_test, Y_test)
    print(metrics)


if __name__ == "__main__":
    main()
