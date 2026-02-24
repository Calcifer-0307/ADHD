import argparse
from src.preprocessing import run_preprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--categorical_cols", nargs="*", default=None)
    args = parser.parse_args()

    out = run_preprocessing(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        categorical_cols=args.categorical_cols,
    )
    print(str(out))


if __name__ == "__main__":
    main()
