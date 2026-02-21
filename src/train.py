"""Training CLI for fake-news-detector package.

Usage:
    python -m src.train --data-dir . --output-dir models
"""
import argparse
from .data import load_dataset
from .model import train_and_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='.', help='Directory containing Fake.csv and True.csv')
    parser.add_argument('--output-dir', default='models', help='Where to save model and vectorizer')
    args = parser.parse_args()

    X, y = load_dataset(data_dir=args.data_dir)
    result = train_and_evaluate(X, y, output_dir=args.output_dir)
    print(f"Saved model to: {result['model_path']}")
    print(f"Saved vectorizer to: {result['vectorizer_path']}")
    print(f"Training accuracy (on full data): {result['accuracy']:.4f}")


if __name__ == '__main__':
    main()
