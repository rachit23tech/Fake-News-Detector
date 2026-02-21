import os
import pandas as pd


def load_dataset(data_dir='.', fake_file='Fake.csv', real_file='True.csv'):
    """Load and combine fake and real news CSV files.

    Returns X (pd.Series of text) and y (pd.Series of labels: 0 fake, 1 real).
    """
    fake_path = os.path.join(data_dir, fake_file)
    real_path = os.path.join(data_dir, real_file)

    if not os.path.exists(fake_path) or not os.path.exists(real_path):
        raise FileNotFoundError(f"Expected dataset files at {fake_path} and {real_path}")

    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)

    fake['label'] = 0
    real['label'] = 1

    df = pd.concat([fake, real], axis=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if 'text' not in df.columns:
        # try common alternatives
        for col in ['content', 'article', 'headline']:
            if col in df.columns:
                df = df.rename(columns={col: 'text'})
                break

    X = df['text']
    y = df['label']
    return X, y
