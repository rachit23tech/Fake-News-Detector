import sys
import os
import joblib


def predict_text(text, model_path='models/fake_news_model.pkl', vect_path='models/vectorizer.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(vect_path):
        raise FileNotFoundError('Model or vectorizer not found in models/. Run training first.')

    model = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vec)[0].max()
    label = 'REAL' if pred == 1 else 'FAKE'
    return label, proba


def main():
    if len(sys.argv) < 2:
        print('Usage: python predict.py "Some news text to classify"')
        sys.exit(1)
    text = sys.argv[1]
    label, proba = predict_text(text)
    if proba is not None:
        print(f"Prediction: {label} (confidence: {proba:.3f})")
    else:
        print(f"Prediction: {label}")


if __name__ == '__main__':
    main()
