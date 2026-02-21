import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


def train_and_evaluate(X, y, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    preds = model.predict(X_vec)
    acc = accuracy_score(y, preds)

    model_path = os.path.join(output_dir, 'fake_news_model.pkl')
    vect_path = os.path.join(output_dir, 'vectorizer.pkl')
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vect_path)

    return {
        'model_path': model_path,
        'vectorizer_path': vect_path,
        'accuracy': acc,
    }


def load_model(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
