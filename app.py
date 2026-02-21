from flask import Flask, render_template, request, redirect, url_for
import os
import joblib

APP_ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_ROOT, 'models', 'fake_news_model.pkl')
VECT_PATH = os.path.join(APP_ROOT, 'models', 'vectorizer.pkl')

app = Flask(__name__)


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
        raise FileNotFoundError('Model or vectorizer not found. Run training first.')
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    return model, vectorizer


model, vectorizer = load_artifacts()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Accept text from form or uploaded file
    text = ''
    if 'news_text' in request.form and request.form['news_text'].strip():
        text = request.form['news_text'].strip()
    elif 'news_file' in request.files and request.files['news_file']:
        f = request.files['news_file']
        try:
            content = f.read()
            # assume utf-8
            text = content.decode('utf-8').strip()
        except Exception:
            text = ''

    if not text:
        return redirect(url_for('index'))

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    label = 'REAL' if int(pred) == 1 else 'FAKE'
    confidence = None
    if hasattr(model, 'predict_proba'):
        confidence = float(model.predict_proba(vec)[0].max())
    elif hasattr(model, 'decision_function'):
        # convert decision_function to pseudo-probability via logistic
        try:
            from scipy.special import expit
            score = model.decision_function(vec)[0]
            confidence = float(expit(score))
        except Exception:
            confidence = None

    return render_template('result.html', text=text, label=label, confidence=confidence)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
