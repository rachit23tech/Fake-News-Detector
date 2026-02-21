# AI-powered Fake News Detector

Starter project scaffold for an AI-powered fake news detector.

Quick start

1. Create and activate a virtual environment (you already have `venv` in this workspace):

   ```powershell
   & .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Place dataset CSVs `Fake.csv` and `True.csv` in the repository root (or update paths in `src/data.py`).

3. Train a model:

   ```powershell
   python -m src.train --data-dir . --output-dir models
   ```

4. Predict from a saved model:

   ```powershell
   python predict.py "Some news text to classify"
   ```

Project layout

- `train.py` - existing simple training script (kept for reference)
- `predict.py` - small inference CLI
- `requirements.txt` - Python dependencies
- `src/` - recommended code modules
- `models/` - default output folder for trained model and vectorizer

Feel free to ask me to extend this into a web app, dataset download script, or Hugging Face transformer-based classifier.
