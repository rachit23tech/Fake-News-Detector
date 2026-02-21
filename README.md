Fake News Detection System

An end-to-end Fake News Detection system that classifies news articles as Real or Fake using Natural Language Processing (NLP) and Machine Learning. The project includes a trained ML model and a Flask-based web application for real-time inference.

Overview

This project implements a binary text classification pipeline to detect misinformation in news content. It covers the complete ML workflow, including data preprocessing, feature extraction, model training, evaluation, and deployment via a web interface.

Key Features

Binary classification of news articles (Real / Fake)

NLP-based text preprocessing

TF-IDF feature extraction

Logistic Regression classifier

Flask web interface for real-time predictions

Pre-trained model included

Tech Stack

Programming Language: Python

Machine Learning / NLP: Scikit-learn, TF-IDF

Backend: Flask

Frontend: HTML, CSS

Model Serialization: Joblib

Project Structure
fake-news-detector/
├── app.py
├── train.py
├── fake_news_model.pkl
├── vectorizer.pkl
├── Fake.csv
├── True.csv
├── templates/
│   └── index.html
└── README.md

Setup and Installation

Environment Setup
python -m venv venv
venv\Scripts\activate
python -m pip install numpy pandas scikit-learn nltk joblib flask
Dataset

The project uses the Fake and Real News Dataset from Kaggle:

Fake.csv

True.csv

Model Training

python train.py

This script performs:

Dataset loading and labeling

TF-IDF vectorization

Model training and evaluation

Saving trained artifacts

Running the Application

python app.py

Model Details

Algorithm: Logistic Regression

Vectorization: TF-IDF

Task: Binary text classification (Fake vs Real)

Evaluation Metric: Accuracy

Example

Input:
Breaking: Government announces free electricity for all citizens starting tomorrow

Output:
Fake
