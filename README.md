# Spam Email Classifier

An end-to-end NLP project that classifies messages as **Spam** or **Not Spam** using classical machine learning, a real public dataset, evaluation reporting, saved model artifacts, and a polished Streamlit web app.

## Overview

This project is designed as a portfolio-ready machine learning application for showcasing:

- NLP preprocessing for short-text classification
- comparative modeling with multiple ML algorithms
- imbalanced-data handling
- evaluation with business-relevant metrics
- deployment-style inference through a web UI

The app uses the **UCI SMS Spam Collection** dataset and compares **Naive Bayes**, **Logistic Regression**, and **Support Vector Machine (SVM)** models before saving the best-performing pipeline for inference.

## Highlights

- Real public dataset with `5,574` labeled messages
- TF-IDF text vectorization with unigram and bigram features
- Three ML models trained and benchmarked
- Precision, recall, F1-score, and confusion matrix analysis
- Handling of class imbalance using upsampling and class weighting
- Saved trained model with `joblib`
- Streamlit UI for single-message and batch prediction
- Custom light and dark theme experience
- Generated reports and visualizations ready for presentation

## Best Model Performance

Current saved best model: `svm`

| Metric | Score |
|---|---:|
| Accuracy | 98.39% |
| Precision | 94.56% |
| Recall | 93.29% |
| F1-score | 93.92% |

Metrics source: [reports/metrics_summary.csv](reports/metrics_summary.csv)

## Tech Stack

- Python
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- Streamlit
- joblib

## Project Structure

```text
emailSpamClassifier/
+-- app.py
+-- train.py
+-- utils.py
+-- requirements.txt
+-- README.md
+-- .gitignore
+-- data/
|   +-- spam.csv
|   +-- sms_spam_collection.zip
|   `-- sms_spam_collection_raw/
+-- models/
|   +-- spam_classifier.joblib
|   `-- metadata.json
`-- reports/
    +-- detailed_metrics.json
    +-- metrics_summary.csv
    `-- figures/
        +-- class_distribution.png
        +-- confusion_matrix_*.png
        `-- model_comparison.png
```

## Dataset

This repository uses the **SMS Spam Collection** dataset from the **UCI Machine Learning Repository**.

- Source: UCI Machine Learning Repository
- Dataset: SMS Spam Collection
- DOI: `10.24432/C5CC84`
- License: `CC BY 4.0`

Dataset profile:

- `5,574` total messages
- `4,827` ham messages
- `747` spam messages

The repository keeps both the converted CSV used by the project and the original downloaded archive for provenance.

## How It Works

### 1. Preprocessing

- lowercase normalization
- punctuation cleanup
- stopword removal
- token filtering
- TF-IDF vectorization

### 2. Model Training

The training pipeline benchmarks:

- Naive Bayes
- Logistic Regression
- Support Vector Machine

### 3. Evaluation

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 4. Deployment

The best model is saved and loaded into the Streamlit app for real-time predictions.

## Why Precision And Recall Matter

- **Precision** matters because false positives can send legitimate messages to spam.
- **Recall** matters because false negatives allow spam to reach the inbox.
- A useful spam detector needs a strong balance of both, which is why **F1-score** is tracked for model selection.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

Run model training:

```bash
python train.py
```

Optional arguments:

```bash
python train.py --data-path data/spam.csv --test-size 0.2
```

Training will:

- preprocess the dataset
- generate the class-distribution chart
- handle class imbalance on the training split
- train all models
- save comparison metrics
- save confusion matrices
- persist the best model to `models/spam_classifier.joblib`

## Run The App

Launch the Streamlit UI:

```bash
streamlit run app.py
```

The app includes:

- single-message prediction
- batch-message prediction
- probability and confidence display
- theme switching
- model metrics view
- evaluation chart view

## Free Deployment

This project is ready to deploy on **Streamlit Community Cloud**.

Deployment settings:

- Repository: `Sandeep9026/Email-spam-classifier`
- Branch: `main`
- Main file path: `app.py`

Deployment steps:

1. Open `https://share.streamlit.io`
2. Sign in with GitHub
3. Click `Create app`
4. Select the repository and branch above
5. Set the main file path to `app.py`
6. Click `Deploy`

The repository already includes:

- a valid `requirements.txt`
- the trained model artifact
- generated report files
- the Streamlit entrypoint

## Output Artifacts

After training, the project generates:

- [models/spam_classifier.joblib](models/spam_classifier.joblib)
- [models/metadata.json](models/metadata.json)
- [reports/metrics_summary.csv](reports/metrics_summary.csv)
- [reports/detailed_metrics.json](reports/detailed_metrics.json)
- evaluation charts in [reports/figures](reports/figures)

## Recruiter Notes

This project is a strong portfolio example because it demonstrates:

- applied NLP workflow design
- model comparison and validation
- handling of imbalanced classes
- reproducible training artifacts
- practical UI integration for inference

## Future Improvements

- hyperparameter tuning with grid search
- probability calibration
- email-specific dataset expansion beyond SMS
- API deployment with FastAPI
- experiment tracking with MLflow
