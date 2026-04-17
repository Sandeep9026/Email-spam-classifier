# Spam Email Classifier

An end-to-end NLP project that detects whether an email-like message is **Spam** or **Not Spam** using classical machine learning. The project is designed as a showcase-ready portfolio piece with clean structure, reproducible training, evaluation artifacts, and a polished Streamlit interface for inference.

## Features

- Text preprocessing with lowercase normalization, punctuation removal, stopword filtering, and token extraction
- TF-IDF vectorization with unigram and bigram support
- Three trained classifiers:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrices
- Imbalance handling with:
  - class weighting for Logistic Regression and SVM
  - minority class upsampling on the training split
- Automatic best-model selection based on evaluation performance
- Saved production artifact for inference using `joblib`
- Streamlit app for single-email and batch-email prediction
- Custom light and dark theme modes in the Streamlit UI
- Logging for traceable training and artifact generation

## Project Structure

```text
emailSpamClassifier/
+-- app.py
+-- train.py
+-- utils.py
+-- requirements.txt
+-- README.md
+-- data/
|   +-- spam.csv
|   +-- sms_spam_collection.zip            # downloaded UCI archive
|   `-- sms_spam_collection_raw/           # extracted original files
+-- models/
|   +-- spam_classifier.joblib             # generated after training
|   `-- metadata.json                      # generated after training
`-- reports/
    +-- detailed_metrics.json              # generated after training
    +-- metrics_summary.csv                # generated after training
    `-- figures/
        +-- class_distribution.png
        +-- confusion_matrix_*.png
        `-- model_comparison.png
```

## Dataset

The repository now bundles the real **SMS Spam Collection** dataset from the **UCI Machine Learning Repository**, converted into the project CSV format at [data/spam.csv](data/spam.csv).

- `label`: `ham` or `spam`
- `text`: raw message text

Dataset source:

- UCI Machine Learning Repository: SMS Spam Collection
- DOI: `10.24432/C5CC84`
- License: `CC BY 4.0`

Current bundled dataset size:

- `5,574` total messages
- `4,827` ham messages
- `747` spam messages

The original downloaded archive and extracted raw files are also kept in the `data/` folder for provenance.

## Why Precision and Recall Matter

- **Precision** is important because false positives can move legitimate emails into spam, causing users to miss important communication.
- **Recall** is important because false negatives allow malicious or unwanted emails to reach the inbox.
- In spam detection, an effective model needs a strong balance between both, so the project also compares **F1-score** when selecting the best model.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Models

```bash
python train.py
```

Optional arguments:

```bash
python train.py --data-path data/spam.csv --test-size 0.2
```

Training will:

- preprocess the dataset
- visualize class distribution
- upsample the minority class on the training split
- train and compare 3 ML models
- save confusion matrices and comparison charts
- persist the best model in `models/spam_classifier.joblib`

## Launch the Streamlit App

```bash
streamlit run app.py
```

The UI supports:

- single email prediction
- batch prediction with one email per line
- prediction probability display
- theme switching between light and dark mode
- model performance preview

## Model Performance

After training on the UCI SMS Spam Collection dataset, the best model is currently `svm`.

Latest saved metrics in [reports/metrics_summary.csv](reports/metrics_summary.csv):

- Accuracy: `98.39%`
- Precision: `94.56%`
- Recall: `93.29%`
- F1-score: `93.92%`

The Streamlit app reads the same artifacts and displays the selected best model automatically.

## Notes for Recruiters / Reviewers

- The codebase keeps business logic modular and easy to extend.
- The saved pipeline handles both vectorization and classification for deployment simplicity.
- The project can be upgraded further with richer datasets, hyperparameter tuning, and experiment tracking tools like MLflow.
