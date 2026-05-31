Prerequisites
You need Python 3.10 and either a local environment or Google Colab (recommended — NB3 and NB4 contain drive.mount calls and were developed there).

Run Order
NB1 → NB2 → NB3* → NB4 → NB5
NB1 — 01_loading_eda.ipynb
Input: speeches_modeling.csv (raw data, must exist in the working directory)
Output: speeches_clean.csv
What it does: loads raw parliamentary speeches, drops empty texts, EDA visualisations, saves clean dataset.
NB2 — 02_preprocessing_features.ipynb
Input: speeches_clean.csv
Output: speeches_features.csv, feature_cols.json, class_meta.json
What it does: text cleaning, all feature engineering (temporal, cyclic, party dummies, interaction terms), train/val/test split.
NB3 — 03_robbert_sentiment_only_with_save__1_.ipynb ⚠️ Run on Google Colab
Input: speeches_features.csv
Output: speeches_with_sentiment.csv (saved to Google Drive at /content/drive/MyDrive/)
What it does: runs RobBERT sentiment inference on ~30,000 speeches with chunking for long texts, adds sentiment_score, sentiment_pos, sentiment_neg, sentiment_label columns.


NB4 — 04_predictive_modelling.ipynb ⚠️ Also Colab-dependent
Input: speeches_with_sentiment.csv (from Drive), feature_cols.json, class_meta.json
Output: model result JSONs saved to /content/drive/MyDrive/outputs/
What it does: trains and evaluates Logistic Regression (with CV) and XGBoost, produces feature importance plots, confusion matrices, ROC curves.

⚠️ Also has a Drive mount. Change output paths if running locally.

NB5 — 05_dashboard_report.ipynb
Input: speeches_final.csv (note: this name differs from speeches_with_sentiment.csv — rename or adjust the path)
What it does: final visualisation dashboard and report figures.

⚠️ NB5 reads speeches_final.csv but NB3 saves speeches_with_sentiment.csv. Either rename the file or change the path in NB5.

Week 8 scripts (run after NB4)
py -3.12 week8_bias_audit.py
py -3.12 week8_bias_mitigation.py
py -3.12 week8_indicator_mitigation.py
py -3.12 week8_proxy_mechanism_audit.py
All expect speeches_with_sentiment.csv in parliamentary_notebooks_speeches/ relative to the repo root.

requirements.txt
txt# Core data
pandas>=1.5.0
numpy>=1.23.0

# Visualisation
matplotlib>=3.6.0
seaborn>=0.12.0

# Machine learning
scikit-learn>=1.2.0
xgboost>=1.7.0

# NLP / RobBERT
transformers>=4.30.0
torch>=2.0.0
sentencepiece>=0.1.99

# Utilities
scipy>=1.10.0
Install with:
bashpip install -r requirements.txt# tweedekamer