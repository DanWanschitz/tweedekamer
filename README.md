# tweedekamer

Predicting whether Dutch parliamentary motions pass, using ~30,000 speeches from the Tweede Kamer. The pipeline runs RobBERT sentiment analysis on actual speech transcripts and feeds the results into XGBoost and Logistic Regression classifiers.

---

## Setup

### Prerequisites

Python 3.10+ and the packages below. Install with:

```bash
pip install -r requirements.txt
```

```text
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
transformers>=4.30.0
torch>=2.0.0
sentencepiece>=0.1.99
scipy>=1.10.0
```

NB3 runs RobBERT inference — a GPU significantly speeds this up (Google Colab T4 recommended), but CPU works too.

---

## Path configuration

Each notebook now contains a **PATH CONFIGURATION** cell at the top — it is the only cell you need to edit before running.

**Google Colab**  
The cell auto-detects Colab, mounts your Drive, and reads/writes everything under:
```
/content/drive/MyDrive/parliament_project/
```
Change that path in the cell if your Drive folder is named differently.

**Local (Jupyter / JupyterLab / VS Code)**  
Files are read from and written to a `data/` folder in your working directory. Place `speeches_modeling.csv` there before running NB1. The folder is created automatically if it doesn't exist.

```python
# The only line you may need to change for local runs:
DATA_DIR = Path("data")   # ← point this at wherever your CSVs live
```

No `drive.mount` calls appear outside this cell, so the notebooks no longer crash in non-Colab environments.

---

## Run order

```
NB1 → NB2 → NB3 → NB4 → NB5
```

### NB1 — `01_loading_eda.ipynb`
| | |
|---|---|
| **Input** | `speeches_modeling.csv` (place this in your `DATA_DIR` before running) |
| **Output** | `speeches_clean.csv` |

Loads raw parliamentary speeches, drops empty/short texts, produces EDA visualisations (speech length, target distribution, speaker/party breakdown, time-of-day patterns).

---

### NB2 — `02_preprocessing_features.ipynb`
| | |
|---|---|
| **Input** | `speeches_clean.csv` |
| **Output** | `speeches_features.csv`, `feature_cols.json`, `class_meta.json` |

Text cleaning, feature engineering (cyclic time features, party one-hots, interaction terms), train/val/test split, feature–target correlation plot.

---

### NB3 — `03_robbert_sentiment_only_with_save.ipynb`
| | |
|---|---|
| **Input** | `speeches_features.csv` |
| **Output** | `speeches_with_sentiment.csv` |

Runs [`DTAI-KULeuven/robbert-v2-dutch-sentiment`](https://huggingface.co/DTAI-KULeuven/robbert-v2-dutch-sentiment) on every speech. Long speeches (> 512 tokens) are split into overlapping chunks and scores are averaged. Adds `sentiment_score`, `sentiment_pos`, `sentiment_neg`, `sentiment_label` columns.

> `03_robbert_sentiment_tone.ipynb` is an extended variant that also runs zero-shot tone classification (`aggressive / mean / neutral / peaceful / kind / happy`) via `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`. Use it if you want the tone features for NB5's tone-over-day charts.

---

### NB4 — `04_predictive_modelling.ipynb`
| | |
|---|---|
| **Input** | `speeches_with_sentiment.csv`, `feature_cols.json`, `class_meta.json` |
| **Output** | model result JSONs → `DATA_DIR/outputs/` |

Trains and evaluates:
- **XGBoost** — structured + party + RobBERT sentiment features
- **Logistic Regression** (with CV + StandardScaler) — same feature set
- **RobBERT classifier** — fine-tuned directly on raw speech text

Produces confusion matrices, ROC curves, and feature importance plots.


## Week 8 bias auditing scripts

Run after NB4, **in numbered order**. All scripts expect `speeches_with_sentiment.csv` in `parliamentary_notebooks_speeches/` relative to the repo root and auto-detect the repo root, so they work whether you run them from inside `auditing/` or from the repo root.

```bash
python 01_bias_audit.py
python 02_bias_mitigation.py
python 03_indicator_mitigation.py
python 04_proxy_mechanism_audit.py
```

### `01_bias_audit.py` → `01_bias_audit_outputs/`

**Diagnose** — find where bias could exist before trying to fix it.

Checks missingness across columns and by subgroup, audits row loss across pipeline files, compares motion pass rates across groups, and trains three logistic regression models with progressively richer feature sets (temporal only → add speech/sentiment → add party) to detect omitted variable bias. Writes a full summary to `01_bias_audit_outputs/week8_bias_audit_summary.txt`.

### `02_bias_mitigation.py` → `02_bias_mitigation_outputs/`

**Fix** — implement concrete mitigations based on the audit findings.

Adds missingness indicator columns, removes fully empty columns, and compares four model variants ending with threshold tuning (finds the classification threshold that minimises the recall gap between accepted vs rejected motions). Note: visibility bias from informal bargaining and lobbying cannot be fixed with public parliamentary data — the summary acknowledges this explicitly.

### `03_indicator_mitigation.py` → `03_indicator_mitigation_outputs/`

**Reframe** — test whether the features are direct causes or symptoms of deeper political mechanisms.

Trains models that predict institutional variables (chair status, party, time bin) from visible debate features to check whether party patterns are embedded in them. Runs controlled outcome models comparing visible-only, institutional-only, and combined feature sets. Recommends framing the model as an *audit of visible indicators* rather than a claim that sentiment or speech duration directly causes outcomes.

### `04_proxy_mechanism_audit.py` → `04_proxy_mechanism_outputs/`

**Characterise** — show how visible variables are patterned by party, role, and topic.

Runs association tests (eta-squared and Cramér's V), builds a detailed party profile table, clusters parties into behavioural groups using K-Means, and reduces party profiles to principal components via PCA. Writes a summary to `04_proxy_mechanism_outputs/proxy_mechanism_summary.txt`.

---

## Repository structure

```
tweedekamer/
├── parliamentary_notebooks_speeches/   ← main 5-notebook pipeline
│   ├── 01_loading_eda.ipynb
│   ├── 02_preprocessing_features.ipynb
│   ├── 03_robbert_sentiment_only_with_save.ipynb
│   ├── 03_robbert_sentiment_tone.ipynb
│   ├── 04_predictive_modelling.ipynb
│   └── 05_dashboard_report.ipynb
├── auditing/                           ← week 8 bias auditing scripts
│   ├── 01_bias_audit.py
│   ├── 01_bias_audit_outputs/
│   ├── 02_bias_mitigation.py
│   ├── 02_bias_mitigation_outputs/
│   ├── 03_indicator_mitigation.py
│   ├── 03_indicator_mitigation_outputs/
│   ├── 04_proxy_mechanism_audit.py
│   └── 04_proxy_mechanism_outputs/
├── database/                           ← DB setup scripts
├── feature-engineering/                ← standalone feature experiments
├── scraping/                           ← Tweede Kamer API scraper
├── dashboard.py
└── speeches_clean.csv                  ← pre-cleaned snapshot
```
