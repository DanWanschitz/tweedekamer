# tweedekamer

Predicting whether Dutch parliamentary motions pass, using ~30,000 speeches from the Tweede Kamer. The pipeline runs RobBERT sentiment analysis on actual speech transcripts and feeds the results into XGBoost and Logistic Regression classifiers.

---

## Setup

```bash
pip install -r requirements.txt
```

NB3 runs RobBERT inference — a GPU significantly speeds this up (Google Colab T4 recommended), but CPU works too.

---

## Path configuration

Each notebook contains a **PATH CONFIGURATION** cell at the top — it is the only cell you need to edit before running.

**Google Colab**  
The cell auto-detects Colab, mounts your Drive, and reads/writes everything under:
```
/content/drive/MyDrive/parliament_project/
```
Change that path in the cell if your Drive folder is named differently.

**Local (Jupyter / JupyterLab / VS Code)**  
Files are read from and written to the `data/` folder inside `mindless_machine_pipeline/`. The folder already contains all intermediate files so you can start at any notebook. Place `speeches_modeling.csv` there if running from scratch.

```python
# The only line you may need to change for local runs:
DATA_DIR = Path("data")   # ← point this at wherever your CSVs live
```

No `drive.mount` calls appear outside this cell, so the notebooks run unchanged on local environments.

---

## Main pipeline

Run in order from inside `mindless_machine_pipeline/`:

```
NB1 → NB2 → NB3 → NB4
```

### NB1 — `01_loading_eda.ipynb`
| | |
|---|---|
| **Input** | `data/speeches_modeling.csv` |
| **Output** | `data/speeches_clean.csv` |

Loads raw parliamentary speeches, drops empty/short texts, produces EDA visualisations (speech length, target distribution, speaker/party breakdown, time-of-day patterns).

---

### NB2 — `02_preprocessing_features.ipynb`
| | |
|---|---|
| **Input** | `data/speeches_clean.csv` |
| **Output** | `data/speeches_features.csv`, `data/feature_cols.json`, `data/class_meta.json` |

Text cleaning, feature engineering (cyclic time features, party one-hots, interaction terms), train/val/test split, feature–target correlation plot.

---

### NB3 — `03_robbert_sentiment_only_with_save.ipynb`
| | |
|---|---|
| **Input** | `data/speeches_features.csv` |
| **Output** | `data/speeches_with_sentiment.csv` |

Runs [`DTAI-KULeuven/robbert-v2-dutch-sentiment`](https://huggingface.co/DTAI-KULeuven/robbert-v2-dutch-sentiment) on every speech. Long speeches (> 512 tokens) are split into overlapping chunks and scores are averaged. Adds `sentiment_score`, `sentiment_pos`, `sentiment_neg`, `sentiment_label` columns.

> `03_robbert_sentiment_tone.ipynb` is an extended variant that also runs zero-shot tone classification (`aggressive / mean / neutral / peaceful / kind / happy`) via `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`.

---

### NB4 — `04_predictive_modelling.ipynb`
| | |
|---|---|
| **Input** | `data/speeches_with_sentiment.csv`, `data/feature_cols.json`, `data/class_meta.json` |
| **Output** | model result JSONs → `data/outputs/` |

Trains and evaluates:
- **XGBoost** — structured + party + RobBERT sentiment features
- **Logistic Regression** (with CV + StandardScaler) — same feature set
- **RobBERT classifier** — fine-tuned directly on raw speech text

Produces confusion matrices, ROC curves, and feature importance plots.

---

## Bias auditing pipeline

Run after NB4, **in numbered order**, from inside `auditing_pipeline/`:

```bash
python 01_bias_audit.py
python 02_bias_mitigation.py
python 03_indicator_mitigation.py
python 04_proxy_mechanism_audit.py
```

All scripts read from `mindless_machine_pipeline/data/` and auto-detect the repo root, so they work from any working directory.

### `01_bias_audit.py` → `01_bias_audit_outputs/`

**Diagnose** — find where bias could exist before trying to fix it.

Checks missingness across all columns and by subgroup (party, time bin, year, chair status, sentiment label), audits row loss across pipeline files, compares motion pass rates across groups, and trains three logistic regression models with progressively richer feature sets (temporal only → add speech/sentiment → add party) to detect omitted variable bias. Writes a full summary to `01_bias_audit_outputs/week8_bias_audit_summary.txt`.

---

### `02_bias_mitigation.py` → `02_bias_mitigation_outputs/`

**Fix** — implement concrete mitigations based on the audit findings.

Adds missingness indicator columns (`speaker_party_missing`, `raw_agenda_time_missing`, `sentiment_available`, `speech_duration_extreme`), removes fully empty columns, and compares four model variants ending with threshold tuning (finds the classification threshold that minimises the recall gap between accepted vs rejected motions). Note: visibility bias from informal bargaining and lobbying cannot be fixed with public parliamentary data — the summary acknowledges this explicitly.

---

### `03_indicator_mitigation.py` → `03_indicator_mitigation_outputs/`

**Reframe** — test whether the features are direct causes or symptoms of deeper political mechanisms.

Trains models that predict institutional variables (chair status, party, time bin) from visible debate features to check whether party patterns are embedded in them. Runs controlled outcome models comparing visible-only, institutional-only, and combined feature sets. Recommends framing the model as an *audit of visible indicators* rather than a claim that sentiment or speech duration directly causes outcomes.

---

### `04_proxy_mechanism_audit.py` → `04_proxy_mechanism_outputs/`

**Characterise** — show how visible variables are patterned by party, role, and topic.

Runs association tests (eta-squared and Cramér's V), builds a detailed party profile per party, clusters parties into behavioural groups using K-Means, and reduces party profiles to principal components via PCA. Writes a summary to `04_proxy_mechanism_outputs/proxy_mechanism_summary.txt`.

---

## Data provenance

`speeches_modeling.csv` was produced by `tweedekamer_scraping.ipynb`, which hits the [Tweede Kamer Open Data API](https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0) to download debate transcripts, parses the XML, and joins speeches to motion outcomes. The raw XMLs and intermediate CSVs (`speeches_raw.csv`, `agendapunten.csv`, `activiteiten.csv`, `besluiten.csv`) are included in the repo. Run the scraping notebook only if you want to collect fresh data.

---

## Repository structure

```
tweedekamer/
├── mindless_machine_pipeline/          ← main 4-notebook pipeline
│   ├── 01_loading_eda.ipynb
│   ├── 02_preprocessing_features.ipynb
│   ├── 03_robbert_sentiment_only_with_save.ipynb
│   ├── 03_robbert_sentiment_tone.ipynb
│   ├── 04_predictive_modelling.ipynb
│   └── data/
│       ├── speeches_modeling.csv       ← pipeline input
│       ├── speeches_clean.csv          ← NB1 output
│       ├── speeches_features.csv       ← NB2 output
│       ├── feature_cols.json           ← NB2 output
│       ├── class_meta.json             ← NB2 output
│       └── speeches_with_sentiment.csv ← NB3 output
├── auditing_pipeline/                  ← week 8 bias auditing scripts
│   ├── 01_bias_audit.py
│   ├── 01_bias_audit_outputs/
│   ├── 02_bias_mitigation.py
│   ├── 02_bias_mitigation_outputs/
│   ├── 03_indicator_mitigation.py
│   ├── 03_indicator_mitigation_outputs/
│   ├── 04_proxy_mechanism_audit.py
│   └── 04_proxy_mechanism_outputs/
├── feature-engineering/                ← standalone feature experiments
│   ├── engineered_topics.csv
│   └── engineered_topics2.csv
├── tweedekamer_scraping.ipynb          ← data collection (run once)
├── speeches_raw.csv                    ← scraper output
├── agendapunten.csv                    ← raw API metadata
├── activiteiten.csv
├── besluiten.csv
└── requirements.txt
```
