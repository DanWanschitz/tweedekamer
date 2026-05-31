# Bias Auditing Pipeline

Four standalone scripts that audit and mitigate potential bias in the
Dutch parliamentary motion prediction project. Run them **in numbered
order** — each script builds conceptually on the previous one.

---

## Run order

```
01_bias_audit.py
02_bias_mitigation.py
03_indicator_mitigation.py
04_proxy_mechanism_audit.py
```

Run from the **repository root** (the folder that contains
`parliamentary_notebooks_speeches/`):

```bash
python 01_bias_audit.py
python 02_bias_mitigation.py
python 03_indicator_mitigation.py
python 04_proxy_mechanism_audit.py
```

All four scripts auto-detect the repo root, so they work whether you
run them from inside the `auditing/` folder or from the repo root.

---

## What each script does

### `01_bias_audit.py` → `01_bias_audit_outputs/`

**Diagnose** — find where bias could exist before trying to fix it.

- Checks missingness across all columns and by subgroup (party, time
  bin, year, chair status, sentiment label)
- Audits row loss across pipeline files (`speeches_clean.csv` →
  `speeches_features.csv` → `speeches_with_sentiment.csv`)
- Compares motion pass rates across subgroups (is one party's motions
  accepted at a very different rate?)
- Trains three logistic regression models with progressively richer
  feature sets (temporal only → add speech/sentiment → add party) to
  detect omitted variable bias
- Checks whether model accuracy is consistent across subgroups
- Writes a full summary: `01_bias_audit_outputs/week8_bias_audit_summary.txt`

---

### `02_bias_mitigation.py` → `02_bias_mitigation_outputs/`

**Fix** — implement concrete mitigations based on the audit findings.

- Adds missingness indicator columns (`speaker_party_missing`,
  `raw_agenda_time_missing`, `sentiment_available`,
  `speech_duration_extreme`) so the model knows *when* a value is absent
- Removes fully empty columns
- Compares four model variants:
  - Baseline A (temporal only)
  - Baseline C (with party indicators)
  - Mitigated D (adds missingness indicators)
  - Mitigated D with **threshold tuning** (finds the classification
    threshold that minimises the recall gap between accepted vs rejected)
- Produces before/after summary: `02_bias_mitigation_outputs/mitigation_before_after_summary.csv`
- Note: *visibility bias* (informal bargaining, lobbying, committee
  deals) cannot be fixed with public parliamentary data — the summary
  acknowledges this explicitly.

---

### `03_indicator_mitigation.py` → `03_indicator_mitigation_outputs/`

**Reframe** — test whether the features are direct causes or symptoms
of deeper political mechanisms.

- Trains models that predict *institutional variables* (chair status,
  party, time bin, sentiment label) from the visible debate features —
  if timing and speech duration can predict party, then party patterns
  are embedded in those features
- Runs controlled outcome models comparing three feature sets (visible
  only / institutional only / combined) to measure how much each group
  adds independently
- Writes a summary recommending the model be framed as an *audit of
  visible indicators* rather than a claim that sentiment or speech
  duration directly causes outcomes

---

### `04_proxy_mechanism_audit.py` → `04_proxy_mechanism_outputs/`

**Characterise** — show how visible variables are patterned by party,
role, and topic.

- **Association tests**: measures how strongly party / role / topic /
  time-of-day are associated with features like speech duration and
  sentiment score (eta-squared for numeric, Cramér's V for categorical)
- **Party profiling**: builds a detailed profile per party (speech
  volume, duration, sentiment, time-of-day share, accepted rate, topic
  concentration)
- **Clustering**: groups parties into behavioural clusters using K-Means
  with silhouette-score selection
- **PCA**: reduces party profiles to principal components to identify
  the main dimensions of variation across parties
- Writes a summary: `04_proxy_mechanism_outputs/proxy_mechanism_summary.txt`

---

## Input required

All scripts read from:

```
parliamentary_notebooks_speeches/speeches_with_sentiment.csv
```

This is the output of `03_robbert_sentiment_only_with_save.ipynb` from
the main pipeline. Run the five main notebooks first.

Scripts 01 and 04 also optionally use:

```
feature-engineering/engineered_topics2.csv   # 04 only, for topic merge
parliamentary_notebooks_speeches/feature_cols.json
parliamentary_notebooks_speeches/class_meta.json
```

These are optional — the scripts skip gracefully if they are absent.

---

## Dependencies

```bash
pip install pandas numpy scikit-learn scipy
```

No GPU required. All four scripts use CPU-only sklearn models.
