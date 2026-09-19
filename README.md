# Stroke Risk Analytics

An end-to-end machine-learning project that explores stroke-risk patterns in 5,110 anonymised patient records. I built it as a compact portfolio project to practise the parts of a real data-science workflow: cleaning imperfect data, choosing metrics that fit the problem, comparing models, and turning an open-ended dataset into a few useful exploratory patient segments.

> Educational portfolio project only. It is not a clinical decision tool and should never be used to diagnose patients.

## What it demonstrates

- Data cleaning: numeric conversion, missing-value handling, and duplicate checks
- Exploratory data analysis with publication-ready Matplotlib charts
- Supervised classification: Logistic Regression and class-weighted Random Forest
- Imbalanced-class evaluation: precision, recall, F1, ROC-AUC, and PR-AUC
- Unsupervised learning: K-Means clustering, silhouette-score selection, and PCA visualization
- Reproducible Python project structure and saved results

## Results from this dataset

The dataset is highly imbalanced: only 249 of 5,110 records are stroke cases. For that reason, the project prioritises **PR-AUC, recall, and F1** over raw accuracy.

| Model | ROC-AUC | PR-AUC | Stroke recall |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.843 | 0.260 | 0.790 |
| Random Forest | 0.809 | 0.163 | 0.129 |

On this train/test split, Logistic Regression provides the better precision–recall trade-off. This does **not** make it suitable for clinical use: a real deployment would need external validation, calibration, fairness checks, and review by clinical experts.

## A few things I found

- Stroke prevalence increases markedly in the older age bands in this dataset.
- The clustering step found three broad, data-driven patient profiles with different average age, glucose level, BMI, and observed stroke rate.
- These clusters are exploratory summaries, not diagnoses or medical categories.

## Project structure

```
stroke_ml_portfolio/
├── src/stroke_analysis.py       # complete reproducible pipeline
├── reports/figures/             # generated charts
├── reports/*.csv                # saved metrics and cluster summaries
├── requirements.txt
└── README.md
```

## Quick start

```bash
cd stroke_ml_portfolio
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/stroke_analysis.py --data /path/to/healthcare-dataset-stroke-data.csv
```

Outputs are written to `reports/`: figures, model comparison metrics, and a patient cluster summary.

## Data

This repository intentionally does not include patient-level source data. Download `healthcare-dataset-stroke-data.csv` separately and pass its local path to the command above. Do not upload sensitive or personally identifiable health data to a public repository.

## CV-ready description

**Stroke Risk Analytics | Python, Pandas, NumPy, Scikit-learn, Matplotlib**  
Built an end-to-end ML pipeline on 5K+ patient records, including data cleaning, EDA, class-imbalanced stroke prediction (Logistic Regression and Random Forest), and K-Means patient segmentation visualized with PCA. Evaluated models using ROC-AUC and PR-AUC and exported reproducible reports.

## Interview talking points

1. **Why not accuracy?** Only a small fraction of records are stroke cases; a model predicting “no stroke” for everyone would look highly accurate but be unhelpful. PR-AUC, recall and F1 reveal performance on the minority class.
2. **Why class weights?** They make misclassifying a stroke case costlier during model fitting without inventing synthetic medical records.
3. **Why use a pipeline?** Imputation, scaling and one-hot encoding are fit only on training data, preventing leakage into test metrics.
4. **What are clusters for?** They are exploratory profiles, not medical labels. They can support targeted analysis after clinical validation.

## Possible next steps

- Tune the decision threshold around recall/precision requirements.
- Add cross-validation and calibration curves.
- Serve predictions in a Streamlit dashboard after appropriate clinical validation.
