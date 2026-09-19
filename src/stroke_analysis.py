"""End-to-end stroke-risk analytics portfolio project.

Run:
    python src/stroke_analysis.py --data /path/to/healthcare-dataset-stroke-data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# A non-interactive backend keeps the project reproducible on servers and CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Load and minimally clean the source data."""
    df = pd.read_csv(path)
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df = df.drop_duplicates().copy()
    return df


def save_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """Create straightforward charts for the project report."""
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    stroke_counts = df["stroke"].value_counts().sort_index()
    axes[0].bar(["No stroke", "Stroke"], stroke_counts.values, color=["#4C78A8", "#E45756"])
    axes[0].set_title("Target distribution")
    axes[0].set_ylabel("Patients")
    for i, value in enumerate(stroke_counts.values):
        axes[0].text(i, value + 50, f"{value:,}", ha="center")

    for label, group in df.groupby("stroke"):
        axes[1].hist(group["age"], bins=np.arange(0, 91, 5), alpha=0.65,
                     label="Stroke" if label else "No stroke")
    axes[1].set_title("Age distribution by outcome")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Patients")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "eda_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    risk_by_age = (df.assign(age_band=pd.cut(df["age"], bins=[0, 30, 45, 60, 75, 100], right=False))
                  .groupby("age_band", observed=True)["stroke"].mean().mul(100))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(risk_by_age.astype(str), risk_by_age.values, marker="o", color="#E45756", linewidth=2)
    ax.set_title("Observed stroke rate by age group")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Stroke rate (%)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_dir / "stroke_rate_by_age.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])


def evaluate_models(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Train two classifiers and save metrics plus ROC/PR curves."""
    X = df.drop(columns=["stroke", "id"])
    y = df["stroke"]
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=350, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    metrics: list[dict[str, float | str]] = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, model in models.items():
        pipeline = Pipeline([("prep", make_preprocessor(numeric_cols, categorical_cols)), ("model", model)])
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        metrics.append({
            "model": name,
            "roc_auc": roc_auc_score(y_test, probabilities),
            "pr_auc": average_precision_score(y_test, probabilities),
            "precision_stroke": report["1"]["precision"],
            "recall_stroke": report["1"]["recall"],
            "f1_stroke": report["1"]["f1-score"],
        })
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probabilities):.2f})")
        axes[1].plot(recall, precision, label=f"{name} (AP={average_precision_score(y_test, probabilities):.2f})")

    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision-recall curves", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "model_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    results = pd.DataFrame(metrics).sort_values("pr_auc", ascending=False)
    results.to_csv(output_dir.parent / "model_metrics.csv", index=False)
    return results


def cluster_patients(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Find exploratory patient profiles using scaled clinical numeric features."""
    features = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
    X = SimpleImputer(strategy="median").fit_transform(df[features])
    X = StandardScaler().fit_transform(X)
    candidates = range(2, 7)
    scores = {k: silhouette_score(X, KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(X)) for k in candidates}
    best_k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE).fit_predict(X)
    clustered = df.assign(cluster=labels)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    points = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", alpha=0.65, s=18)
    ax.set(title=f"Patient segments: K-Means (k={best_k})", xlabel="PCA component 1", ylabel="PCA component 2")
    ax.legend(*scatter.legend_elements(), title="Cluster")
    fig.tight_layout()
    fig.savefig(output_dir / "patient_clusters_pca.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = clustered.groupby("cluster").agg(
        patients=("id", "size"), average_age=("age", "mean"), average_glucose=("avg_glucose_level", "mean"),
        average_bmi=("bmi", "mean"), observed_stroke_rate=("stroke", "mean")
    ).round(2)
    summary["observed_stroke_rate"] *= 100
    summary.to_csv(output_dir.parent / "cluster_summary.csv")
    return pd.DataFrame({"k": list(scores.keys()), "silhouette_score": list(scores.values())})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stroke-risk analytics project.")
    parser.add_argument("--data", required=True, help="Path to healthcare-dataset-stroke-data.csv")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    save_eda(df, output_dir)
    metrics = evaluate_models(df, output_dir)
    silhouettes = cluster_patients(df, output_dir)
    silhouettes.to_csv(project_root / "reports" / "silhouette_scores.csv", index=False)
    print(f"Analysed {len(df):,} patient records.")
    print("\nModel results (higher PR-AUC is better for this imbalanced target):")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSaved charts and CSV reports to reports/.")


if __name__ == "__main__":
    main()
