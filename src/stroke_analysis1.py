
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_csv(sys.argv[1])


data["bmi"] = pd.to_numeric(data["bmi"], errors="coerce")
data["bmi"] = data["bmi"].fillna(data["bmi"].median())
data = data.drop_duplicates()

counts = data["stroke"].value_counts().sort_index()
plt.bar(["No stroke", "Stroke"], counts.values, color=["steelblue", "tomato"])
plt.title("Patient count by stroke outcome")
plt.ylabel("Number of patients")
plt.tight_layout()
plt.savefig("reports/stroke_counts.png")
plt.close()

# Prepare model inputs. Convert text categories into 0/1 columns.
X = data.drop(columns=["stroke", "id"])
y = data["stroke"]
X = pd.get_dummies(X, drop_first=True)

# Keep the same stroke / no-stroke balance in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale the inputs for Logistic Regression.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and compare Logistic Regression and Random Forest.
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]

    predictions = np.where(probabilities >= 0.5, 1, 0)
    print("\n" + name)
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 3))
    print("PR-AUC:", round(average_precision_score(y_test, probabilities), 3))
    print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 3))
    print("F1:", round(f1_score(y_test, predictions, zero_division=0), 3))

# Group patients using a few health-related numeric columns.
cluster_columns = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
cluster_data = data[cluster_columns]
cluster_data = SimpleImputer(strategy="median").fit_transform(cluster_data)
cluster_data = StandardScaler().fit_transform(cluster_data)

clusters = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(cluster_data)

# Use PCA to reduce the five columns to two dimensions for a chart.
points = PCA(n_components=2).fit_transform(cluster_data)
plt.scatter(points[:, 0], points[:, 1], c=clusters, cmap="viridis", s=18)
plt.title("Patient groups using K-Means and PCA")
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.tight_layout()
plt.savefig("reports/patient_groups.png")
plt.close()

print("\nCharts saved in the reports folder.")
