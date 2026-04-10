# main.py
import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# configuration
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

RANDOM_STATE = 42


# text preprocessing
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)                 # this normalizes whitespace
    text = re.sub(r"[^a-z0-9\s']", " ", text)        # removes most punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# this is a metrics helper
def compute_metrics(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)

    # macro: treats classes equally
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    # weighted: accounts for class support
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f1_w,
    }

def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS.keys()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[LABELS[i] for i in LABELS])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=200)
    plt.close(fig)

def save_metric_comparison_plot(lr_metrics: dict, dt_metrics: dict, filename="metrics_comparison.png"):
    # this compares the headline metrics specified in the project:
    # accuracy, precision, recall, f1-score (macro averages)
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    lr_vals = [lr_metrics[k] for k in keys]
    dt_vals = [dt_metrics[k] for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, lr_vals, width, label="Logistic Regression")
    ax.bar(x + width/2, dt_vals, width, label="Decision Tree")

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Performance Comparison (Macro Averages)")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=200)
    plt.close(fig)



# Main
def main():
    # firstly Load dataset
    ds = load_dataset("wangrongsheng/ag_news")
    train = ds["train"]
    test = ds["test"]

    X_train_raw = train["text"]
    y_train = np.array(train["label"])
    X_test_raw = test["text"]
    y_test = np.array(test["label"])

    # then Basic dataset summary
    print("Train size:", len(X_train_raw))
    print("Test size :", len(X_test_raw))
    unique, counts = np.unique(y_train, return_counts=True)
    print("Train label distribution:", {LABELS[int(k)]: int(v) for k, v in zip(unique, counts)})

    # preprocess
    X_train = [clean_text(t) for t in X_train_raw]
    X_test = [clean_text(t) for t in X_test_raw]

    # examples
    for i in range(3):
        print("\nExample", i+1)
        print("Before:", X_train_raw[i])
        print("After :", X_train[i])

    # tf-idf vectorization
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),     # (1,1) baseline is ok; (1,2) often improves headlines
        max_features=50000,     # control dimensionality
        min_df=2,               # this ignores very rare terms
        stop_words="english"    # this is optional
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("\nTF-IDF feature count:", X_train_tfidf.shape[1])

    # Logistic Regression 
    lr = LogisticRegression(
        max_iter=200,
        C=2.0,
        solver="lbfgs",
        n_jobs=None,            # we're keeping it default for portability
        random_state=RANDOM_STATE
    )
    lr.fit(X_train_tfidf, y_train)
    y_pred_lr = lr.predict(X_test_tfidf)

    # Decision Tree 
    dt = DecisionTreeClassifier(
        criterion="gini",
        max_depth=40,           
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    )
    dt.fit(X_train_tfidf, y_train)
    y_pred_dt = dt.predict(X_test_tfidf)

    # evaluation
    lr_metrics = compute_metrics(y_test, y_pred_lr)
    dt_metrics = compute_metrics(y_test, y_pred_dt)

    print("\n--- Logistic Regression Metrics ---")
    for k, v in lr_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n--- Decision Tree Metrics ---")
    for k, v in dt_metrics.items():
        print(f"{k}: {v:.4f}")

    # this saves metrics to file for easy report insertion
    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_safe(v) for v in obj]
        elif isinstance(obj, type):
            return str(obj)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    results = {
        "tfidf_params": make_json_safe(vectorizer.get_params()),
        "logreg_params": make_json_safe(lr.get_params()),
        "decision_tree_params": make_json_safe(dt.get_params()),
        "logreg_metrics": lr_metrics,
        "decision_tree_metrics": dt_metrics,
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Graphs
    save_confusion_matrix(y_test, y_pred_lr, "Confusion Matrix - Logistic Regression", "confusion_lr.png")
    save_confusion_matrix(y_test, y_pred_dt, "Confusion Matrix - Decision Tree", "confusion_dt.png")
    save_metric_comparison_plot(lr_metrics, dt_metrics, "metrics_comparison.png")

    # this presents the decision tree
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(
        dt,
        max_depth=3, 
        filled=True,
        feature_names=None,     # too many TF-IDF features, meaning not useful to print all
        class_names=[LABELS[i] for i in LABELS],
        ax=ax
    )
    ax.set_title("Decision Tree (Top Levels Only)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "decision_tree.png"), dpi=200)
    plt.close(fig)

    print("\nSaved outputs to:", OUTPUT_DIR)
    print(" - confusion_lr.png, confusion_dt.png")
    print(" - metrics_comparison.png")
    print(" - decision_tree.png")
    print(" - results.json")

if __name__ == "__main__":
    main()
