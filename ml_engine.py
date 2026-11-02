"""
Machine Learning Pipeline for DNA Sequence Classification
Computational Biology - Scikit-Learn Training & Prediction Engine
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
import os
import sys

# Ensure local imports work cleanly
sys.path.append(os.path.dirname(__file__))
from feature_extractor import extract_features, extract_batch_features


# --- Canonical Assignment 2 Dataset ---
ASSIGNMENT_CORE_DATASET = [
    ("S1", "ATGCGATC", "Group A"),
    ("S2", "ATGCGTAC", "Group A"),
    ("S3", "GCGCGTAT", "Group B"),
    ("S4", "GCGCGCGC", "Group B"),
]

# --- Augmented Biological Dataset (CpG / GC-rich vs AT-rich / Balanced) ---
AUGMENTED_DATASET_EXTRAS = [
    # Group A: Moderate GC / AT-balanced (40% - 50% GC)
    ("A_01", "AAGCTTAG", "Group A"),
    ("A_02", "ATATGCAT", "Group A"),
    ("A_03", "TACGATGC", "Group A"),
    ("A_04", "TAGCTAAT", "Group A"),
    ("A_05", "GATCGAAT", "Group A"),
    ("A_06", "TGCAATTC", "Group A"),
    ("A_07", "ATGCATGC", "Group A"),
    ("A_08", "TTACGCAT", "Group A"),
    ("A_09", "AATGCGAT", "Group A"),
    ("A_10", "CTAATGCG", "Group A"),

    # Group B: GC-rich / CpG island motifs (62.5% - 100% GC)
    ("B_01", "GCGCGGCC", "Group B"),
    ("B_02", "CCGGCCGG", "Group B"),
    ("B_03", "CGCGCGAT", "Group B"),
    ("B_04", "GGCGCGCG", "Group B"),
    ("B_05", "GCCGCGTA", "Group B"),
    ("B_06", "CGCCGCGG", "Group B"),
    ("B_07", "GGCCGGCC", "Group B"),
    ("B_08", "GCGCATGC", "Group B"),
    ("B_09", "CGCGCCGG", "Group B"),
    ("B_10", "GCGCGCAT", "Group B"),
]

FEATURE_COLUMNS = ["Length", "Count_A", "Count_T", "Count_G", "Count_C", "GC_Content_%"]


def get_training_dataframe(dataset_type: str = "core") -> pd.DataFrame:
    """
    Returns DataFrame for training dataset:
      - 'core': Assignment S1-S4 only
      - 'expanded': Core + 20 biological augmented sequences
    """
    if dataset_type == "core":
        return extract_batch_features(ASSIGNMENT_CORE_DATASET, include_extended=False)
    else:
        combined = ASSIGNMENT_CORE_DATASET + AUGMENTED_DATASET_EXTRAS
        return extract_batch_features(combined, include_extended=False)


class DNAClassifierPipeline:
    """
    Manages model initialization, training, evaluation, and inference.
    """

    def __init__(self, model_type: str = "Random Forest", **kwargs):
        self.model_type = model_type
        self.feature_columns = FEATURE_COLUMNS
        self.labels = ["Group A", "Group B"]
        self.model = self._create_model(model_type, **kwargs)
        self.is_trained = False
        self.training_df = None
        self.decision_tree_rules = ""

    def _create_model(self, model_type: str, n_samples: int = 4, **kwargs):
        if model_type == "Decision Tree":
            max_depth = kwargs.get("max_depth", 3)
            return DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        elif model_type == "Random Forest":
            n_estimators = kwargs.get("n_estimators", 50)
            max_depth = kwargs.get("max_depth", 4)
            # For small N datasets (such as N=4 assignment dataset), disable bootstrap
            # to prevent degenerate single-class subsamples
            use_bootstrap = kwargs.get("bootstrap", n_samples > 10)
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                bootstrap=use_bootstrap,
                random_state=42
            )
        elif model_type == "Logistic Regression":
            return LogisticRegression(random_state=42, max_iter=200)
        elif model_type == "Support Vector Machine":
            return SVC(kernel="linear", probability=True, random_state=42)
        else:
            return RandomForestClassifier(n_estimators=50, bootstrap=False, random_state=42)

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Trains the classifier on provided DataFrame and computes evaluation metrics.
        """
        self.training_df = df.copy()
        X = df[self.feature_columns].values
        y = df["Label"].values

        # Re-instantiate model with sample count awareness
        self.model = self._create_model(self.model_type, n_samples=len(y))

        self.model.fit(X, y)
        self.is_trained = True

        # In-sample metrics
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, pos_label="Group B", zero_division=0)
        rec = recall_score(y, y_pred, pos_label="Group B", zero_division=0)
        f1 = f1_score(y, y_pred, pos_label="Group B", zero_division=0)
        cm = confusion_matrix(y, y_pred, labels=["Group A", "Group B"])

        # Leave-One-Out Cross Validation (ideal for small n assignment dataset)
        loo_acc = None
        if len(y) >= 4:
            loo = LeaveOneOut()
            try:
                cv_scores = cross_val_score(self.model, X, y, cv=loo, scoring="accuracy")
                loo_acc = float(np.mean(cv_scores))
            except Exception:
                loo_acc = acc

        # Feature Importances
        feature_importances = {}
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_importances = {col: round(float(imp), 4) for col, imp in zip(self.feature_columns, importances)}
        elif hasattr(self.model, "coef_"):
            coefs = np.abs(self.model.coef_[0])
            total = np.sum(coefs) if np.sum(coefs) > 0 else 1.0
            norm_coefs = coefs / total
            feature_importances = {col: round(float(imp), 4) for col, imp in zip(self.feature_columns, norm_coefs)}

        # Decision tree text logic
        if self.model_type == "Decision Tree":
            self.decision_tree_rules = export_text(self.model, feature_names=self.feature_columns)
        elif self.model_type == "Random Forest" and len(self.model.estimators_) > 0:
            self.decision_tree_rules = export_text(self.model.estimators_[0], feature_names=self.feature_columns)

        return {
            "accuracy": round(float(acc) * 100.0, 2),
            "precision": round(float(prec) * 100.0, 2),
            "recall": round(float(rec) * 100.0, 2),
            "f1_score": round(float(f1) * 100.0, 2),
            "loo_cv_accuracy": round(float(loo_acc) * 100.0, 2) if loo_acc is not None else None,
            "confusion_matrix": cm.tolist(),
            "feature_importances": feature_importances,
            "sample_count": len(y),
            "tree_rules": self.decision_tree_rules,
        }

    def predict_sequence(self, raw_seq: str, seq_id: str = "Test_Sequence") -> Dict[str, Any]:
        """
        Extracts features and predicts class + probability for a test DNA sequence.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call train() first.")

        feat = extract_features(raw_seq, seq_id=seq_id, include_extended=True)
        feature_vector = np.array([[feat[c] for c in self.feature_columns]])

        predicted_class = self.model.predict(feature_vector)[0]

        # Probability estimates
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feature_vector)[0]
            class_indices = {cls: idx for idx, cls in enumerate(self.model.classes_)}
            prob_a = float(probs[class_indices.get("Group A", 0)])
            prob_b = float(probs[class_indices.get("Group B", 1)])
        else:
            prob_a = 1.0 if predicted_class == "Group A" else 0.0
            prob_b = 1.0 if predicted_class == "Group B" else 0.0

        confidence = prob_b if predicted_class == "Group B" else prob_a

        return {
            "id": seq_id,
            "sequence": feat["Sequence"],
            "features": feat,
            "predicted_class": predicted_class,
            "confidence_pct": round(confidence * 100.0, 2),
            "prob_group_a_pct": round(prob_a * 100.0, 2),
            "prob_group_b_pct": round(prob_b * 100.0, 2),
        }
