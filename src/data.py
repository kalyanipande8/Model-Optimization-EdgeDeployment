"""Synthetic classification dataset generation."""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np


def load_data(seed=42):
    X, y = make_classification(
        n_samples=6000,
        n_features=40,
        n_informative=25,
        n_redundant=8,
        n_repeated=0,
        n_classes=4,
        n_clusters_per_class=2,
        class_sep=1.2,
        flip_y=0.03,
        random_state=seed,
    )
    X = X.astype(np.float32)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mean) / std

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
