
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# CONFIG

FILES = {
    "spectral": "20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv",
    "cepstral": "20240106_dfall_obs_data_and_cepstral_features_revision1_n469.csv"
}
TARGET = "Context2"
RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER = 10  # randomized search iterations


# Helpers

def load_dataset(path):
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in {path}")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def build_preprocessor(X):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])
    return pre


def run_one_dataset(tag, path):
    print(f"\n=== Running {tag} dataset ===")
    df = load_dataset(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    pre = build_preprocessor(X)

    # Models + hyperparameter spaces
    setups = [
        ("SVM", SVC(probability=True, random_state=RANDOM_STATE), {
            "clf__C": np.logspace(-2, 2, 5),
            "clf__gamma": np.logspace(-3, -1, 3),
            "clf__kernel": ["rbf"]
        }),
        ("DecisionTree", DecisionTreeClassifier(random_state=RANDOM_STATE), {
            "clf__max_depth": [3, 5, 7, 10],
            "clf__min_samples_leaf": [2, 4, 6]
        }),
        ("RandomForest", RandomForestClassifier(random_state=RANDOM_STATE), {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [5, 10, 15],
            "clf__max_features": ["sqrt"]
        }),
        ("AdaBoost", AdaBoostClassifier(random_state=RANDOM_STATE), {
            "clf__n_estimators": [50, 100],
            "clf__learning_rate": [0.1, 0.5, 1.0]
        }),
        ("GaussianNB", GaussianNB(), {
            "clf__var_smoothing": np.logspace(-9, -7, 3)
        }),
        ("MLP", MLPClassifier(max_iter=500, random_state=RANDOM_STATE), {
            "clf__hidden_layer_sizes": [(64,), (128,), (64,32)],
            "clf__alpha": [0.0001, 0.001]
        })
    ]

    results = []
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, estimator, param_dist in setups:
        try:
            pipe = Pipeline([("pre", pre), ("clf", estimator)])
            search = RandomizedSearchCV(pipe, param_dist, n_iter=N_ITER,
                                        scoring="f1_weighted", cv=cv,
                                        n_jobs=-1, random_state=RANDOM_STATE)
            search.fit(X, y)
            best = search.best_estimator_

            # Evaluate with cross_val_score for stability
            scores_acc = cross_val_score(best, X, y, cv=cv, scoring="accuracy")
            scores_f1 = cross_val_score(best, X, y, cv=cv, scoring="f1_weighted")

            row = {
                "model": name,
                "cv_best_f1_w": float(search.best_score_),
                "cv_mean_acc": float(scores_acc.mean()),
                "cv_mean_f1_w": float(scores_f1.mean())
            }
            results.append(row)

            print(f"  {name} done (CV f1={scores_f1.mean():.3f})")

        except Exception as e:
            print(f"  {name} skipped -> {e}")

    res_df = pd.DataFrame(results).sort_values(by="cv_mean_f1_w", ascending=False)
    out_csv = f"Lab07_results_{tag}.csv"
    res_df.to_csv(out_csv, index=False)

    print("\n--- Results Table ---")
    print(res_df.round(3).to_string(index=False))
    print(f"\nSaved: {out_csv}")

    return res_df


def main():
    for tag, path in FILES.items():
        try:
            run_one_dataset(tag, path)
        except Exception as e:
            print(f"Failed on {tag}: {e}")


if __name__ == "__main__":
    main()

