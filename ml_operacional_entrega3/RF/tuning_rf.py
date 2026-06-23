"""Tuning rapido/extendido para pipeline operacional Random Forest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_operacional_entrega3.utils.metricas_operacionales import scorer_mae_asimetrico
from ml_operacional_entrega3.utils.pipeline_operacional import (
    RANDOM_STATE,
    SEGMENTS,
    binary_target_los14,
    clean_regressor_params,
    export_oof_dataset,
    generate_oof_probabilities,
    load_segment_split,
    prepare_xy,
    save_json,
)


FAST_RUN = False  # Cambiar a True para tuning rapido (n_iter=2) o False para tuning completo (n_iter=50)
N_ITER = 2 if FAST_RUN else 50
OUT_DIR = Path(__file__).resolve().parent


def _base_clf() -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        bootstrap=True,
    )


def _base_reg() -> TransformedTargetRegressor:
    base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, bootstrap=True)
    return TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)


def main() -> None:
    print(f"Tuning RF operacional | FAST_RUN={FAST_RUN} | n_iter={N_ITER}")

    for segment in SEGMENTS:
        print(f"\n[RF] Segmento: {segment}")
        train_df = load_segment_split(segment, "train")
        X, y = prepare_xy(train_df)
        y_bin = binary_target_los14(y)
        print(f"  X_train={X.shape}; positivos LOS>=14={int(y_bin.sum())}")

        clf_distributions = {
            "n_estimators": [30, 50, 80] if FAST_RUN else [200, 400, 600, 800],
            "max_depth": [8, 10, 12, 15, 20, 25],
            "min_samples_split": [10, 20, 40, 60],
            "min_samples_leaf": [5, 8, 15, 30],
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
            "max_samples": [0.5, 0.7, 0.9],
        }
        search_clf = RandomizedSearchCV(
            estimator=_base_clf(),
            param_distributions=clf_distributions,
            n_iter=N_ITER,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        search_clf.fit(X, y_bin)
        best_clf = search_clf.best_params_
        save_json(OUT_DIR / f"best_params_clf_{segment}.json", best_clf)
        print(f"  Mejor ROC-AUC CV: {search_clf.best_score_:.4f}")

        prob_oof = generate_oof_probabilities("rf", best_clf, X, y_bin)
        train_prob = export_oof_dataset(segment, train_df, prob_oof, "rf")
        X_reg, y_reg = prepare_xy(train_prob, include_prob=True)

        reg_distributions = {
            "regressor__n_estimators": [30, 50, 80] if FAST_RUN else [200, 400, 600, 800],
            "regressor__max_depth": [8, 10, 12, 15, 20, 25],
            "regressor__min_samples_split": [10, 20, 40, 60],
            "regressor__min_samples_leaf": [5, 8, 15, 30],
            "regressor__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
            "regressor__max_samples": [0.5, 0.7, 0.9],
        }
        search_reg = RandomizedSearchCV(
            estimator=_base_reg(),
            param_distributions=reg_distributions,
            n_iter=N_ITER,
            scoring=scorer_mae_asimetrico(alpha=2.0),
            cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        search_reg.fit(X_reg, y_reg)
        best_reg = clean_regressor_params(search_reg.best_params_)
        save_json(OUT_DIR / f"best_params_reg_{segment}.json", best_reg)
        print(f"  Mejor MAE asimetrico CV: {-search_reg.best_score_:.4f}")


if __name__ == "__main__":
    main()
