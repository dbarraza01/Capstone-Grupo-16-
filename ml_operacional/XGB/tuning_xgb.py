"""Tuning rapido/extendido para pipeline operacional XGBoost.

FAST_RUN=True usa n_iter=2 para validar sintaxis y salidas en segundos.
Cambiar a False habilita n_iter=50 para la busqueda completa.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_operacional.utils.metricas_operacionales import scorer_mae_asimetrico
from ml_operacional.utils.pipeline_operacional import (
    RANDOM_STATE,
    SEGMENTS,
    binary_target_los14,
    class_ratio,
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


def _base_clf() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def _base_reg() -> TransformedTargetRegressor:
    base = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    return TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)


def main() -> None:
    print(f"Tuning XGB operacional | FAST_RUN={FAST_RUN} | n_iter={N_ITER}")

    for segment in SEGMENTS:
        print(f"\n[XGB] Segmento: {segment}")
        train_df = load_segment_split(segment, "train")
        X, y = prepare_xy(train_df)
        y_bin = binary_target_los14(y)
        ratio = class_ratio(y_bin)
        print(f"  X_train={X.shape}; positivos LOS>=14={int(y_bin.sum())}; scale_pos_weight={ratio:.3f}")

        clf_distributions = {
            "n_estimators": [30, 50, 80] if FAST_RUN else [200, 400, 600, 800],
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.03, 0.05, 0.08, 0.10],
            "subsample": [0.6, 0.75, 0.9],
            "colsample_bytree": [0.4, 0.6, 0.8],
            "min_child_weight": [3, 5, 9, 15],
            "gamma": [1.0, 3.0, 5.0, 7.0],
            "reg_alpha": [0.0, 1.0, 2.5, 5.0],
            "reg_lambda": [1.0, 3.0, 5.0, 9.0],
            "scale_pos_weight": [1.0, ratio],
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

        prob_oof = generate_oof_probabilities("xgb", best_clf, X, y_bin)
        train_prob = export_oof_dataset(segment, train_df, prob_oof, "xgb")
        X_reg, y_reg = prepare_xy(train_prob, include_prob=True)

        reg_distributions = {
            "regressor__n_estimators": [40, 80, 120] if FAST_RUN else [300, 500, 700, 900],
            "regressor__max_depth": [3, 4, 5, 6, 7],
            "regressor__learning_rate": [0.03, 0.05, 0.08, 0.10],
            "regressor__subsample": [0.6, 0.75, 0.9],
            "regressor__colsample_bytree": [0.4, 0.6, 0.8],
            "regressor__min_child_weight": [3, 5, 9, 15],
            "regressor__gamma": [1.0, 3.0, 5.0, 7.0],
            "regressor__reg_alpha": [0.0, 1.0, 2.5, 5.0],
            "regressor__reg_lambda": [1.0, 3.0, 5.0, 9.0],
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
