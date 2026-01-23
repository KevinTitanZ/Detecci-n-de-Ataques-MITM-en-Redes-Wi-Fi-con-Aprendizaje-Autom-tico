import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay
)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


@dataclass
class ModelSpec:
    name: str
    estimator: Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_label_column(df: pd.DataFrame, label_col: Optional[str]) -> str:
    if label_col and label_col in df.columns:
        return label_col

    # Heurística común
    candidates = ["label", "y", "target", "class", "attack", "is_attack"]
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "No pude inferir la columna label. Pásala explícitamente con --label-col."
    )


def to_binary_labels(y: pd.Series) -> pd.Series:
    """
    Convierte etiquetas a binario 0/1 si vienen como strings tipo:
    'normal'/'attack', 'benign'/'malicious', etc.
    Si ya es numérico, lo respeta (pero lo castea a int si es 0/1).
    """
    if pd.api.types.is_numeric_dtype(y):
        # Si son 0/1 o 0/1.0
        unique = sorted(pd.unique(y.dropna()))
        if set(unique).issubset({0, 1, 0.0, 1.0}):
            return y.astype(int)
        # Si es multiclass numérico, lo dejamos tal cual (pero esto es para binario)
        return y

    y_str = y.astype(str).str.lower().str.strip()
    # Mapas típicos
    pos_tokens = {"attack", "malicious", "mitm", "evil_twin", "arp_spoof", "1", "true", "yes"}
    neg_tokens = {"normal", "benign", "legit", "0", "false", "no"}

    mapped = []
    for v in y_str:
        if v in pos_tokens:
            mapped.append(1)
        elif v in neg_tokens:
            mapped.append(0)
        else:
            # fallback: intenta detectar por substring
            if "attack" in v or "mitm" in v or "spoof" in v or "evil" in v:
                mapped.append(1)
            elif "normal" in v or "benign" in v or "legit" in v:
                mapped.append(0)
            else:
                raise ValueError(f"No puedo mapear etiqueta '{v}' a binario. Ajusta tu columna label.")
    return pd.Series(mapped, index=y.index, dtype=int)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Separar columnas numéricas/categóricas por robustez (por si tu dataset trae strings)
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop"
    )
    return pre


def get_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame) -> List[str]:
    """
    Recupera nombres de features después de ColumnTransformer (num + onehot).
    """
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    names = []
    # num passthrough (luego scaler)
    names.extend(num_cols)

    # cat onehot
    if len(cat_cols) > 0:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
        names.extend(ohe_names)

    return names


def evaluate_model(
    name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    # Probabilidades o score (para ROC/PR)
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if y_score is not None:
        # En SVM decision_function puede ser cualquier rango; roc_auc funciona igual
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
        metrics["avg_precision"] = float(average_precision_score(y_test, y_score))
    else:
        metrics["roc_auc"] = None
        metrics["avg_precision"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["report"] = classification_report(y_test, y_pred, zero_division=0)

    return metrics


def plot_confusion(cm: np.ndarray, title: str, outpath: str) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_roc_pr_curves(
    fitted_models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    outdir: str
) -> None:
    # ROC
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    for name, model in fitted_models.items():
        try:
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
        except Exception:
            continue
    plt.title("Curvas ROC (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curves_test.png"), dpi=220)
    plt.close()

    # PR
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    for name, model in fitted_models.items():
        try:
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
        except Exception:
            continue
    plt.title("Curvas Precision-Recall (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr_curves_test.png"), dpi=220)
    plt.close()


def plot_metric_bars(metrics_df: pd.DataFrame, outpath: str) -> None:
    cols = ["accuracy", "precision", "recall", "f1"]
    plot_df = metrics_df[["model"] + cols].copy()
    plot_df = plot_df.melt(id_vars="model", var_name="metric", value_name="value")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_df, x="metric", y="value", hue="model")
    plt.ylim(0, 1.0)
    plt.title("Comparación de métricas (Test)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_feature_importance_tree(
    model: Pipeline,
    model_name: str,
    feature_names: List[str],
    outpath: str,
    top_k: int = 20
) -> None:
    # Buscar el estimador final dentro del pipeline
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return

    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    plt.figure(figsize=(9, 6))
    sns.barplot(x=top_vals, y=top_names, orient="h")
    plt.title(f"Importancia de variables (Top {top_k}) - {model_name}")
    plt.xlabel("Importancia")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Comparación RF/SVM/DT en dataset final con figuras.")
    parser.add_argument("--data", required=True, help="Ruta al CSV del dataset final (features + label).")
    parser.add_argument("--label-col", default=None, help="Nombre de la columna de etiqueta. Ej: label")
    parser.add_argument("--drop-cols", default="", help="Columnas a eliminar (separadas por coma). Ej: ts,session_id")
    parser.add_argument("--outdir", default="reports/model_comparison_final", help="Carpeta de salida.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Proporción test.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Proporción validación (desde train_temp).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svm-kernel", default="rbf", choices=["rbf", "linear"], help="Kernel para SVM.")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    figdir = os.path.join(args.outdir, "figures")
    ensure_dir(figdir)

    df = pd.read_csv(args.data)

    label_col = infer_label_column(df, args.label_col)
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Separar X/y
    y_raw = df[label_col]
    X = df.drop(columns=[label_col] + drop_cols)

    # Limpieza básica: quitar columnas constantes
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    # Convertir y a binario si aplica (si ya es 0/1, queda igual)
    y = to_binary_labels(y_raw)

    # Split train/val/test estratificado
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    # val-size es proporción del trainval
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=args.val_size, random_state=args.seed, stratify=y_trainval
    )

    pre = build_preprocessor(X_train)

    # Modelos
    dt = DecisionTreeClassifier(
        random_state=args.seed,
        class_weight="balanced"
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    if args.svm_kernel == "linear":
        svm = SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=args.seed
        )
    else:
        svm = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=args.seed
        )

    specs = [
        ModelSpec("DecisionTree", dt),
        ModelSpec("RandomForest", rf),
        ModelSpec(f"SVM_{args.svm_kernel}", svm),
    ]

    fitted: Dict[str, Pipeline] = {}
    results: List[Dict[str, Any]] = []

    # Entrenamiento (usamos train; val queda para que luego tú ajustes hiperparámetros si quieres)
    for spec in specs:
        pipe = Pipeline(steps=[
            ("pre", pre),
            ("clf", spec.estimator)
        ])

        pipe.fit(X_train, y_train)
        fitted[spec.name] = pipe

        r = evaluate_model(spec.name, pipe, X_test, y_test)
        results.append(r)

        cm = np.array(r["confusion_matrix"])
        plot_confusion(
            cm,
            title=f"Matriz de confusión (Test) - {spec.name}",
            outpath=os.path.join(figdir, f"confusion_{spec.name}.png")
        )

    # Curvas ROC y PR (comparativas)
    plot_roc_pr_curves(fitted, X_test, y_test, figdir)

    # Tabla de métricas
    metrics_rows = []
    for r in results:
        metrics_rows.append({
            "model": r["model"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "roc_auc": r["roc_auc"],
            "avg_precision": r["avg_precision"],
        })
    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="f1", ascending=False)
    metrics_df.to_csv(os.path.join(args.outdir, "metrics_test.csv"), index=False)

    plot_metric_bars(metrics_df, os.path.join(figdir, "metrics_bar_test.png"))

    # Importancia de variables (RF + DT)
    # Necesitamos feature_names post-transformación (incluye onehot)
    # Para eso, fit_transform ya ocurrió dentro del pipeline; tomamos el preprocessor ya fitteado
    # desde cualquier modelo entrenado.
    any_model = next(iter(fitted.values()))
    pre_fitted = any_model.named_steps["pre"]
    feature_names = get_feature_names(pre_fitted, X_train)

    for name, model in fitted.items():
        outpath = os.path.join(figdir, f"feature_importance_{name}.png")
        plot_feature_importance_tree(model, name, feature_names, outpath, top_k=20)

    # Guardar reporte detallado
    with open(os.path.join(args.outdir, "full_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Guardar resumen en consola
    print("\n=== RESUMEN (TEST) ===")
    print(metrics_df.to_string(index=False))
    print("\nFiguras en:", figdir)
    print("CSV métricas:", os.path.join(args.outdir, "metrics_test.csv"))
    print("JSON completo:", os.path.join(args.outdir, "full_results.json"))

    # Reportes por modelo
    for r in results:
        print(f"\n--- Classification report: {r['model']} ---")
        print(r["report"])


if __name__ == "__main__":
    main()