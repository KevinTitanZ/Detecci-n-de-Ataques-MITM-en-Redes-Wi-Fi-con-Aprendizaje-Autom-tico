#!/usr/bin/env python3
"""
Calibración de umbrales compatible con el modelo actual (train_model.py).

Requisitos:
- models/mitm_detector.h5 (generado por train_model.py)
- data/processed/dataset_features.csv (generado por extract_features.py)

Salidas:
- results/threshold_analysis.csv
- results/threshold_analysis.png
- results/optimal_config.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    classification_report,
)

class CompatibleThresholdCalibrator:
    def __init__(self, model_path='models/mitm_detector.h5'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No existe el modelo en {model_path}. Entrena primero con: python3 scripts/train_model.py"
            )
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = StandardScaler()

        # Esperado por el modelo (para tu train_model.py suele ser input_dim)
        self.expected_features = self.model.input_shape[1]
        print(f"[+] Modelo cargado. Espera {self.expected_features} características.")

    def load_and_prepare_data(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No existe el dataset en {csv_path}. Ejecuta extract_features.py primero.")

        df = pd.read_csv(csv_path)

        exclude_cols = ['label', 'filename', 'window_start', 'window_end', 'window_id']
        feature_columns = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_columns].values
        y = df['label'].values.astype(int)

        print(f"[+] Dataset: {len(X)} muestras")
        print(f"[+] Features en CSV: {X.shape[1]}")
        print(f"[+] Clase 0 (normal): {int((y==0).sum())}")
        print(f"[+] Clase 1 (ataque): {int((y==1).sum())}")

        # Ajuste de dimensiones para compatibilidad con el modelo
        if X.shape[1] < self.expected_features:
            missing = self.expected_features - X.shape[1]
            X = np.pad(X, ((0, 0), (0, missing)), mode='constant', constant_values=0)
            print(f"[+] Se agregaron {missing} features con 0 (padding) para compatibilidad.")
        elif X.shape[1] > self.expected_features:
            X = X[:, :self.expected_features]
            print(f"[+] Se recortaron features extra para compatibilidad. Ahora: {X.shape[1]}")

        return X, y, feature_columns

    def comprehensive_threshold_analysis(self, X, y):
        # Split fijo para que sea reproducible
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Escalado (ojo: esto es para calibración; para producción usa el scaler del entrenamiento)
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # Probabilidades
        y_proba = self.model.predict(X_test_s, verbose=0)
        y_proba = y_proba.flatten()

        thresholds = np.arange(0.05, 0.96, 0.05)
        rows = []

        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)

            # Matriz de confusión robusta
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
            prec1 = tp / (tp + fp + 1e-12)
            rec1 = tp / (tp + fn + 1e-12)
            f1_1 = 2 * (prec1 * rec1) / (prec1 + rec1 + 1e-12)

            # Score orientado a ciberseguridad (prioriza recall)
            cybersec_score = (rec1 * 0.5) + (prec1 * 0.3) + (acc * 0.2)

            rows.append({
                "threshold": float(thr),
                "accuracy": float(acc),
                "precision_1": float(prec1),
                "recall_1": float(rec1),
                "f1_1": float(f1_1),
                "cybersec_score": float(cybersec_score),
                "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            })

        results_df = pd.DataFrame(rows)

        optimal = self.find_optimal_thresholds(results_df)
        self.create_visualizations(results_df, y_test, y_proba)

        # Reporte final con el umbral recomendado
        rec_thr = optimal.get("cybersec_optimal", 0.5)
        y_pred_final = (y_proba >= rec_thr).astype(int)
        print("\n=== REPORTE CON UMBRAL RECOMENDADO ===")
        print(f"Umbral recomendado: {rec_thr:.2f}")
        print(classification_report(y_test, y_pred_final, zero_division=0))

        return optimal, results_df

    def find_optimal_thresholds(self, results_df):
        optimal = {}

        # Máx recall ataques
        r1 = results_df.loc[results_df["recall_1"].idxmax()]
        optimal["max_recall_attacks"] = float(r1["threshold"])

        # Mejor F1 ataques
        f1 = results_df.loc[results_df["f1_1"].idxmax()]
        optimal["max_f1_attacks"] = float(f1["threshold"])

        # Mejor cybersec_score
        cs = results_df.loc[results_df["cybersec_score"].idxmax()]
        optimal["cybersec_optimal"] = float(cs["threshold"])

        print("\n=== UMBRALES ÓPTIMOS ===")
        print(f"1) Máximo Recall (ataques): {optimal['max_recall_attacks']:.2f}")
        print(f"2) Máximo F1 (ataques):     {optimal['max_f1_attacks']:.2f}")
        print(f"3) Óptimo Ciberseguridad:   {optimal['cybersec_optimal']:.2f}")

        cs_row = results_df[results_df["threshold"] == optimal["cybersec_optimal"]].iloc[0]
        print("\n=== MÉTRICAS EN UMBRAL ÓPTIMO CIBERSEGURIDAD ===")
        print(f"Accuracy:   {cs_row['accuracy']:.3f}")
        print(f"Precision1: {cs_row['precision_1']:.3f}")
        print(f"Recall1:    {cs_row['recall_1']:.3f}")
        print(f"F1_ataques: {cs_row['f1_1']:.3f}")
        print(f"FP: {int(cs_row['fp'])} | FN: {int(cs_row['fn'])}")

        return optimal

    def create_visualizations(self, results_df, y_test, y_proba):
        os.makedirs("results", exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Métricas vs umbral
        axes[0, 0].plot(results_df["threshold"], results_df["accuracy"], label="Accuracy")
        axes[0, 0].plot(results_df["threshold"], results_df["precision_1"], label="Precision (ataques)")
        axes[0, 0].plot(results_df["threshold"], results_df["recall_1"], label="Recall (ataques)")
        axes[0, 0].plot(results_df["threshold"], results_df["f1_1"], label="F1 (ataques)")
        axes[0, 0].set_title("Métricas vs Umbral")
        axes[0, 0].set_xlabel("Umbral")
        axes[0, 0].set_ylabel("Métrica")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Precision-Recall
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(rec, prec)
        axes[0, 1].plot(rec, prec, label=f"PR AUC={pr_auc:.3f}")
        axes[0, 1].set_title("Curva Precision-Recall")
        axes[0, 1].set_xlabel("Recall")
        axes[0, 1].set_ylabel("Precision")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        axes[1, 0].plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
        axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[1, 0].set_title("Curva ROC")
        axes[1, 0].set_xlabel("FPR")
        axes[1, 0].set_ylabel("TPR")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Distribución de probabilidades
        normal_probs = y_proba[y_test == 0]
        attack_probs = y_proba[y_test == 1]
        axes[1, 1].hist(normal_probs, bins=30, alpha=0.7, label="Normal", density=True)
        axes[1, 1].hist(attack_probs, bins=30, alpha=0.7, label="Ataques", density=True)
        axes[1, 1].axvline(0.5, color="black", linestyle="--", label="Umbral 0.5")
        axes[1, 1].set_title("Distribución de Probabilidades")
        axes[1, 1].set_xlabel("Probabilidad (p(ataque))")
        axes[1, 1].set_ylabel("Densidad")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig("results/threshold_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("[+] Guardado: results/threshold_analysis.png")

def main():
    dataset_path = "data/processed/dataset_features.csv"
    model_path = "models/mitm_detector.h5"

    os.makedirs("results", exist_ok=True)

    calibrator = CompatibleThresholdCalibrator(model_path=model_path)
    X, y, feature_cols = calibrator.load_and_prepare_data(dataset_path)

    optimal, results_df = calibrator.comprehensive_threshold_analysis(X, y)

    results_df.to_csv("results/threshold_analysis.csv", index=False)
    print("[+] Guardado: results/threshold_analysis.csv")

    # Guardar config recomendada
    rec_thr = optimal.get("cybersec_optimal", 0.5)
    best_row = results_df[results_df["threshold"] == rec_thr].iloc[0]

    config = {
        "recommended_threshold": float(rec_thr),
        "model_path": model_path,
        "expected_accuracy": float(best_row["accuracy"]),
        "expected_precision_attack": float(best_row["precision_1"]),
        "expected_recall_attack": float(best_row["recall_1"]),
        "expected_f1_attack": float(best_row["f1_1"]),
    }

    with open("results/optimal_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("[+] Guardado: results/optimal_config.json")
    print("\n✅ Calibración completada.")

if __name__ == "__main__":
    main()