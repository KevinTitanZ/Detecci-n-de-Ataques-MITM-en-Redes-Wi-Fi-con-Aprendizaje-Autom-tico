#!/usr/bin/env python3
"""
Calibración de umbrales compatible con el modelo actual
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json

class CompatibleThresholdCalibrator:
    def __init__(self, model_path='models/mitm_detector.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = StandardScaler()
        
        # Obtener el número de características esperadas por el modelo
        self.expected_features = self.model.input_shape[1]
        print(f"[+] Modelo espera {self.expected_features} características")
        
    def load_and_prepare_data(self, csv_path):
        """Cargar y preparar datos compatibles con el modelo"""
        df = pd.read_csv(csv_path)
        exclude_cols = ['label', 'filename', 'window_start', 'window_end', 'window_id']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_columns].values
        y = df['label'].values
        
        print(f"Dataset cargado: {len(X)} muestras")
        print(f"Características en dataset: {X.shape[1]}")
        print(f"Clase 0 (normal): {sum(y == 0)}")
        print(f"Clase 1 (ataques): {sum(y == 1)}")
        
        # Ajustar número de características
        if X.shape[1] < self.expected_features:
            # Agregar características faltantes con ceros
            missing_features = self.expected_features - X.shape[1]
            X_padded = np.pad(X, ((0, 0), (0, missing_features)), mode='constant', constant_values=0)
            print(f"[+] Agregadas {missing_features} características con valor 0")
            X = X_padded
        elif X.shape[1] > self.expected_features:
            # Recortar características extra
            X = X[:, :self.expected_features]
            print(f"[+] Recortadas {X.shape[1] - self.expected_features} características extra")
        
        return X, y, feature_columns
    
    def comprehensive_threshold_analysis(self, X, y):
        """Análisis exhaustivo de umbrales"""
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Obtener probabilidades
        y_proba = self.model.predict(X_test_scaled, verbose=0)
        
        # Manejar diferentes formatos de salida del modelo
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]  # Probabilidad de clase positiva
        else:
            y_proba = y_proba.flatten()
        
        print("\n=== ANÁLISIS DE UMBRALES ===")
        
        # Probar umbrales
        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            
            # Calcular matriz de confusión
            try:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            except:
                continue
            
            # Métricas
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
            
            # Score para ciberseguridad (prioriza recall)
            cybersec_score = (recall_1 * 0.5) + (precision_1 * 0.3) + (accuracy * 0.2)
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision_1': precision_1,
                'recall_1': recall_1,
                'f1_1': f1_1,
                'cybersec_score': cybersec_score,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            })
        
        results_df = pd.DataFrame(results)
        
        # Encontrar umbrales óptimos
        optimal_thresholds = self.find_optimal_thresholds(results_df)
        
        # Crear visualizaciones
        self.create_visualizations(results_df, y_test, y_proba)
        
        return optimal_thresholds, results_df
    
    def find_optimal_thresholds(self, results_df):
        """Encontrar umbrales óptimos"""
        
        print("\n=== UMBRALES ÓPTIMOS ===")
        
        optimal = {}
        
        # 1. Máximo recall para ataques
        max_recall_1 = results_df.loc[results_df['recall_1'].idxmax()]
        optimal['max_recall_attacks'] = max_recall_1['threshold']
        print(f"\n🎯 1. MÁXIMO RECALL ATAQUES")
        print(f"   Umbral: {max_recall_1['threshold']:.3f}")
        print(f"   Recall: {max_recall_1['recall_1']:.3f}")
        print(f"   Precision: {max_recall_1['precision_1']:.3f}")
        print(f"   Accuracy: {max_recall_1['accuracy']:.3f}")
        print(f"   Falsos negativos: {max_recall_1['fn']}")
        
        # 2. Recall >= 0.9 con mejor precision
        high_recall = results_df[results_df['recall_1'] >= 0.9]
        if len(high_recall) > 0:
            best_high_recall = high_recall.loc[high_recall['precision_1'].idxmax()]
            optimal['high_recall_90'] = best_high_recall['threshold']
            print(f"\n🔥 2. RECALL ≥ 90%")
            print(f"   Umbral: {best_high_recall['threshold']:.3f}")
            print(f"   Recall: {best_high_recall['recall_1']:.3f}")
            print(f"   Precision: {best_high_recall['precision_1']:.3f}")
            print(f"   Accuracy: {best_high_recall['accuracy']:.3f}")
        
        # 3. Máximo F1 para ataques
        max_f1_1 = results_df.loc[results_df['f1_1'].idxmax()]
        optimal['max_f1_attacks'] = max_f1_1['threshold']
        print(f"\n⚖️ 3. BALANCE ÓPTIMO (MAX F1)")
        print(f"   Umbral: {max_f1_1['threshold']:.3f}")
        print(f"   F1: {max_f1_1['f1_1']:.3f}")
        print(f"   Recall: {max_f1_1['recall_1']:.3f}")
        print(f"   Precision: {max_f1_1['precision_1']:.3f}")
        
        # 4. Máximo score ciberseguridad
        max_cybersec = results_df.loc[results_df['cybersec_score'].idxmax()]
        optimal['cybersec_optimal'] = max_cybersec['threshold']
        print(f"\n🛡️ 4. ÓPTIMO CIBERSEGURIDAD")
        print(f"   Umbral: {max_cybersec['threshold']:.3f}")
        print(f"   Score: {max_cybersec['cybersec_score']:.3f}")
        print(f"   Recall: {max_cybersec['recall_1']:.3f}")
        print(f"   Precision: {max_cybersec['precision_1']:.3f}")
        
        return optimal
    
    def create_visualizations(self, results_df, y_test, y_proba):
        """Crear visualizaciones"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Métricas vs Umbral
        axes[0, 0].plot(results_df['threshold'], results_df['accuracy'], 'b-', label='Accuracy', linewidth=2)
        axes[0, 0].plot(results_df['threshold'], results_df['precision_1'], 'r-', label='Precision', linewidth=2)
        axes[0, 0].plot(results_df['threshold'], results_df['recall_1'], 'g-', label='Recall', linewidth=2)
        axes[0, 0].plot(results_df['threshold'], results_df['f1_1'], 'm-', label='F1', linewidth=2)
        axes[0, 0].set_xlabel('Umbral')
        axes[0, 0].set_ylabel('Métrica')
        axes[0, 0].set_title('Métricas vs Umbral')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        axes[0, 1].plot(recall, precision, 'b-', linewidth=2, label=f'PR AUC = {pr_auc:.3f}')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Curva Precision-Recall')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        axes[1, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC AUC = {roc_auc:.3f}')
        axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[1, 0].set_xlabel('Tasa Falsos Positivos')
        axes[1, 0].set_ylabel('Tasa Verdaderos Positivos')
        axes[1, 0].set_title('Curva ROC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Distribución de probabilidades
        normal_probs = y_proba[y_test == 0]
        attack_probs = y_proba[y_test == 1]
        
        axes[1, 1].hist(normal_probs, bins=20, alpha=0.7, label='Normal', color='blue', density=True)
        axes[1, 1].hist(attack_probs, bins=20, alpha=0.7, label='Ataques', color='red', density=True)
        axes[1, 1].axvline(x=0.5, color='black', linestyle='--', label='Umbral 0.5')
        axes[1, 1].set_xlabel('Probabilidad')
        axes[1, 1].set_ylabel('Densidad')
        axes[1, 1].set_title('Distribución de Probabilidades')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('results/threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n[+] Visualizaciones guardadas en: results/threshold_analysis.png")
    
    def generate_recommendations(self, optimal_thresholds, results_df):
        """Generar recomendaciones"""
        
        print(f"\n" + "="*50)
        print(f"🎯 RECOMENDACIONES PARA TU TESIS")
        print(f"="*50)
        
        # Recomendación principal
        if 'cybersec_optimal' in optimal_thresholds:
            main_threshold = optimal_thresholds['cybersec_optimal']
            main_row = results_df[results_df['threshold'] == main_threshold].iloc[0]
            
            print(f"\n🏆 RECOMENDACIÓN PRINCIPAL:")
            print(f"   Umbral: {main_threshold:.3f}")
            print(f"   ✅ Recall: {main_row['recall_1']:.1%}")
            print(f"   ✅ Precision: {main_row['precision_1']:.1%}")
            print(f"   ✅ Accuracy: {main_row['accuracy']:.1%}")
            print(f"   ⚠️ Falsos negativos: {main_row['fn']}")
            print(f"   ⚠️ Falsos positivos: {main_row['fp']}")
        
        # Guardar configuración
        config = {
            'recommended_threshold': main_threshold if 'cybersec_optimal' in optimal_thresholds else 0.5,
            'model_path': 'models/mitm_detector.h5',
            'scaler_path': 'models/scaler.pkl',
            'expected_recall': main_row['recall_1'] if 'cybersec_optimal' in optimal_thresholds else 0.0,
            'expected_precision': main_row['precision_1'] if 'cybersec_optimal' in optimal_thresholds else 0.0,
            'expected_accuracy': main_row['accuracy'] if 'cybersec_optimal' in optimal_thresholds else 0.0
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/optimal_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n[+] Configuración guardada en: results/optimal_config.json")

def main():
    dataset_path = 'data/processed/dataset_features.csv'
    
    if not os.path.exists(dataset_path):
        print(f"[-] Dataset no encontrado: {dataset_path}")
        return
    
    os.makedirs('results', exist_ok=True)
    
    print("🚀 INICIANDO CALIBRACIÓN COMPATIBLE")
    print("="*50)
    
    # Crear calibrador
    calibrator = CompatibleThresholdCalibrator()
    
    # Cargar datos
    X, y, feature_columns = calibrator.load_and_prepare_data(dataset_path)
    
    # Análisis de umbrales
    optimal_thresholds, results_df = calibrator.comprehensive_threshold_analysis(X, y)
    
    # Generar recomendaciones
    calibrator.generate_recommendations(optimal_thresholds, results_df)
    
    # Guardar resultados
    results_df.to_csv('results/threshold_analysis.csv', index=False)
    
    print(f"\n" + "="*50)
    print(f"✅ CALIBRACIÓN COMPLETADA")
    print(f"="*50)
    print(f"📁 Archivos generados:")
    print(f"   - results/threshold_analysis.csv")
    print(f"   - results/threshold_analysis.png")
    print(f"   - results/optimal_config.json")

if __name__ == "__main__":
    main()
