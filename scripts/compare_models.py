#!/usr/bin/env python3
"""docker ps
Comparación de modelos para detección MITM
Compara CNN+Bi-LSTM vs modelos tradicionales
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Modelos a comparar
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import time
from datetime import datetime

class ModelComparator:
    def __init__(self, dataset_path='data/processed/dataset_features.csv'):
        self.dataset_path = dataset_path
        self.results = {}
        self.models = {}
        self.scaler = StandardScaler()
        
        # Definir modelos a comparar
        self.model_configs = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
            'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        print(f"🔬 COMPARADOR DE MODELOS INICIALIZADO")
        print(f"Modelos a evaluar: {len(self.model_configs)}")
    
    def load_data(self):
        """Cargar y preparar datos"""
        
        print(f"\n📊 CARGANDO DATOS...")
        df = pd.read_csv(self.dataset_path)
        
        # Excluir columnas no numéricas
        exclude_cols = ['label', 'filename', 'window_start', 'window_end', 'window_id']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_columns].values
        y = df['label'].values
        
        print(f"✅ Dataset cargado:")
        print(f"   - Muestras: {len(X)}")
        print(f"   - Características: {X.shape[1]}")
        print(f"   - Clase 0 (normal): {sum(y == 0)}")
        print(f"   - Clase 1 (ataques): {sum(y == 1)}")
        
        return X, y, feature_columns
    
    def evaluate_model(self, name, model, X_train, X_test, y_train, y_test):
        """Evaluar un modelo específico"""
        
        print(f"\n🔄 Evaluando: {name}")
        start_time = time.time()
        
        try:
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Métricas básicas
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # AUC si hay probabilidades
            roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            
            # Matriz de confusión
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Tiempo de entrenamiento
            training_time = time.time() - start_time
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'training_time': training_time,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            print(f"   ✅ Completado en {training_time:.2f}s")
            print(f"   📈 F1: {f1:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def evaluate_cnn_lstm(self, X_test, y_test):
        """Evaluar modelo CNN+Bi-LSTM existente"""
        
        print(f"\n🧠 Evaluando: CNN+Bi-LSTM (Tu modelo)")
        
        try:
            # Cargar modelo existente
            model_path = 'models/mitm_detector.h5'
            if not os.path.exists(model_path):
                print(f"   ❌ Modelo no encontrado: {model_path}")
                return None
            
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load('models/scaler.pkl')
            
            # Ajustar características si es necesario
            expected_features = model.input_shape[1]
            if X_test.shape[1] < expected_features:
                missing_features = expected_features - X_test.shape[1]
                X_test_padded = np.pad(X_test, ((0, 0), (0, missing_features)), mode='constant', constant_values=0)
                X_test = X_test_padded
            elif X_test.shape[1] > expected_features:
                X_test = X_test[:, :expected_features]
            
            # Normalizar
            X_test_scaled = scaler.transform(X_test)
            
            # Predicciones
            y_proba = model.predict(X_test_scaled, verbose=0)
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                y_proba = y_proba[:, 1]
            else:
                y_proba = y_proba.flatten()
            
            # Usar umbral óptimo (0.25)
            y_pred = (y_proba > 0.25).astype(int)
            
            # Métricas
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            # Matriz de confusión
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_f1_mean': None,  # No aplica para modelo pre-entrenado
                'cv_f1_std': None,
                'training_time': None,  # Ya está entrenado
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            print(f"   ✅ Evaluación completada")
            print(f"   �� F1: {f1:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def run_comparison(self):
        """Ejecutar comparación completa"""
        
        print(f"\n🚀 INICIANDO COMPARACIÓN DE MODELOS")
        print(f"="*50)
        
        # Cargar datos
        X, y, feature_columns = self.load_data()
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar para modelos tradicionales
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Evaluar modelos tradicionales
        for name, model in self.model_configs.items():
            result = self.evaluate_model(name, model, X_train_scaled, X_test_scaled, y_train, y_test)
            if result:
                self.results[name] = result
                self.models[name] = model
        
        # Evaluar CNN+Bi-LSTM
        cnn_result = self.evaluate_cnn_lstm(X_test, y_test)
        if cnn_result:
            self.results['CNN+Bi-LSTM'] = cnn_result
        
        # Generar reporte
        self.generate_comparison_report()
        self.create_visualizations()
        
        print(f"\n✅ COMPARACIÓN COMPLETADA")
        print(f"Resultados guardados en: results/model_comparison.csv")
    
    def generate_comparison_report(self):
        """Generar reporte de comparación"""
        
        print(f"\n📊 GENERANDO REPORTE DE COMPARACIÓN")
        
        # Crear DataFrame con resultados
        comparison_data = []
        
        for name, result in self.results.items():
            comparison_data.append({
                'Modelo': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'ROC-AUC': result['roc_auc'],
                'CV F1 Mean': result['cv_f1_mean'],
                'CV F1 Std': result['cv_f1_std'],
                'Training Time (s)': result['training_time'],
                'True Positives': result['tp'],
                'True Negatives': result['tn'],
                'False Positives': result['fp'],
                'False Negatives': result['fn']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Ordenar por F1-Score
        df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
        
        # Guardar CSV
        os.makedirs('results', exist_ok=True)
        df_comparison.to_csv('results/model_comparison.csv', index=False)
        
        # Mostrar top 5
        print(f"\n🏆 TOP 5 MODELOS (por F1-Score):")
        print(f"="*60)
        
        for i, row in df_comparison.head().iterrows():
            print(f"{row['Modelo']:20} | F1: {row['F1-Score']:.3f} | Recall: {row['Recall']:.3f} | Precision: {row['Precision']:.3f}")
        
        # Análisis específico para ciberseguridad
        print(f"\n🛡️ ANÁLISIS PARA CIBERSEGURIDAD (Recall prioritario):")
        print(f"="*60)
        
        df_by_recall = df_comparison.sort_values('Recall', ascending=False)
        for i, row in df_by_recall.head().iterrows():
            fn_rate = row['False Negatives'] / (row['True Positives'] + row['False Negatives']) * 100
            print(f"{row['Modelo']:20} | Recall: {row['Recall']:.3f} | FN Rate: {fn_rate:.1f}% | F1: {row['F1-Score']:.3f}")
        
        return df_comparison
    
    def create_visualizations(self):
        """Crear visualizaciones comparativas"""
        
        print(f"\n📈 CREANDO VISUALIZACIONES...")
        
        # Preparar datos para gráficos
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comparación de Modelos para Detección MITM', fontsize=16, fontweight='bold')
        
        # 1. Comparación de métricas principales
        metric_data = {metric: [self.results[model][metric] for model in models] for metric in metrics}
        
        ax = axes[0, 0]
        x = np.arange(len(models))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            if all(v is not None for v in metric_data[metric]):
                ax.bar(x + i*width, metric_data[metric], width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Modelos')
        ax.set_ylabel('Score')
        ax.set_title('Métricas por Modelo')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Recall vs Precision
        ax = axes[0, 1]
        recalls = [self.results[model]['recall'] for model in models]
        precisions = [self.results[model]['precision'] for model in models]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        for i, model in enumerate(models):
            ax.scatter(recalls[i], precisions[i], c=[colors[i]], s=100, label=model, alpha=0.7)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Recall vs Precision')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 3. F1-Score ranking
        ax = axes[0, 2]
        f1_scores = [self.results[model]['f1_score'] for model in models]
        sorted_indices = np.argsort(f1_scores)[::-1]
        
        ax.barh(range(len(models)), [f1_scores[i] for i in sorted_indices], color=colors)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([models[i] for i in sorted_indices])
        ax.set_xlabel('F1-Score')
        ax.set_title('Ranking por F1-Score')
        ax.grid(True, alpha=0.3)
        
        # 4. Matriz de confusión para top 3
        top_3_models = sorted(models, key=lambda x: self.results[x]['f1_score'], reverse=True)[:3]
        
        for idx, model in enumerate(top_3_models):
            if idx >= 3:
                break
                
            ax = axes[1, idx]
            result = self.results[model]
            cm = np.array([[result['tn'], result['fp']], [result['fn'], result['tp']]])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
            ax.set_title(f'{model}\nF1: {result["f1_score"]:.3f}')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Gráfico adicional: Tiempo de entrenamiento vs Performance
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        training_times = []
        f1_scores = []
        model_names = []
        
        for model in models:
            if self.results[model]['training_time'] is not None:
                training_times.append(self.results[model]['training_time'])
                f1_scores.append(self.results[model]['f1_score'])
                model_names.append(model)
        
        if training_times:
            scatter = ax.scatter(training_times, f1_scores, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
            
            for i, model in enumerate(model_names):
                ax.annotate(model, (training_times[i], f1_scores[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            ax.set_xlabel('Tiempo de Entrenamiento (segundos)')
            ax.set_ylabel('F1-Score')
            ax.set_title('Tiempo de Entrenamiento vs Performance')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('results/time_vs_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"✅ Visualizaciones guardadas en:")
        print(f"   - results/model_comparison.png")
        print(f"   - results/time_vs_performance.png")

def main():
    """Función principal"""
    
    dataset_path = 'data/processed/dataset_features.csv'
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset no encontrado: {dataset_path}")
        print(f"Ejecuta primero: python3 scripts/extract_features.py")
        return
    
    # Crear comparador
    comparator = ModelComparator(dataset_path)
    
    # Ejecutar comparación
    comparator.run_comparison()
    
    print(f"\n🎯 COMPARACIÓN COMPLETADA")
    print(f"📁 Archivos generados:")
    print(f"   - results/model_comparison.csv")
    print(f"   - results/model_comparison.png")
    print(f"   - results/time_vs_performance.png")

if __name__ == "__main__":
    main()
