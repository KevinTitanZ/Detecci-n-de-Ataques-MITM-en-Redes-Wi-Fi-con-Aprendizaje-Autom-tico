#!/usr/bin/env python3
"""
Modelo MITM avanzado con class weights, focal loss y optimizaciones
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Bidirectional, Dense, Dropout, Reshape, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json

class AdvancedMITMDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.class_weights = None
        
    def focal_loss(self, gamma=2.0, alpha=0.75):
        """Implementar Focal Loss para manejar desbalance de clases"""
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Calcular componentes del focal loss
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_loss = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t)
            
            return tf.reduce_mean(focal_loss)
        
        return focal_loss_fixed
    
    def weighted_binary_crossentropy(self, pos_weight):
        """Binary crossentropy con peso para clase positiva"""
        def loss(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Aplicar peso a la clase positiva
            loss = -(pos_weight * y_true * tf.math.log(y_pred) + 
                    (1 - y_true) * tf.math.log(1 - y_pred))
            
            return tf.reduce_mean(loss)
        
        return loss
    
    def create_advanced_model(self, input_dim, loss_type='focal'):
        """Crear modelo avanzado con diferentes tipos de pérdida"""
        
        model = Sequential([
            Reshape((input_dim, 1), input_shape=(input_dim,)),
            
            # Bloque CNN 1
            Conv1D(64, 3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Bloque CNN 2
            Conv1D(32, 3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Bloque CNN 3 (adicional)
            Conv1D(16, 3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Capas Bi-LSTM
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
            Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
            
            # Capas densas con regularización
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        # Configurar función de pérdida
        if loss_type == 'focal':
            loss_fn = self.focal_loss(gamma=2.0, alpha=0.75)
        elif loss_type == 'weighted':
            pos_weight = self.class_weights[1] / self.class_weights[0] if self.class_weights else 2.0
            loss_fn = self.weighted_binary_crossentropy(pos_weight)
        else:
            loss_fn = 'binary_crossentropy'
        
        # Compilar modelo
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss_fn,
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def load_and_prepare_data(self, csv_path):
        """Cargar y preparar datos con validación"""
        
        df = pd.read_csv(csv_path)
        exclude_cols = ['label', 'filename', 'window_start', 'window_end', 'window_id']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns].values
        y = df['label'].values
        
        print(f"Dataset cargado: {len(X)} muestras, {len(self.feature_columns)} características")
        print(f"Clase 0 (normal): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
        print(f"Clase 1 (ataques): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
        
        # Verificar balance de clases
        if sum(y == 1) == 0:
            raise ValueError("No hay muestras de ataque en el dataset")
        
        if sum(y == 0) == 0:
            raise ValueError("No hay muestras normales en el dataset")
        
        # Calcular class weights
        classes = np.unique(y)
        class_weights_array = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = {i: class_weights_array[i] for i in range(len(classes))}
        
        print(f"\nClass weights calculados:")
        print(f"Clase 0 (normal): {self.class_weights[0]:.3f}")
        print(f"Clase 1 (ataques): {self.class_weights[1]:.3f}")
        
        return X, y
    
    def train_with_cross_validation(self, X, y, n_splits=5):
        """Entrenar con validación cruzada para mayor robustez"""
        
        print(f"\n=== ENTRENAMIENTO CON VALIDACIÓN CRUZADA ({n_splits} folds) ===")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        cv_recalls = []
        cv_precisions = []
        
        fold = 1
        for train_idx, val_idx in skf.split(X, y):
            print(f"\n--- Fold {fold}/{n_splits} ---")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Normalizar
            scaler_fold = StandardScaler()
            X_train_scaled = scaler_fold.fit_transform(X_train_fold)
            X_val_scaled = scaler_fold.transform(X_val_fold)
            
            # Crear modelo
            model_fold = self.create_advanced_model(X_train_scaled.shape[1], loss_type='focal')
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_recall',
                    patience=15,
                    restore_best_weights=True,
                    mode='max'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=8,
                    min_lr=1e-6
                )
            ]
            
            # Entrenar
            history = model_fold.fit(
                X_train_scaled, y_train_fold,
                validation_data=(X_val_scaled, y_val_fold),
                epochs=100,
                batch_size=16,
                class_weight=self.class_weights,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluar
            y_pred_proba = model_fold.predict(X_val_scaled, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calcular métricas
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            
            accuracy = accuracy_score(y_val_fold, y_pred)
            precision = precision_score(y_val_fold, y_pred)
            recall = recall_score(y_val_fold, y_pred)
            
            cv_scores.append(accuracy)
            cv_recalls.append(recall)
            cv_precisions.append(precision)
            
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            
            fold += 1
        
        # Resultados de CV
        print(f"\n=== RESULTADOS VALIDACIÓN CRUZADA ===")
        print(f"Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        print(f"Precision: {np.mean(cv_precisions):.3f} ± {np.std(cv_precisions):.3f}")
        print(f"Recall: {np.mean(cv_recalls):.3f} ± {np.std(cv_recalls):.3f}")
        
        return {
            'accuracy_mean': np.mean(cv_scores),
            'accuracy_std': np.std(cv_scores),
            'precision_mean': np.mean(cv_precisions),
            'precision_std': np.std(cv_precisions),
            'recall_mean': np.mean(cv_recalls),
            'recall_std': np.std(cv_recalls)
        }
    
    def train_final_model(self, X, y):
        """Entrenar modelo final con todos los datos"""
        
        print(f"\n=== ENTRENANDO MODELO FINAL ===")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Crear modelo final
        self.model = self.create_advanced_model(X_train_scaled.shape[1], loss_type='focal')
        
        # Callbacks avanzados
        callbacks = [
            EarlyStopping(
                monitor='val_recall',
                patience=20,
                restore_best_weights=True,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                'models/best_mitm_detector_advanced.h5',
                monitor='val_recall',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Entrenar
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=150,
            batch_size=16,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar modelo final
        self.evaluate_final_model(X_test_scaled, y_test, history)
        
        return history
    
    def evaluate_final_model(self, X_test, y_test, history):
        """Evaluación exhaustiva del modelo final"""
        
        print(f"\n=== EVALUACIÓN MODELO FINAL ===")
        
        # Predicciones
        y_pred_proba = self.model.predict(X_test, verbose=0)
        
        # Probar múltiples umbrales
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            
            print(f"\n--- Umbral: {threshold} ---")
            print(classification_report(y_test, y_pred))
            
            # Calcular F1 para ataques
            from sklearn.metrics import f1_score
            f1_attacks = f1_score(y_test, y_pred)
            
            if f1_attacks > best_f1:
                best_f1 = f1_attacks
                best_threshold = threshold
        
        print(f"\n🎯 MEJOR UMBRAL: {best_threshold} (F1 ataques: {best_f1:.3f})")
        
        # Crear visualizaciones
        self.create_training_visualizations(history, X_test, y_test, y_pred_proba)
        
        # Guardar configuración
        config = {
            'model_path': 'models/best_mitm_detector_advanced.h5',
            'scaler_path': 'models/scaler_advanced.pkl',
            'best_threshold': best_threshold,
            'best_f1_score': best_f1,
            'class_weights': self.class_weights,
            'feature_columns': self.feature_columns
        }
        
        with open('results/advanced_model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        return best_threshold
    
    def create_training_visualizations(self, history, X_test, y_test, y_pred_proba):
        """Crear visualizaciones del entrenamiento"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Recall
        axes[0, 2].plot(history.history['recall'], label='Training Recall')
        axes[0, 2].plot(history.history['val_recall'], label='Validation Recall')
        axes[0, 2].set_title('Model Recall')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Recall')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Precision
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. AUC
        axes[1, 1].plot(history.history['auc'], label='Training AUC')
        axes[1, 1].plot(history.history['val_auc'], label='Validation AUC')
        axes[1, 1].set_title('Model AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Distribución de probabilidades
        normal_probs = y_pred_proba[y_test == 0]
        attack_probs = y_pred_proba[y_test == 1]
        
        axes[1, 2].hist(normal_probs, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
        axes[1, 2].hist(attack_probs, bins=30, alpha=0.7, label='Ataques', color='red', density=True)
        axes[1, 2].axvline(x=0.5, color='black', linestyle='--', label='Umbral 0.5')
        axes[1, 2].set_xlabel('Probabilidad')
        axes[1, 2].set_ylabel('Densidad')
        axes[1, 2].set_title('Distribución de Probabilidades')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('results/advanced_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n[+] Visualizaciones guardadas en: results/advanced_training_results.png")
    
    def save_model(self):
        """Guardar modelo y componentes"""
        
        os.makedirs('models', exist_ok=True)
        
        # Guardar modelo
        self.model.save('models/mitm_detector_advanced.h5')
        
        # Guardar scaler
        joblib.dump(self.scaler, 'models/scaler_advanced.pkl')
        
        # Guardar metadatos
        metadata = {
            'feature_columns': self.feature_columns,
            'class_weights': self.class_weights,
            'model_architecture': 'CNN + Bi-LSTM with Focal Loss',
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[+] Modelo guardado: models/mitm_detector_advanced.h5")
        print(f"[+] Scaler guardado: models/scaler_advanced.pkl")
        print(f"[+] Metadatos guardados: models/model_metadata.json")

def main():
    dataset_path = 'data/processed/dataset_features.csv'
    
    if not os.path.exists(dataset_path):
        print(f"[-] Dataset no encontrado: {dataset_path}")
        print("    Ejecuta primero: python3 scripts/extract_features.py")
        return
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("🚀 INICIANDO ENTRENAMIENTO AVANZADO")
    print("="*60)
    
    # Crear detector
    detector = AdvancedMITMDetector()
    
    # Cargar datos
    X, y = detector.load_and_prepare_data(dataset_path)
    
    if len(X) < 50:
        print("⚠️ Dataset pequeño. Recomendado: más de 50 muestras por clase")
        print("   Continuando con entrenamiento...")
    
    # Validación cruzada
    cv_results = detector.train_with_cross_validation(X, y)
    
    # Entrenar modelo final
    history = detector.train_final_model(X, y)
    
    # Guardar modelo
    detector.save_model()
    
    print(f"\n" + "="*60)
    print(f"✅ ENTRENAMIENTO AVANZADO COMPLETADO")
    print(f"="*60)
    print(f"📊 Resultados CV:")
    print(f"   Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
    print(f"   Precision: {cv_results['precision_mean']:.3f} ± {cv_results['precision_std']:.3f}")
    print(f"   Recall: {cv_results['recall_mean']:.3f} ± {cv_results['recall_std']:.3f}")
    print(f"\n📁 Archivos generados:")
    print(f"   - models/mitm_detector_advanced.h5")
    print(f"   - models/best_mitm_detector_advanced.h5")
    print(f"   - models/scaler_advanced.pkl")
    print(f"   - results/advanced_training_results.png")
    print(f"   - results/advanced_model_config.json")

if __name__ == "__main__":
    main()
