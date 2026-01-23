#!/usr/bin/env python3
"""
Modelo CNN + Bi-LSTM para detección MITM
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Bidirectional, Dense, Dropout, Reshape
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # ← NUEVO: Para guardar el scaler
import os

class MITMDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def load_data(self, csv_path):
        """Cargar y preparar datos"""
        print(f"Cargando datos de: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Seleccionar columnas de características (excluir metadatos)
        exclude_cols = ['label', 'filename', 'window_start', 'window_end']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns].values
        y = df['label'].values
        
        print(f"Características: {len(self.feature_columns)}")
        print(f"Muestras: {len(X)}")
        print(f"Clase 0 (normal): {sum(y == 0)}")
        print(f"Clase 1 (ataque): {sum(y == 1)}")
        
        return X, y
    
    def create_model(self, input_shape):
        """Crear modelo CNN + Bi-LSTM"""
        model = Sequential([
            # Reshape para Conv1D
            Reshape((input_shape, 1), input_shape=(input_shape,)),
            
            # Capas convolucionales
            Conv1D(64, 3, activation='relu', padding='same'),
            Dropout(0.2),
            Conv1D(32, 3, activation='relu', padding='same'),
            Dropout(0.2),
            
            # Capas Bi-LSTM
            Bidirectional(LSTM(50, return_sequences=True)),
            Bidirectional(LSTM(25)),
            
            # Capas densas
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, X, y, test_size=0.2, validation_split=0.2, epochs=50):
        """Entrenar el modelo"""
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Normalizar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # ← NUEVO: Guardar el scaler INMEDIATAMENTE después de fit
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("✅ Scaler guardado en: models/scaler.pkl")
        
        # Crear modelo
        self.model = self.create_model(X_train_scaled.shape[1])
        
        print("\n=== ARQUITECTURA DEL MODELO ===")
        self.model.summary()
        
        # Entrenar
        print("\n=== ENTRENANDO MODELO ===")
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        # Evaluar
        print("\n=== EVALUACIÓN ===")
        y_pred = (self.model.predict(X_test_scaled) > 0.5).astype(int)
        
        print("Reporte de clasificación:")
        print(classification_report(y_test, y_pred))
        
        # Matriz de confusión
        os.makedirs('results', exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión - CNN + Bi-LSTM')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✅ Matriz de confusión guardada en: results/confusion_matrix.png")
        
        # Guardar modelo
        self.model.save('models/mitm_detector.h5')
        print("✅ Modelo guardado en: models/mitm_detector.h5")
        
        # ← NUEVO: Guardar historial de entrenamiento
        history_df = pd.DataFrame(history.history)
        history_df.to_csv('results/training_history.csv', index=False)
        print("✅ Historial guardado en: results/training_history.csv")
        
        # ← NUEVO: Gráfica de entrenamiento
        self.plot_training_history(history)
        
        return history, X_test_scaled, y_test
    
    def plot_training_history(self, history):
        """Graficar historial de entrenamiento"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Train')
        axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Train')
        axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        print("✅ Gráficas de entrenamiento guardadas en: results/training_history.png")
        plt.close()
    
    def predict(self, X):
        """Hacer predicciones"""
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def main():
    # Verificar que existe el dataset
    dataset_path = 'data/processed/combined_dataset.csv'  # ← CAMBIADO
    
    # Si no existe el combinado, buscar el original
    if not os.path.exists(dataset_path):
        dataset_path = 'data/processed/dataset_features.csv'
    
    if not os.path.exists(dataset_path):
        print(f"[-] No se encontró el dataset: {dataset_path}")
        print("    Ejecuta primero: python scripts/merge_datasets.py")
        return
    
    # Crear detector
    detector = MITMDetector()
    
    # Cargar datos
    X, y = detector.load_data(dataset_path)
    
    if len(X) < 10:
        print("[-] Dataset muy pequeño. Captura más tráfico primero.")
        return
    
    # Entrenar modelo
    history, X_test, y_test = detector.train(X, y, epochs=30)
    
    print("\n" + "="*60)
    print("🎉 ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print("Archivos generados:")
    print("  ✅ models/mitm_detector.h5 (modelo entrenado)")
    print("  ✅ models/scaler.pkl (normalizador)")
    print("  ✅ results/confusion_matrix.png")
    print("  ✅ results/training_history.png")
    print("  ✅ results/training_history.csv")
    print("="*60)
    print("\n🚀 Siguiente paso:")
    print("   sudo venv/bin/python scripts/real_time_detector.py --interface eth0")
    print("="*60)

if __name__ == "__main__":
    main()