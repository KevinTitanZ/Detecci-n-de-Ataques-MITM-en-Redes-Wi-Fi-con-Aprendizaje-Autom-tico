#!/usr/bin/env python3
"""
Optimización de hiperparámetros para el modelo MITM
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, make_scorer, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Bidirectional, Dense, Dropout, Reshape
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import os

def create_optimized_model(conv_filters=64, lstm_units=50, dropout_rate=0.3, learning_rate=0.001):
    """Crear modelo con hiperparámetros configurables"""
    model = Sequential([
        Reshape((input_dim, 1), input_shape=(input_dim,)),
        
        # Capas convolucionales optimizadas
        Conv1D(conv_filters, 3, activation='relu', padding='same'),
        Dropout(dropout_rate),
        Conv1D(conv_filters//2, 3, activation='relu', padding='same'),
        Dropout(dropout_rate),
        
        # Capas Bi-LSTM optimizadas
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Bidirectional(LSTM(lstm_units//2)),
        
        # Capas densas
        Dense(50, activation='relu'),
        Dropout(dropout_rate + 0.1),
        Dense(25, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def optimize_hyperparameters():
    """Optimizar hiperparámetros del modelo"""
    
    # Cargar datos
    dataset_path = 'data/processed/dataset_features.csv'
    if not os.path.exists(dataset_path):
        print("[-] Dataset no encontrado")
        return
    
    df = pd.read_csv(dataset_path)
    exclude_cols = ['label', 'filename', 'window_start', 'window_end', 'window_id']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_columns].values
    y = df['label'].values
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    global input_dim
    input_dim = X_scaled.shape[1]
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("=== OPTIMIZACIÓN DE HIPERPARÁMETROS ===")
    
    # Configuraciones a probar
    configs = [
        {'conv_filters': 32, 'lstm_units': 25, 'dropout_rate': 0.2, 'epochs': 30},
        {'conv_filters': 64, 'lstm_units': 50, 'dropout_rate': 0.3, 'epochs': 30},
        {'conv_filters': 128, 'lstm_units': 75, 'dropout_rate': 0.4, 'epochs': 30},
        {'conv_filters': 64, 'lstm_units': 100, 'dropout_rate': 0.25, 'epochs': 40},
    ]
    
    best_score = 0
    best_config = None
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuración {i+1}/{len(configs)} ---")
        print(f"Conv filters: {config['conv_filters']}")
        print(f"LSTM units: {config['lstm_units']}")
        print(f"Dropout: {config['dropout_rate']}")
        
        # Crear y entrenar modelo
        model = create_optimized_model(
            conv_filters=config['conv_filters'],
            lstm_units=config['lstm_units'],
            dropout_rate=config['dropout_rate']
        )
        
        # Entrenar
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=config['epochs'],
            batch_size=16,
            verbose=0
        )
        
        # Evaluar
        y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
        
        # Calcular métricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Score combinado (priorizando recall para ataques)
        combined_score = (recall * 0.4) + (precision * 0.3) + (accuracy * 0.3)
        
        result = {
            'config': config,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'combined_score': combined_score
        }
        
        results.append(result)
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"Score combinado: {combined_score:.3f}")
        
        if combined_score > best_score:
            best_score = combined_score
            best_config = config
            # Guardar mejor modelo
            model.save('models/best_mitm_detector.h5')
    
    print(f"\n=== MEJOR CONFIGURACIÓN ===")
    print(f"Score: {best_score:.3f}")
    print(f"Configuración: {best_config}")
    
    # Guardar resultados
    results_df = pd.DataFrame([
        {
            'conv_filters': r['config']['conv_filters'],
            'lstm_units': r['config']['lstm_units'],
            'dropout_rate': r['config']['dropout_rate'],
            'epochs': r['config']['epochs'],
            'accuracy': r['accuracy'],
            'precision': r['precision'],
            'recall': r['recall'],
            'f1': r['f1'],
            'combined_score': r['combined_score']
        }
        for r in results
    ])
    
    results_df.to_csv('results/hyperparameter_optimization.csv', index=False)
    print(f"\n[+] Resultados guardados en: results/hyperparameter_optimization.csv")
    print(f"[+] Mejor modelo guardado en: models/best_mitm_detector.h5")

if __name__ == "__main__":
    optimize_hyperparameters()
