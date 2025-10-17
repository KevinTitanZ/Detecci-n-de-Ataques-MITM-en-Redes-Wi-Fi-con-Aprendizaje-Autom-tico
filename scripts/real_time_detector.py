#!/usr/bin/env python3
"""
Sistema de detección MITM en tiempo real
"""

import numpy as np
import pandas as pd
from scapy.all import sniff, Dot11, IP, TCP, UDP, ARP
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import time
from collections import deque
import threading
import argparse

class RealTimeMITMDetector:
    def __init__(self, model_path='models/mitm_detector.h5', window_size=2.0):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = StandardScaler()  # Cargar scaler entrenado
        self.window_size = window_size
        self.packet_buffer = deque()
        self.running = False
        
        # Cargar scaler si existe
        try:
            self.scaler = joblib.load('models/scaler.pkl')
            print("[+] Scaler cargado")
        except:
            print("[-] No se pudo cargar scaler, usando uno nuevo")
        
        print(f"[+] Modelo cargado: {model_path}")
    
    def extract_packet_features(self, packet):
        """Extraer características de un paquete individual"""
        features = {
            'timestamp': float(packet.time),
            'size': len(packet),
            'has_dot11': int(packet.haslayer(Dot11)),
            'has_ip': int(packet.haslayer(IP)),
            'has_tcp': int(packet.haslayer(TCP)),
            'has_udp': int(packet.haslayer(UDP)),
            'has_arp': int(packet.haslayer(ARP)),
        }
        
        # Características IP
        if packet.haslayer(IP):
            ip = packet[IP]
            features.update({
                'ip_len': ip.len,
                'ip_ttl': ip.ttl,
                'ip_proto': ip.proto,
            })
        else:
            features.update({'ip_len': 0, 'ip_ttl': 0, 'ip_proto': 0})
        
        # Características TCP
        if packet.haslayer(TCP):
            tcp = packet[TCP]
            features.update({
                'tcp_sport': tcp.sport,
                'tcp_dport': tcp.dport,
                'tcp_flags': tcp.flags,
                'tcp_window': tcp.window,
            })
        else:
            features.update({
                'tcp_sport': 0, 'tcp_dport': 0, 
                'tcp_flags': 0, 'tcp_window': 0
            })
        
        return features
    
    def process_window(self, window_packets):
        """Procesar ventana de paquetes y extraer características"""
        if not window_packets:
            return None
        
        # Convertir a DataFrame
        df = pd.DataFrame(window_packets)
        
        # Calcular características de la ventana
        features = {
            'packet_count': len(df),
            'total_bytes': df['size'].sum(),
            'size_mean': df['size'].mean(),
            'size_std': df['size'].std() if len(df) > 1 else 0,
            'size_min': df['size'].min(),
            'size_max': df['size'].max(),
            'ratio_ip': df['has_ip'].mean(),
            'ratio_tcp': df['has_tcp'].mean(),
            'ratio_udp': df['has_udp'].mean(),
            'ratio_arp': df['has_arp'].mean(),
        }
        
        # Inter-arrival times
        if len(df) > 1:
            df_sorted = df.sort_values('timestamp')
            iats = df_sorted['timestamp'].diff().dropna()
            features.update({
                'iat_mean': iats.mean(),
                'iat_std': iats.std() if len(iats) > 1 else 0,
                'iat_min': iats.min(),
                'iat_max': iats.max(),
            })
        else:
            features.update({
                'iat_mean': 0, 'iat_std': 0, 'iat_min': 0, 'iat_max': 0
            })
        
        # Características IP específicas
        ip_packets = df[df['has_ip'] == 1]
        if len(ip_packets) > 0:
            features.update({
                'ip_len_mean': ip_packets['ip_len'].mean(),
                'ip_ttl_mean': ip_packets['ip_ttl'].mean(),
                'ip_proto_variety': ip_packets['ip_proto'].nunique(),
            })
        else:
            features.update({
                'ip_len_mean': 0, 'ip_ttl_mean': 0, 'ip_proto_variety': 0
            })
        
        return features
    
    def packet_handler(self, packet):
        """Manejar cada paquete capturado"""
        if not self.running:
            return
        
        # Extraer características del paquete
        pkt_features = self.extract_packet_features(packet)
        
        # Agregar al buffer
        self.packet_buffer.append(pkt_features)
        
        # Mantener solo paquetes de los últimos X segundos
        current_time = time.time()
        while (self.packet_buffer and 
               current_time - self.packet_buffer[0]['timestamp'] > self.window_size * 2):
            self.packet_buffer.popleft()
        
        # Procesar ventana si tenemos suficientes paquetes
        if len(self.packet_buffer) >= 10:  # Mínimo 10 paquetes
            self.analyze_current_window()
    
    def analyze_current_window(self):
        """Analizar la ventana actual de paquetes"""
        if len(self.packet_buffer) < 10:
            return
        
        # Tomar últimos paquetes para la ventana
        window_packets = list(self.packet_buffer)[-50:]  # Últimos 50 paquetes
        
        # Extraer características de la ventana
        window_features = self.process_window(window_packets)
        
        if window_features is None:
            return
        
        try:
            # Preparar para predicción
            # Nota: Aquí necesitarías ajustar las características para que coincidan
            # con las que usaste durante el entrenamiento
            feature_vector = np.array(list(window_features.values())).reshape(1, -1)
            
            # Normalizar (si tienes el scaler entrenado)
            try:
                feature_vector_scaled = self.scaler.transform(feature_vector)
            except:
                feature_vector_scaled = feature_vector
            
            # Hacer predicción
            prediction = self.model.predict(feature_vector_scaled, verbose=0)[0][0]
            
            # Mostrar resultado si es sospechoso
            if prediction > 0.5:  # Umbral de detección
                print(f"\n🚨 POSIBLE ATAQUE DETECTADO!")
                print(f"   Probabilidad: {prediction:.3f}")
                print(f"   Paquetes en ventana: {window_features['packet_count']}")
                print(f"   Bytes totales: {window_features['total_bytes']}")
                print(f"   Timestamp: {time.strftime('%H:%M:%S')}")
                
                # Log detallado
                self.log_detection(window_features, prediction)
        
        except Exception as e:
            print(f"[-] Error en predicción: {e}")
    
    def log_detection(self, features, probability):
        """Registrar detección en archivo"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        with open('results/real_time_detections.log', 'a') as f:
            f.write(f"{timestamp} - DETECTION - Probability: {probability:.3f}\n")
            f.write(f"  Features: {features}\n\n")
    
    def start_monitoring(self, interface='wlan0', duration=None):
        """Iniciar monitoreo en tiempo real"""
        print(f"=== INICIANDO DETECCIÓN EN TIEMPO REAL ===")
        print(f"Interfaz: {interface}")
        print(f"Ventana: {self.window_size} segundos")
        print("Presiona Ctrl+C para detener")
        
        self.running = True
        
        try:
            # Iniciar captura
            sniff(
                iface=interface,
                prn=self.packet_handler,
                timeout=duration,
                stop_filter=lambda x: not self.running
            )
        except KeyboardInterrupt:
            print("\n[!] Monitoreo detenido por usuario")
        except Exception as e:
            print(f"[-] Error en monitoreo: {e}")
        finally:
            self.running = False

def main():
    parser = argparse.ArgumentParser(description='Detector MITM en tiempo real')
    parser.add_argument('--interface', default='wlan0', help='Interfaz de red')
    parser.add_argument('--model', default='models/mitm_detector.h5', help='Modelo a usar')
    parser.add_argument('--duration', type=int, help='Duración en segundos')
    parser.add_argument('--window', type=float, default=2.0, help='Tamaño de ventana')
    
    args = parser.parse_args()
    
    # Verificar modelo
    if not os.path.exists(args.model):
        print(f"[-] Modelo no encontrado: {args.model}")
        return
    
    # Crear detector
    detector = RealTimeMITMDetector(
        model_path=args.model,
        window_size=args.window
    )
    
    # Iniciar monitoreo
    detector.start_monitoring(
        interface=args.interface,
        duration=args.duration
    )

if __name__ == "__main__":
    import os
    main()
