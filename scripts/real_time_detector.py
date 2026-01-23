#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scapy.all import sniff, Dot11, IP, TCP, UDP, ARP
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import time
import json
from collections import deque
import argparse
import os

class RealTimeMITMDetector:
    def __init__(self, model_path='models/mitm_detector.h5', window_size=2.0, threshold=0.25):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = None
        self.window_size = window_size
        self.threshold = threshold
        self.packet_buffer = deque()
        self.running = False
        self.window_counter = 0
        
        try:
            self.scaler = joblib.load('models/scaler.pkl')
            print("[+] Scaler cargado")
        except:
            print("[-] Scaler no encontrado, predicciones sin normalizar")
        
        print(f"[+] Modelo: {model_path}")
        print(f"[+] Umbral: {self.threshold}")

    def extract_packet_features(self, packet):
        features = {
            'timestamp': float(packet.time), 'size': len(packet),
            'has_dot11': int(packet.haslayer(Dot11)), 'has_ip': int(packet.haslayer(IP)),
            'has_tcp': int(packet.haslayer(TCP)), 'has_udp': int(packet.haslayer(UDP)),
            'has_arp': int(packet.haslayer(ARP)),
        }
        if packet.haslayer(IP):
            ip = packet[IP]
            features.update({'ip_len': ip.len, 'ip_ttl': ip.ttl, 'ip_proto': ip.proto})
        else:
            features.update({'ip_len': 0, 'ip_ttl': 0, 'ip_proto': 0})
        
        if packet.haslayer(TCP):
            tcp = packet[TCP]
            features.update({'tcp_sport': tcp.sport, 'tcp_dport': tcp.dport, 
                           'tcp_flags': int(tcp.flags), 'tcp_window': tcp.window})
        else:
            features.update({'tcp_sport': 0, 'tcp_dport': 0, 'tcp_flags': 0, 'tcp_window': 0})
        return features

    def process_window(self, window_packets):
        if not window_packets: return None
        df = pd.DataFrame(window_packets)
        features = {
            'packet_count': len(df), 'total_bytes': df['size'].sum(),
            'size_mean': df['size'].mean(), 'size_std': df['size'].std() if len(df) > 1 else 0,
            'size_min': df['size'].min(), 'size_max': df['size'].max(),
            'ratio_ip': df['has_ip'].mean(), 'ratio_tcp': df['has_tcp'].mean(),
            'ratio_udp': df['has_udp'].mean(), 'ratio_arp': df['has_arp'].mean(),
        }
        if len(df) > 1:
            iats = df.sort_values('timestamp')['timestamp'].diff().dropna()
            features.update({'iat_mean': iats.mean(), 'iat_std': iats.std() if len(iats) > 1 else 0,
                           'iat_min': iats.min(), 'iat_max': iats.max()})
        else:
            features.update({'iat_mean': 0, 'iat_std': 0, 'iat_min': 0, 'iat_max': 0})
        
        ip_p = df[df['has_ip'] == 1]
        features.update({
            'ip_len_mean': ip_p['ip_len'].mean() if not ip_p.empty else 0,
            'ip_ttl_mean': ip_p['ip_ttl'].mean() if not ip_p.empty else 0,
            'ip_proto_var': ip_p['ip_proto'].nunique() if not ip_p.empty else 0
        })
        return features

    def packet_handler(self, packet):
        if not self.running: return
        self.packet_buffer.append(self.extract_packet_features(packet))
        curr = time.time()
        while self.packet_buffer and curr - self.packet_buffer[0]['timestamp'] > self.window_size * 2:
            self.packet_buffer.popleft()
        if len(self.packet_buffer) >= 10:
            self.analyze_current_window()

    def analyze_current_window(self):
        window_packets = list(self.packet_buffer)[-50:]
        window_features = self.process_window(window_packets)
        if window_features is None: return
        try:
            vector = np.array(list(window_features.values())).reshape(1, -1)
            if vector.shape[1] < 35:
                vector = np.pad(vector, ((0, 0), (0, 35 - vector.shape[1])), mode='constant')
            elif vector.shape[1] > 35:
                vector = vector[:, :35]
            
            if self.scaler:
                vector = self.scaler.transform(vector)
            
            prob = float(self.model.predict(vector, verbose=0)[0][0])
            self.emit_event(prob, window_features)
            
            if prob >= self.threshold:
                print(f"🚨 ATAQUE! Prob={prob:.3f} ARP={window_features['ratio_arp']:.2f}")
        except Exception as e:
            print(f"[-] Error: {e}")

    def emit_event(self, prob, features):
        self.window_counter += 1
        event = {
            "ts": time.time(), "prob_attack": prob, "threshold": self.threshold,
            "state": "ATAQUE" if prob >= self.threshold else "NORMAL",
            "packet_count": int(features['packet_count']), 
            "total_bytes": int(features['total_bytes']),
            "ratio_arp": float(features['ratio_arp'])
        }
        os.makedirs('results', exist_ok=True)
        with open('results/alerts.jsonl', 'a') as f:
            f.write(json.dumps(event) + "\n")

    def start_monitoring(self, interface):
        self.running = True
        if os.path.exists('results/alerts.jsonl'): 
            os.remove('results/alerts.jsonl')
        print(f"[*] Monitoreando {interface}... (Ctrl+C para detener)")
        try:
            sniff(iface=interface, prn=self.packet_handler, store=0)
        except KeyboardInterrupt:
            print("\n[!] Detenido")
        finally:
            self.running = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--interface', default='eth0')
    parser.add_argument('--threshold', type=float, default=0.25)
    args = parser.parse_args()
    detector = RealTimeMITMDetector(threshold=args.threshold)
    detector.start_monitoring(args.interface)