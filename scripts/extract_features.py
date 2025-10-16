#!/usr/bin/env python3
"""
Extractor de características de archivos PCAP para detección MITM - VERSIÓN CORREGIDA
"""

from scapy.all import rdpcap, Dot11, RadioTap, Ether, IP, TCP, UDP
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    def __init__(self, window_size=2.0):
        self.window_size = window_size
    
    def extract_from_pcap(self, pcap_path):
        """Extraer características de un archivo PCAP"""
        print(f"Procesando: {pcap_path}")
        
        try:
            packets = rdpcap(str(pcap_path))
            if not packets:
                print(f"  [-] Archivo vacío: {pcap_path}")
                return pd.DataFrame()
            
            print(f"  [+] Paquetes cargados: {len(packets)}")
            
            # Extraer información básica de cada paquete
            packet_data = []
            
            for i, packet in enumerate(packets):
                pkt_info = {
                    'timestamp': float(packet.time),
                    'size': len(packet),
                    'has_dot11': int(packet.haslayer(Dot11)),
                    'has_ip': int(packet.haslayer(IP)),
                    'has_tcp': int(packet.haslayer(TCP)),
                    'has_udp': int(packet.haslayer(UDP)),
                    'has_ether': int(packet.haslayer(Ether)),
                }
                
                # Información 802.11 si está disponible
                if packet.haslayer(Dot11):
                    dot11 = packet[Dot11]
                    pkt_info.update({
                        'dot11_type': getattr(dot11, 'type', 0),
                        'dot11_subtype': getattr(dot11, 'subtype', 0),
                        'dot11_retry': 1 if (hasattr(dot11, 'FCfield') and dot11.FCfield & 0x08) else 0,
                        'dot11_to_ds': 1 if (hasattr(dot11, 'FCfield') and dot11.FCfield & 0x01) else 0,
                        'dot11_from_ds': 1 if (hasattr(dot11, 'FCfield') and dot11.FCfield & 0x02) else 0,
                    })
                else:
                    pkt_info.update({
                        'dot11_type': -1, 'dot11_subtype': -1, 'dot11_retry': 0,
                        'dot11_to_ds': 0, 'dot11_from_ds': 0
                    })
                
                # Información IP si está disponible
                if packet.haslayer(IP):
                    ip = packet[IP]
                    pkt_info.update({
                        'ip_len': ip.len,
                        'ip_ttl': ip.ttl,
                        'ip_proto': ip.proto,
                    })
                else:
                    pkt_info.update({
                        'ip_len': 0, 'ip_ttl': 0, 'ip_proto': 0
                    })
                
                # Información TCP si está disponible
                if packet.haslayer(TCP):
                    tcp = packet[TCP]
                    pkt_info.update({
                        'tcp_sport': tcp.sport,
                        'tcp_dport': tcp.dport,
                        'tcp_flags': tcp.flags,
                        'tcp_window': tcp.window,
                    })
                else:
                    pkt_info.update({
                        'tcp_sport': 0, 'tcp_dport': 0, 'tcp_flags': 0, 'tcp_window': 0
                    })
                
                # Información de señal si está disponible (RadioTap)
                if packet.haslayer(RadioTap):
                    radiotap = packet[RadioTap]
                    pkt_info['rssi'] = getattr(radiotap, 'dBm_AntSignal', 0)
                else:
                    pkt_info['rssi'] = 0
                
                packet_data.append(pkt_info)
            
            if not packet_data:
                print(f"  [-] No se pudieron extraer datos de paquetes")
                return pd.DataFrame()
            
            # Convertir a DataFrame
            df = pd.DataFrame(packet_data)
            print(f"  [+] DataFrame creado con {len(df)} filas y {len(df.columns)} columnas")
            
            # Crear ventanas temporales
            windows_df = self.create_windows(df)
            print(f"  [+] Ventanas creadas: {len(windows_df)}")
            
            return windows_df
            
        except Exception as e:
            print(f"  [-] Error procesando {pcap_path}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def create_windows(self, df):
        """Crear ventanas temporales y extraer características"""
        if df.empty:
            return pd.DataFrame()
        
        # Ordenar por timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calcular inter-arrival times
        df['iat'] = df['timestamp'].diff().fillna(0)
        
        # Crear ventanas
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        
        windows = []
        current_time = start_time
        window_id = 0
        
        while current_time < end_time:
            window_end = current_time + self.window_size
            
            # Filtrar paquetes en esta ventana
            window_mask = (df['timestamp'] >= current_time) & (df['timestamp'] < window_end)
            window_df = df[window_mask].copy()
            
            if len(window_df) > 0:
                features = self.extract_window_features(window_df, window_id)
                if features:  # Solo agregar si se extrajeron características
                    features['window_start'] = current_time
                    features['window_end'] = window_end
                    windows.append(features)
                    window_id += 1
            
            current_time = window_end
        
        result_df = pd.DataFrame(windows) if windows else pd.DataFrame()
        return result_df
    
    def extract_window_features(self, window_df, window_id):
        """Extraer características de una ventana temporal"""
        try:
            features = {}
            
            # Características básicas
            features['window_id'] = window_id
            features['packet_count'] = len(window_df)
            features['total_bytes'] = window_df['size'].sum()
            features['duration'] = window_df['timestamp'].max() - window_df['timestamp'].min()
            
            # Estadísticas de tamaño de paquetes
            sizes = window_df['size']
            features['size_mean'] = float(sizes.mean())
            features['size_std'] = float(sizes.std()) if len(sizes) > 1 else 0.0
            features['size_min'] = float(sizes.min())
            features['size_max'] = float(sizes.max())
            features['size_median'] = float(sizes.median())
            features['size_q25'] = float(sizes.quantile(0.25))
            features['size_q75'] = float(sizes.quantile(0.75))
            
            # Estadísticas de inter-arrival time
            iats = window_df['iat'][window_df['iat'] > 0]  # Excluir el primer 0
            if len(iats) > 0:
                features['iat_mean'] = float(iats.mean())
                features['iat_std'] = float(iats.std()) if len(iats) > 1 else 0.0
                features['iat_min'] = float(iats.min())
                features['iat_max'] = float(iats.max())
                features['iat_median'] = float(iats.median())
            else:
                features.update({
                    'iat_mean': 0.0, 'iat_std': 0.0, 'iat_min': 0.0,
                    'iat_max': 0.0, 'iat_median': 0.0
                })
            
            # Ratios de tipos de paquetes
            total_pkts = len(window_df)
            features['ratio_dot11'] = float(window_df['has_dot11'].sum()) / total_pkts
            features['ratio_ip'] = float(window_df['has_ip'].sum()) / total_pkts
            features['ratio_tcp'] = float(window_df['has_tcp'].sum()) / total_pkts
            features['ratio_udp'] = float(window_df['has_udp'].sum()) / total_pkts
            features['ratio_ether'] = float(window_df['has_ether'].sum()) / total_pkts
            
            # Características 802.11 específicas
            dot11_pkts = window_df[window_df['has_dot11'] == 1]
            if len(dot11_pkts) > 0:
                features['dot11_mgmt_ratio'] = float((dot11_pkts['dot11_type'] == 0).sum()) / len(dot11_pkts)
                features['dot11_ctrl_ratio'] = float((dot11_pkts['dot11_type'] == 1).sum()) / len(dot11_pkts)
                features['dot11_data_ratio'] = float((dot11_pkts['dot11_type'] == 2).sum()) / len(dot11_pkts)
                features['dot11_retry_ratio'] = float(dot11_pkts['dot11_retry'].mean())
            else:
                features.update({
                    'dot11_mgmt_ratio': 0.0, 'dot11_ctrl_ratio': 0.0,
                    'dot11_data_ratio': 0.0, 'dot11_retry_ratio': 0.0
                })
            
            # Características IP
            ip_pkts = window_df[window_df['has_ip'] == 1]
            if len(ip_pkts) > 0:
                features['ip_len_mean'] = float(ip_pkts['ip_len'].mean())
                features['ip_ttl_mean'] = float(ip_pkts['ip_ttl'].mean())
                features['ip_proto_variety'] = float(ip_pkts['ip_proto'].nunique())
            else:
                features.update({
                    'ip_len_mean': 0.0, 'ip_ttl_mean': 0.0, 'ip_proto_variety': 0.0
                })
            
            # Características TCP
            tcp_pkts = window_df[window_df['has_tcp'] == 1]
            if len(tcp_pkts) > 0:
                features['tcp_port_variety'] = float(pd.concat([tcp_pkts['tcp_sport'], tcp_pkts['tcp_dport']]).nunique())
                features['tcp_flags_variety'] = float(tcp_pkts['tcp_flags'].nunique())
                features['tcp_window_mean'] = float(tcp_pkts['tcp_window'].mean())
            else:
                features.update({
                    'tcp_port_variety': 0.0, 'tcp_flags_variety': 0.0, 'tcp_window_mean': 0.0
                })
            
            # RSSI statistics
            rssi_values = window_df['rssi'][window_df['rssi'] != 0]
            if len(rssi_values) > 0:
                features['rssi_mean'] = float(rssi_values.mean())
                features['rssi_std'] = float(rssi_values.std()) if len(rssi_values) > 1 else 0.0
                features['rssi_min'] = float(rssi_values.min())
                features['rssi_max'] = float(rssi_values.max())
            else:
                features.update({
                    'rssi_mean': 0.0, 'rssi_std': 0.0, 'rssi_min': 0.0, 'rssi_max': 0.0
                })
            
            return features
            
        except Exception as e:
            print(f"    [-] Error extrayendo características de ventana {window_id}: {e}")
            return None

def process_dataset(input_dir, output_file):
    """Procesar todo el dataset"""
    extractor = FeatureExtractor(window_size=2.0)
    all_features = []
    
    input_path = Path(input_dir)
    
    print("=== PROCESANDO DATASET ===")
    
    # Procesar archivos normales
    normal_dir = input_path / "normal"
    if normal_dir.exists():
        normal_files = list(normal_dir.glob("*.pcap"))
        print(f"Archivos normales encontrados: {len(normal_files)}")
        
        for pcap_file in normal_files:
            features_df = extractor.extract_from_pcap(pcap_file)
            if not features_df.empty:
                features_df['label'] = 0  # Normal
                features_df['filename'] = pcap_file.name
                all_features.append(features_df)
                print(f"  [+] {pcap_file.name}: {len(features_df)} ventanas")
            else:
                print(f"  [-] {pcap_file.name}: No se pudieron extraer características")
    
    # Procesar archivos de ataque
    mitm_dir = input_path / "mitm"
    if mitm_dir.exists():
        mitm_files = list(mitm_dir.glob("*.pcap"))
        print(f"Archivos de ataque encontrados: {len(mitm_files)}")
        
        for pcap_file in mitm_files:
            features_df = extractor.extract_from_pcap(pcap_file)
            if not features_df.empty:
                features_df['label'] = 1  # Ataque
                features_df['filename'] = pcap_file.name
                all_features.append(features_df)
                print(f"  [+] {pcap_file.name}: {len(features_df)} ventanas")
            else:
                print(f"  [-] {pcap_file.name}: No se pudieron extraer características")
    
    # Combinar y guardar
    if all_features:
        final_dataset = pd.concat(all_features, ignore_index=True)
        
        # Rellenar NaN con 0
        final_dataset = final_dataset.fillna(0)
        
        # Crear directorio si no existe
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar
        final_dataset.to_csv(output_file, index=False)
        
        print(f"\n=== DATASET CREADO ===")
        print(f"Archivo: {output_file}")
        print(f"Total ventanas: {len(final_dataset)}")
        print(f"Ventanas normales: {sum(final_dataset['label'] == 0)}")
        print(f"Ventanas de ataque: {sum(final_dataset['label'] == 1)}")
        print(f"Características: {len([col for col in final_dataset.columns if col not in ['label', 'filename', 'window_start', 'window_end']])}")
        
        return final_dataset
    else:
        print("[-] No se encontraron archivos para procesar")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description='Extraer características de dataset MITM')
    parser.add_argument('--input', default='data/raw', help='Directorio con archivos PCAP')
    parser.add_argument('--output', default='data/processed/dataset_features.csv', 
                       help='Archivo CSV de salida')
    
    args = parser.parse_args()
    
    dataset = process_dataset(args.input, args.output)
    
    if not dataset.empty:
        print("\n=== ESTADÍSTICAS BÁSICAS ===")
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        print(dataset[numeric_cols].describe())

if __name__ == "__main__":
    main()
