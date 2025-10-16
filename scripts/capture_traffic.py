#!/usr/bin/env python3
"""
Script para capturar tráfico Wi-Fi para dataset MITM
Uso: python3 capture_traffic.py --mode normal --duration 300
"""

import subprocess
import time
import os
import signal
import argparse
from datetime import datetime

class TrafficCapture:
    def __init__(self, interface="wlan0", output_dir="data/raw"):
        self.interface = interface
        self.output_dir = output_dir
        self.capture_process = None
        
    def start_capture(self, filename):
        """Iniciar captura con tcpdump"""
        filepath = os.path.join(self.output_dir, filename)
        cmd = f"sudo tcpdump -i {self.interface} -w {filepath}"
        
        try:
            self.capture_process = subprocess.Popen(cmd.split())
            print(f"[+] Captura iniciada: {filename}")
            print(f"[+] Comando: {cmd}")
            return True
        except Exception as e:
            print(f"[-] Error iniciando captura: {e}")
            return False
    
    def stop_capture(self):
        """Detener captura"""
        if self.capture_process:
            self.capture_process.send_signal(signal.SIGTERM)
            self.capture_process.wait()
            print("[+] Captura detenida")
    
    def capture_normal_traffic(self, duration=300):
        """Capturar tráfico normal"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"normal/normal_{timestamp}.pcap"
        
        if self.start_capture(filename):
            print(f"\n=== CAPTURANDO TRÁFICO NORMAL ===")
            print(f"Duración: {duration} segundos ({duration//60} minutos)")
            print("INSTRUCCIONES:")
            print("1. Navega sitios web (google.com, youtube.com)")
            print("2. Ve un video corto")
            print("3. Descarga un archivo pequeño")
            print("4. Usa WhatsApp Web o similar")
            print("5. Mantén actividad normal de internet")
            print("\nPresiona Ctrl+C para detener antes de tiempo")
            
            try:
                time.sleep(duration)
            except KeyboardInterrupt:
                print("\n[!] Detenido por usuario")
            
            self.stop_capture()
            return filename
        return None

def main():
    parser = argparse.ArgumentParser(description='Capturar tráfico para dataset MITM')
    parser.add_argument('--mode', choices=['normal', 'attack'], required=True,
                       help='Tipo de tráfico a capturar')
    parser.add_argument('--duration', type=int, default=300,
                       help='Duración en segundos (default: 300)')
    parser.add_argument('--interface', default='wlan0mon',
                       help='Interfaz de red (default: wlan0mon)')
    
    args = parser.parse_args()
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists('data/raw'):
        print("[-] Error: Ejecuta desde el directorio raíz del proyecto")
        print("    cd ~/mitm_detection_project")
        return
    
    capturer = TrafficCapture(interface=args.interface)
    
    if args.mode == 'normal':
        filename = capturer.capture_normal_traffic(args.duration)
        if filename:
            print(f"\n[+] Archivo guardado: {filename}")
    else:
        print("Modo attack aún no implementado. Usa --mode normal por ahora.")

if __name__ == "__main__":
    main()
