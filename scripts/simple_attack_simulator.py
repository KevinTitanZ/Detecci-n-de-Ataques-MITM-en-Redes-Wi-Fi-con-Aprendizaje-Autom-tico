#!/usr/bin/env python3
"""
Simulador simple de ataques MITM para tu red 192.168.11.x
"""

from scapy.all import *
import time
import random
import subprocess
import os
import signal
from datetime import datetime

class SimpleAttackSimulator:
    def __init__(self, interface="wlan0", your_ip="192.168.11.66"):
        self.interface = interface
        self.your_ip = your_ip
        self.gateway = "192.168.11.1"  # Gateway típico
        self.running = False
    
    def simulate_arp_anomalies(self, duration=300):
        """Simular anomalías ARP sin hacer ataque real"""
        print(f"[+] Simulando anomalías ARP por {duration} segundos")
        
        start_time = time.time()
        self.running = True
        
        while self.running and (time.time() - start_time) < duration:
            # Generar paquetes ARP anómalos
            
            # 1. ARP requests excesivos
            for _ in range(5):
                fake_ip = f"192.168.11.{random.randint(100, 200)}"
                arp_req = ARP(op=1, pdst=fake_ip, psrc=self.your_ip)
                send(arp_req, verbose=False)
                time.sleep(0.1)
            
            # 2. ARP replies no solicitados
            fake_mac = "02:00:00:00:00:01"
            arp_reply = ARP(op=2, pdst=self.gateway, hwdst="ff:ff:ff:ff:ff:ff",
                           psrc=f"192.168.11.{random.randint(50, 99)}", hwsrc=fake_mac)
            send(arp_reply, verbose=False)
            
            # 3. Gratuitous ARP anómalos
            grat_arp = ARP(op=2, pdst=self.your_ip, hwdst="ff:ff:ff:ff:ff:ff",
                          psrc=self.your_ip, hwsrc=fake_mac)
            send(grat_arp, verbose=False)
            
            time.sleep(2)
            
            if int(time.time() - start_time) % 30 == 0:
                print(f"[+] Anomalías ARP... {int(time.time() - start_time)}s")
    
    def simulate_dns_anomalies(self, duration=300):
        """Simular consultas DNS anómalas"""
        print(f"[+] Simulando anomalías DNS por {duration} segundos")
        
        start_time = time.time()
        self.running = True
        
        suspicious_domains = [
            "malicious-site.com", "phishing-bank.net", "fake-update.org",
            "suspicious-download.info", "malware-host.biz"
        ]
        
        while self.running and (time.time() - start_time) < duration:
            # Consultas DNS sospechosas
            domain = random.choice(suspicious_domains)
            
            # Consulta DNS anómala
            dns_query = IP(dst="8.8.8.8")/UDP(dport=53)/DNS(rd=1, qd=DNSQR(qname=domain))
            send(dns_query, verbose=False)
            
            # Múltiples consultas rápidas (comportamiento de malware)
            for _ in range(3):
                dns_query = IP(dst="1.1.1.1")/UDP(dport=53)/DNS(rd=1, qd=DNSQR(qname=f"random{random.randint(1000,9999)}.com"))
                send(dns_query, verbose=False)
                time.sleep(0.05)
            
            time.sleep(5)
            
            if int(time.time() - start_time) % 60 == 0:
                print(f"[+] Anomalías DNS... {int(time.time() - start_time)}s")
    
    def simulate_port_scan(self, duration=300):
        """Simular escaneo de puertos"""
        print(f"[+] Simulando escaneo de puertos por {duration} segundos")
        
        start_time = time.time()
        self.running = True
        
        target_ips = [f"192.168.11.{i}" for i in range(1, 10)]
        common_ports = [21, 22, 23, 25, 53, 80, 110, 443, 993, 995]
        
        while self.running and (time.time() - start_time) < duration:
            target_ip = random.choice(target_ips)
            port = random.choice(common_ports)
            
            # SYN scan
            syn_packet = IP(dst=target_ip)/TCP(dport=port, flags="S")
            send(syn_packet, verbose=False)
            
            # NULL scan
            null_packet = IP(dst=target_ip)/TCP(dport=port, flags="")
            send(null_packet, verbose=False)
            
            time.sleep(0.5)
            
            if int(time.time() - start_time) % 60 == 0:
                print(f"[+] Escaneo de puertos... {int(time.time() - start_time)}s")

def capture_with_simulation(attack_type, duration=300, interface="wlan0"):
    """Capturar tráfico mientras se simula ataque"""
    
    # Crear directorio
    os.makedirs('data/raw/mitm', exist_ok=True)
    
    # Nombre del archivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/raw/mitm/mitm_{attack_type}_{timestamp}.pcap"
    
    # Iniciar captura
    cmd = f"sudo tcpdump -i {interface} -w {filename}"
    capture_process = subprocess.Popen(cmd.split())
    
    print(f"[+] Captura iniciada: {filename}")
    print(f"[+] Simulando ataque: {attack_type}")
    
    # Crear simulador
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
    
    parser = argparse.ArgumentParser(description='Simulador simple de ataques MITM')
    parser.add_argument('--attack', choices=['arp', 'dns', 'portscan'], 
                       required=True, help='Tipo de ataque a simular')
    parser.add_argument('--duration', type=int, default=300, help='Duración en segundos')
    parser.add_argument('--interface', default='wlan0', help='Interfaz de red')
    
    args = parser.parse_args()
    
    if os.geteuid() != 0:
        print("[-] Requiere permisos de root: sudo python3 ...")
        exit(1)
    
    filename = capture_with_simulation(args.attack, args.duration, args.interface)
    print(f"\n[+] Listo! Archivo: {filename}")
    print("Siguiente paso: python3 scripts/extract_features.py")
