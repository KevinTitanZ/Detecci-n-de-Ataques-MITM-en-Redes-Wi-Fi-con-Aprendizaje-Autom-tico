#!/usr/bin/env python3
"""
Monitor de víctimas en red 10.10.0.0/24 desde Kali Linux
No requiere acceso al MikroTik
"""

import subprocess
import time
import json
import re
from datetime import datetime
import os
import threading
from scapy.all import ARP, Ether, srp, sniff, get_if_list
import socket

class NetworkVictimMonitor:
    def __init__(self, network="10.10.0.0/24", interface="eth0"):
        self.network = network
        self.interface = interface
        self.victims = {}
        self.active_connections = {}
        self.log_file = "logs/network_victims.log"
        
        os.makedirs("logs", exist_ok=True)
        
        print(f"👥 MONITOR DE VÍCTIMAS DE RED")
        print(f"Red objetivo: {network}")
        print(f"Interfaz: {interface}")
    
    def arp_scan(self):
        """Escanear red con ARP"""
        
        try:
            print(f"🔍 Escaneando red {self.network}...")
            
            # Crear paquete ARP
            arp_request = ARP(pdst=self.network)
            broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
            arp_request_broadcast = broadcast / arp_request
            
            # Enviar y recibir
            answered_list = srp(arp_request_broadcast, timeout=2, verbose=False)[0]
            
            victims = {}
            
            for element in answered_list:
                victim = {
                    'ip': element[1].psrc,
                    'mac': element[1].hwsrc,
                    'hostname': self.get_hostname(element[1].psrc),
                    'vendor': self.get_mac_vendor(element[1].hwsrc),
                    'last_seen': datetime.now().isoformat(),
                    'response_time': element[1].time if hasattr(element[1], 'time') else 0
                }
                victims[element[1].hwsrc] = victim
            
            return victims
            
        except Exception as e:
            print(f"❌ Error en ARP scan: {e}")
            return {}
    
    def nmap_scan(self):
        """Escanear con nmap como respaldo"""
        
        try:
            cmd = f"nmap -sn {self.network}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
            
            victims = {}
            lines = result.stdout.split('\n')
            
            current_ip = None
            current_mac = None
            
            for line in lines:
                # Buscar IP
                ip_match = re.search(r'Nmap scan report for (\d+\.\d+\.\d+\.\d+)', line)
                if ip_match:
                    current_ip = ip_match.group(1)
                
                # Buscar MAC
                mac_match = re.search(r'MAC Address: ([0-9A-Fa-f:]{17})', line)
                if mac_match and current_ip:
                    current_mac = mac_match.group(1)
                    
                    victim = {
                        'ip': current_ip,
                        'mac': current_mac,
                        'hostname': self.get_hostname(current_ip),
                        'vendor': self.get_mac_vendor(current_mac),
                        'last_seen': datetime.now().isoformat(),
                        'method': 'nmap'
                    }
                    victims[current_mac] = victim
                    
                    current_ip = None
                    current_mac = None
            
            return victims
            
        except Exception as e:
            print(f"❌ Error en nmap scan: {e}")
            return {}
    
    def get_hostname(self, ip):
        """Obtener hostname por IP"""
        
        try:
            hostname = socket.gethostbyaddr(ip)[0]
            return hostname
        except:
            return "Desconocido"
    
    def get_mac_vendor(self, mac):
        """Obtener fabricante por MAC"""
        
        oui = mac.upper().replace(':', '')[:6]
        
        vendors = {
            # Apple
            '001122': 'Apple', '001B63': 'Apple', '3C07F4': 'Apple',
            '00F76F': 'Apple', '001EC2': 'Apple', '7C6D62': 'Apple',
            '001FF3': 'Apple', '28E02C': 'Apple', '40A6D9': 'Apple',
            '64B0A6': 'Apple', '78CA39': 'Apple', '7CF05F': 'Apple',
            '8C2937': 'Apple', '90FD61': 'Apple', 'A4B197': 'Apple',
            'A85B78': 'Apple', 'B0CA68': 'Apple', 'C0847A': 'Apple',
            'CC08E0': 'Apple', 'D0E140': 'Apple', 'E4CE8F': 'Apple',
            'F0B479': 'Apple',
            
            # Samsung
            '001632': 'Samsung', '0018AF': 'Samsung', '001D25': 'Samsung',
            '002454': 'Samsung', '0025D3': 'Samsung', '34E2FD': 'Samsung',
            '38AA3C': 'Samsung', '3C5AB4': 'Samsung', '5C0A5B': 'Samsung',
            '78D6F0': 'Samsung', '8C3AE3': 'Samsung', 'A020A6': 'Samsung',
            'E8508B': 'Samsung', 'EC1F72': 'Samsung',
            
            # Intel
            '00A0C9': 'Intel', '001B21': 'Intel', '0050F2': 'Intel',
            '001E64': 'Intel', '001F3B': 'Intel', '0024D7': 'Intel',
            '7085C2': 'Intel', '8086F2': 'Intel', 'A45E60': 'Intel',
            
            # Realtek
            '00E04C': 'Realtek', '525400': 'Realtek',
            
            # Xiaomi
            '34CE00': 'Xiaomi', '64B473': 'Xiaomi', '786A89': 'Xiaomi',
            '8CFABA': 'Xiaomi', 'F8A45F': 'Xiaomi',
            
            # Huawei
            '001E10': 'Huawei', '0025BC': 'Huawei', '28F366': 'Huawei',
            '4C549F': 'Huawei', '6C96CF': 'Huawei', '8C34FD': 'Huawei',
            
            # TP-Link
            '001A92': 'TP-Link', '50C7BF': 'TP-Link', 'A42BB0': 'TP-Link',
        }
        
        return vendors.get(oui, 'Desconocido')
    
    def monitor_traffic(self, duration=10):
        """Monitorear tráfico para detectar actividad"""
        
        print(f"📡 Monitoreando tráfico por {duration} segundos...")
        
        self.traffic_victims = set()
        
        def packet_handler(packet):
            if packet.haslayer('IP'):
                src_ip = packet['IP'].src
                if src_ip.startswith('10.10.0.'):
                    self.traffic_victims.add(src_ip)
        
        try:
            sniff(iface=self.interface, prn=packet_handler, timeout=duration, store=0)
        except Exception as e:
            print(f"⚠️ Error monitoreando tráfico: {e}")
        
        return self.traffic_victims
    
    def ping_sweep(self):
        """Ping sweep rápido"""
        
        print(f"🏓 Ping sweep en {self.network}...")
        
        victims = {}
        base_ip = self.network.split('/')[0].rsplit('.', 1)[0]
        
        def ping_host(ip):
            try:
                result = subprocess.run(['ping', '-c', '1', '-W', '1', ip], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # Obtener MAC de tabla ARP
                    arp_result = subprocess.run(['arp', '-n', ip], 
                                              capture_output=True, text=True)
                    mac_match = re.search(r'([0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2})', 
                                        arp_result.stdout)
                    
                    if mac_match:
                        mac = mac_match.group(1)
                        victims[mac] = {
                            'ip': ip,
                            'mac': mac,
                            'hostname': self.get_hostname(ip),
                            'vendor': self.get_mac_vendor(mac),
                            'last_seen': datetime.now().isoformat(),
                            'method': 'ping'
                        }
            except:
                pass
        
        # Ping en paralelo
        threads = []
        for i in range(1, 255):
            ip = f"{base_ip}.{i}"
            thread = threading.Thread(target=ping_host, args=(ip,))
            threads.append(thread)
            thread.start()
        
        # Esperar threads
        for thread in threads:
            thread.join()
        
        return victims
    
    def show_victims(self, victims):
        """Mostrar víctimas detectadas"""
        
        if not victims:
            print("👥 No hay víctimas detectadas")
            return
        
        print(f"\n👥 VÍCTIMAS DETECTADAS EN RED 10.10.0.0/24 ({len(victims)}):")
        print("="*90)
        print(f"{'IP':<15} {'MAC':<17} {'Hostname':<20} {'Vendor':<15} {'Método':<10}")
        print("-"*90)
        
        for mac, victim in victims.items():
            method = victim.get('method', 'arp')
            print(f"{victim['ip']:<15} {mac:<17} {victim['hostname']:<20} "
                  f"{victim['vendor']:<15} {method:<10}")
    
    def log_victims(self, victims):
        """Registrar víctimas en log"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'victim_count': len(victims),
            'victims': victims
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def start_monitoring(self, interval=30):
        """Iniciar monitoreo continuo"""
        
        print(f"\n🚀 INICIANDO MONITOREO DE RED")
        print(f"Red: {self.network}")
        print(f"Interfaz: {self.interface}")
        print(f"Intervalo: {interval} segundos")
        print(f"Presiona Ctrl+C para detener\n")
        
        try:
            while True:
                print(f"\n�� Escaneando red... {datetime.now().strftime('%H:%M:%S')}")
                
                # Múltiples métodos de detección
                arp_victims = self.arp_scan()
                nmap_victims = self.nmap_scan()
                
                # Combinar resultados
                all_victims = {}
                all_victims.update(arp_victims)
                all_victims.update(nmap_victims)
                
                # Detectar nuevas víctimas
                new_victims = []
                for mac, victim in all_victims.items():
                    if mac not in self.victims:
                        new_victims.append(victim)
                        print(f"🎯 NUEVA VÍCTIMA: {victim['ip']} ({victim['hostname']}) - {victim['vendor']}")
                
                # Actualizar lista
                self.victims.update(all_victims)
                
                # Mostrar víctimas
                self.show_victims(all_victims)
                
                # Log
                self.log_victims(all_victims)
                
                if new_victims:
                    print(f"\n🆕 {len(new_victims)} nuevas víctimas detectadas")
                
                print(f"\n⏳ Esperando {interval} segundos...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Monitoreo detenido")
            self.show_final_report()
    
    def show_final_report(self):
        """Mostrar reporte final"""
        
        print(f"\n📊 REPORTE FINAL")
        print(f"="*50)
        print(f"Total víctimas detectadas: {len(self.victims)}")
        print(f"Log guardado en: {self.log_file}")
        
        # Guardar reporte JSON
        report_file = f"results/network_victims_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("results", exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(self.victims, f, indent=2)
        
        print(f"Reporte guardado en: {report_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor de víctimas de red')
    parser.add_argument('-n', '--network', default='10.10.0.0/24', help='Red a escanear')
    parser.add_argument('-i', '--interface', default='eth0', help='Interfaz de red')
    parser.add_argument('-t', '--interval', type=int, default=30, help='Intervalo de escaneo')
    
    args = parser.parse_args()
    
    # Verificar permisos
    if os.geteuid() != 0:
        print("❌ Este script requiere permisos de root")
        print("Ejecuta: sudo python3 scripts/network_victim_monitor.py")
        return
    
    monitor = NetworkVictimMonitor(args.network, args.interface)
    monitor.start_monitoring(args.interval)

if __name__ == "__main__":
    main()
