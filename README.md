
# Para Poder entrar al proyecto
cd ~/mitm_detection_project  

# Para capturar el trafico de red:
python3 scripts/capture_traffic.py --mode normal --duration 300 --interface wlan0

# Para capturar el trafico de red Malo:
sudo python3 scripts/simple_attack_simulator.py --attack arp --duration 120
sudo python3 scripts/simple_attack_simulator.py --attack dns --duration 120
sudo python3 scripts/simple_attack_simulator.py --attack portscan --duration 120
-- UNIVERSAL
chmod +x scripts/universal_attack_simulator.py
# ARP (lo que ya hiciste)
sudo python3 scripts/universal_attack_simulator.py --attack arp --duration 120

# DNS anómalo
sudo python3 scripts/universal_attack_simulator.py --attack dns --duration 120

# Escaneo de puertos
sudo python3 scripts/universal_attack_simulator.py --attack portscan --duration 120

# Inyección de tráfico malicioso
sudo python3 scripts/universal_attack_simulator.py --attack injection --duration 120

# Todos los ataques (ARP + DNS + Portscan + Inyección)
sudo python3 scripts/universal_attack_simulator.py --attack all --duration 300


# sirve para activiar el lugar donde estan las descargar en .venv
source /home/kevin/mitm_detection_project/.venv/bin/activate


python3 -m venv venv
source .venv/bin/activate 



# Instala dependecias que se necesitan
pip install --upgrade pip

pip install tensorflow

pip install pandas numpy scikit-learn matplotlib seaborn jupyter

pip install xgboost

pip install netifaces

# Para poder entrenar EL MODELO priemro hay que entrar
source .venv/bin/activate 


# 1. Generar más datos (repetir varias veces)
python3 scripts/capture_traffic.py --mode normal --duration 300 --interface wlan0
sudo python3 scripts/simple_attack_simulator.py --attack arp --duration 180

# 2. Procesar datos NO SE debe estar n el .venv
python3 scripts/extract_features.py

# Ejecutar calibración completa "ESTO NOS AYUDA A CALIBRAR"
python3 scripts/scripts/calibrate_threshold_fixed.py

# 3. Entrenar BÁSICO (rápido) SI SE necesita estar en el .venv
python3 scripts/train_model.py

# 4. Ver si mejoraron las métricas
# Si accuracy > 85% y tienes 50+ muestras → ir al paso 5
# Si no → volver al paso 1

# 5. SOLO AL FINAL: Optimizar
python3 scripts/optimize_model.py



---------------------------------------------------------------------
# USAR DETECTOR EN TIEMPO REAL

# Hacer ejecutable
chmod +x scripts/real_time_detector_advanced.py

# Ejecutar con configuración óptima
sudo python3 scripts/real_time_detector_advanced.py -i wlan0

# O con umbral personalizado
sudo python3 scripts/real_time_detector_advanced.py -i wlan0 -t 0.4

# Con filtro específico
sudo python3 scripts/real_time_detector_advanced.py -i wlan0 -f "tcp or arp"


-----------------------------------------
#  COMANDOS PARA USAR CON TU MIKROTIK
# 1. MONITOREAR VÍCTIMAS CONECTADAS
sudo python3 scripts/monitor_mikrotik_victims.py -n 10.42.0.0/24 -i eth0

# 2. CAPTURAR TRÁFICO NORMAL (víctimas navegando)
sudo python3 scripts/capture_from_mikrotik.py -i eth0 -t normal -d 300

# 3. CAPTURAR TRÁFICO DE ATAQUES (cuando haces MITM)
sudo python3 scripts/capture_from_mikrotik.py -i eth0 -t attack -d 180

# 4. CAPTURAR TODO EL TRÁFICO
sudo python3 scripts/capture_from_mikrotik.py -i eth0 -t mixed -d 600

# 5. CAPTURAR SOLO TRÁFICO WEB
sudo python3 scripts/capture_from_mikrotik.py -i eth0 -t web -d 240




-----------------------------
# DOCKER
# Probar construcción
docker-compose build mitm-detector

# Probar ejecución básica
docker-compose run --rm mitm-detector bash


# 1. DETECTOR EN TIEMPO REAL
docker-compose up mitm-detector

# 2. ENTRENAR MODELOS
docker-compose --profile training up mitm-trainer

# 3. COMPARAR MODELOS
docker-compose --profile analysis up mitm-comparator

# 4. CAPTURAR TRÁFICO
docker-compose --profile capture up mitm-capturer

# 5. SIMULAR ATAQUES
docker-compose --profile attack up mitm-attacker

# 6. SHELL INTERACTIVO
docker-compose run --rm mitm-detector bash

# 7. COMANDO PERSONALIZADO
docker-compose run --rm mitm-detector python3 scripts/extract_features.py