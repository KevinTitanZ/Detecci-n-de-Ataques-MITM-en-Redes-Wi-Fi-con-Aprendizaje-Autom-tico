
# Para Poder entrar al proyecto
cd ~/mitm_detection_project  

# Para capturar el trafico de red:
python3 scripts/capture_traffic.py --mode normal --duration 300 --interface wlan0

# Para capturar el trafico de red Malo:
sudo python3 scripts/simple_attack_simulator.py --attack arp --duration 120
sudo python3 scripts/simple_attack_simulator.py --attack dns --duration 120
sudo python3 scripts/simple_attack_simulator.py --attack portscan --duration 120

# sirve para activiar el lugar donde estan las descargar en .venv
source /home/kevin/mitm_detection_project/.venv/bin/activate

source .venv/bin/activate 

# Instala dependecias que se necesitan
pip install --upgrade pip

pip install tensorflow

pip install pandas numpy scikit-learn matplotlib seaborn jupyter


# Para poder entrenar EL MODELO priemro hay que entrar
source .venv/bin/activate 


# 1. Generar más datos (repetir varias veces)
python3 scripts/capture_traffic.py --mode normal --duration 300 --interface wlan0
sudo python3 scripts/simple_attack_simulator.py --attack arp --duration 180

# 2. Procesar datos NO SE debe estar n el .venv
python3 scripts/extract_features.py

# Ejecutar calibración completa "ESTO NOS AYUDA A CALIBRAR"
python3 scripts/calibrate_threshold.py

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