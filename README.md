# 🛡️ MITM Detection Project

Este proyecto permite **capturar tráfico de red**, **simular ataques MITM**, **extraer características**, **entrenar modelos de Machine Learning** y **detectar ataques en tiempo real**, tanto de forma local como usando **Docker** y **MikroTik**.

El README está pensado para que **cualquier persona pueda clonar el proyecto y ejecutarlo desde cero en otra computadora**, sin errores por el entorno virtual (venv).

---

## 📁 1. Entrar al proyecto

```bash
cd ~/mitm_detection_project
```

---

## 🧪 2. Crear y usar el entorno virtual (VENV)

> ⚠️ El entorno virtual **NO se versiona** en GitHub. Cada computadora debe crearlo nuevamente.

### 🔹 Crear el entorno virtual

```bash
python3 -m venv .venv
```

### 🔹 Activar el entorno virtual

```bash
source .venv/bin/activate
```

> Cuando el venv está activo verás `(.venv)` al inicio de la terminal.

---

## 🧹 3. Eliminar el VENV (si hay errores o cambias de PC)

Si el entorno virtual falla o copias el proyecto a otra computadora:

```bash
rm -rf .venv
```

Luego vuelve a crearlo siguiendo el **paso 2**.

---

## 📦 4. Instalar dependencias (CON el VENV activo)

```bash
pip install --upgrade pip

pip install tensorflow
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
pip install xgboost
pip install netifaces
pip install streamlit plotly
pip install scapy
```

---

## 🌐 5. Captura de tráfico de red

### 🔹 Tráfico normal

```bash
python3 scripts/capture_traffic.py --mode normal --duration 300 --interface wlan0
```

### 🔹 Tráfico malicioso (ataques simples)

```bash
sudo python3 scripts/simple_attack_simulator.py --attack arp --duration 120
sudo python3 scripts/simple_attack_simulator.py --attack dns --duration 120
sudo python3 scripts/simple_attack_simulator.py --attack portscan --duration 120
```

---

## ⚔️ 6. Simulador universal de ataques

### Hacer el script ejecutable

```bash
chmod +x scripts/universal_attack_simulator.py
```

### Ejecutar ataques

```bash
# ARP
sudo python3 scripts/universal_attack_simulator.py --attack arp --duration 120

# DNS anómalo
sudo python3 scripts/universal_attack_simulator.py --attack dns --duration 120

# Escaneo de puertos
sudo python3 scripts/universal_attack_simulator.py --attack portscan --duration 120

# Inyección de tráfico
sudo python3 scripts/universal_attack_simulator.py --attack injection --duration 120

# Todos los ataques
sudo python3 scripts/universal_attack_simulator.py --attack all --duration 300
```

---

## 🧠 7. Entrenamiento del modelo

### 🔹 Paso 1: Generar datos (CON VENV activo)

```bash
python3 scripts/capture_traffic.py --mode normal --duration 300 --interface wlan0
sudo python3 scripts/simple_attack_simulator.py --attack arp --duration 180
```

### 🔹 Paso 2: Extraer características (SIN VENV)

```bash
python3 scripts/extract_features.py
```

### 🔹 Paso 3: Entrenar modelo básico (CON VENV)

```bash
python3 scripts/train_model.py
```

### 🔹 Paso 4: Calibrar umbral

```bash
python3 scripts/calibrate_threshold_fixed.py
```

📊 Si la **accuracy > 85%** y hay **50+ muestras**, continuar.

### 🔹 Paso 5: Optimizar modelo (solo al final)

```bash
python3 scripts/optimize_model.py
```

---

## 🚀 8. Detector en tiempo real

### Dashboard

```bash
streamlit run scripts/dashboard.py
```

### Detector avanzado

```bash
chmod +x scripts/real_time_detector_advanced.py

sudo python3 scripts/real_time_detector_advanced.py -i wlan0

# Umbral personalizado
sudo python3 scripts/real_time_detector_advanced.py -i wlan0 -t 0.4

# Filtro específico
sudo python3 scripts/real_time_detector_advanced.py -i wlan0 -f "tcp or arp"
```

### Detector simple

```bash
sudo python3 scripts/real_time_detector.py --interface eth0 --threshold 0.25
```

---

## 💣 9. Ataque manual (pruebas)

```bash
sudo arpspoof -i eth0 -t 10.0.2.15 10.0.2.2
```

---

## 📡 10. Integración con MikroTik

```bash
# Monitorear víctimas
sudo python3 scripts/monitor_mikrotik_victims.py -n 10.42.0.0/24 -i eth0

# Tráfico normal
sudo python3 scripts/capture_from_mikrotik.py -i eth0 -t normal -d 300

# Tráfico de ataque
sudo python3 scripts/capture_from_mikrotik.py -i eth0 -t attack -d 180

# Todo el tráfico
sudo python3 scripts/capture_from_mikrotik.py -i eth0 -t mixed -d 600

# Solo tráfico web
sudo python3 scripts/capture_from_mikrotik.py -i eth0 -t web -d 240
```

---

## 🐳 11. Uso con Docker

### Construir imágenes

```bash
docker-compose build mitm-detector
```

### Ejecutar servicios

```bash
# Detector en tiempo real
docker-compose up mitm-detector

# Entrenamiento
docker-compose --profile training up mitm-trainer

# Comparación de modelos
docker-compose --profile analysis up mitm-comparator

# Captura de tráfico
docker-compose --profile capture up mitm-capturer

# Simular ataques
docker-compose --profile attack up mitm-attacker
```

### Shell interactivo

```bash
docker-compose run --rm mitm-detector bash
```

### Comando personalizado

```bash
docker-compose run --rm mitm-detector python3 scripts/extract_features.py
```

---

## ✅ Notas finales

* Usa **sudo** cuando se capture tráfico o se simulen ataques
* El **VENV siempre se crea localmente**
* Docker es recomendado para entornos controlados
* Este proyecto está orientado a **detección MITM en tiempo real con ML**

---

📌 *Proyecto académico – Seguridad de Redes / Machine Learning*
