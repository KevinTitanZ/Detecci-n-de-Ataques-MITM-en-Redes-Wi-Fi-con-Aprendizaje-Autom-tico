FROM python:3.11-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TF_CPP_MIN_LOG_LEVEL=2

# Metadatos
LABEL maintainer="Kevin Ordoñez <kevin@espe.edu.ec>"
LABEL description="Detector MITM en tiempo real con CNN+Bi-LSTM"
LABEL version="1.0"

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    # Herramientas de red
    tcpdump \
    wireless-tools \
    net-tools \
    iproute2 \
    aircrack-ng \
    wireshark-common \
    # Herramientas de desarrollo
    gcc \
    g++ \
    make \
    # Librerías necesarias
    libpcap-dev \
    libffi-dev \
    libssl-dev \
    # Utilidades
    curl \
    wget \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Crear usuario para el detector
RUN useradd -m -s /bin/bash -G sudo detector && \
    echo "detector ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Crear estructura de directorios
RUN mkdir -p \
    data/raw \
    data/processed \
    models \
    results \
    alerts \
    logs \
    scripts \
    config

# Copiar código fuente
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY data/ ./data/
COPY results/ ./results/

# Copiar script de entrada
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Permisos para scripts
RUN chmod +x scripts/*.py && \
    chown -R detector:detector /app

# Cambiar a usuario detector para operaciones normales
USER detector

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import tensorflow as tf; import scapy; print('✅ Servicios OK')" || exit 1

# Puerto para API futura
EXPOSE 8080

# Punto de entrada
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Comando por defecto
CMD ["detector"]