# =============================================================================
# Imagen de la API de Riesgo Crediticio
# -----------------------------------------------------------------------------
# Empaqueta el código, el modelo entrenado, las dependencias y el servidor de
# aplicación (Uvicorn) en una unidad que corre igual en cualquier máquina.
# Ese es el punto de Docker: se acabó el "en mi compu funciona".
#
# Construir (desde la raíz del repositorio):
#     docker build -t riesgo-crediticio-api:1.3.0 .
#
# Ejecutar:
#     docker run -p 8000:8000 riesgo-crediticio-api:1.3.0
#     -> http://localhost:8000/docs
# =============================================================================

# Imagen base "slim": Python 3.12 sobre Debian mínimo. Se fija la versión de
# Python porque el modelo fue serializado con 3.12 y las versiones del stack
# científico tienen que coincidir para deserializarlo sin advertencias.
FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE: no generar archivos .pyc dentro del contenedor.
# PYTHONUNBUFFERED: que los logs salgan al instante y no queden en el buffer
#   (si no, `docker logs` no muestra nada hasta que el buffer se llena).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- 1. Dependencias --------------------------------------------------------
# Se copian ANTES que el código a propósito: Docker cachea cada instrucción, y
# como las dependencias cambian mucho menos seguido que el código, así un cambio
# en un .py no obliga a reinstalar todo de nuevo.
#
# Se usa `requirements-api.txt` (sólo lo que la API necesita en ejecución) en
# lugar del `requirements.txt` de desarrollo: este último trae Jupyter, XGBoost,
# Streamlit y matplotlib, que multiplican el tamaño de la imagen sin que el
# servicio los use nunca. Menos paquetes = imagen más chica, build más rápido y
# menos superficie de ataque.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# --- 2. Usuario sin privilegios ---------------------------------------------
# Por defecto los contenedores corren como root. Si alguien encontrara una
# vulnerabilidad en la API, tendría permisos de root dentro del contenedor.
RUN useradd --create-home --uid 1000 apiuser

# --- 3. Código y modelo -----------------------------------------------------
# Sólo se copia lo que la API necesita en ejecución. Los notebooks, el dataset,
# los scripts de entrenamiento y los tests no van a la imagen de producción.
COPY --chown=apiuser:apiuser mlops_pipeline/src/model_deploy.py .
COPY --chown=apiuser:apiuser mlops_pipeline/src/ft_engineering.py .
COPY --chown=apiuser:apiuser mlops_pipeline/src/Cargar_datos.py .
COPY --chown=apiuser:apiuser mlops_pipeline/src/modelo_riesgo.joblib .

USER apiuser

# --- 4. Ejecución -----------------------------------------------------------
# Documenta que el servicio escucha en el 8000 (no publica el puerto por sí solo:
# eso lo hace el -p del `docker run`).
EXPOSE 8000

# Chequeo de salud: Docker consulta /health periódicamente y marca el contenedor
# como "unhealthy" si el modelo no está cargado. `start-period` da margen para
# que el modelo termine de cargarse antes del primer chequeo.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# --host 0.0.0.0 es obligatorio: con el 127.0.0.1 por defecto, Uvicorn sólo
# aceptaría conexiones desde adentro del contenedor y el -p no serviría de nada.
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
