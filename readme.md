# 💳 Modelo de Riesgo Crediticio — Proyecto Integrador (Módulo 5)

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=julietazuccaro_PI-riesgo-crediticio-M5&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=julietazuccaro_PI-riesgo-crediticio-M5)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=julietazuccaro_PI-riesgo-crediticio-M5&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=julietazuccaro_PI-riesgo-crediticio-M5)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=julietazuccaro_PI-riesgo-crediticio-M5&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=julietazuccaro_PI-riesgo-crediticio-M5)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=julietazuccaro_PI-riesgo-crediticio-M5&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=julietazuccaro_PI-riesgo-crediticio-M5)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=julietazuccaro_PI-riesgo-crediticio-M5&metric=coverage)](https://sonarcloud.io/summary/new_code?id=julietazuccaro_PI-riesgo-crediticio-M5)
[![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=julietazuccaro_PI-riesgo-crediticio-M5&metric=duplicated_lines_density)](https://sonarcloud.io/summary/new_code?id=julietazuccaro_PI-riesgo-crediticio-M5)

Pipeline de MLOps para predecir el **comportamiento de pago** de los clientes de una
financiera: dado un solicitante de crédito, estimar si **pagará a tiempo** o no.
El proyecto cubre el ciclo completo: análisis de datos, ingeniería de características,
entrenamiento y selección de modelos, **monitoreo de data drift** y **despliegue del
modelo como API REST dockerizada**.

---

## 🎯 Caso de negocio

Una financiera necesita decidir a quién otorgar crédito. Cada préstamo que **no se paga
a tiempo** genera una pérdida, mientras que rechazar a un buen pagador implica un costo de
oportunidad. El objetivo es un modelo que **detecte con anticipación a los clientes con
riesgo de mora** (`Pago_atiempo = 0`), para apoyar la decisión de otorgamiento.

- **Variable objetivo:** `Pago_atiempo` (1 = pagó a tiempo, 0 = moroso).
- **Dataset:** `Base_de_datos.csv` — 10.763 préstamos, 23 variables (montos, plazos,
  puntajes de buró, datos laborales, saldos, etc.).
- **Desafío central:** el dataset está **fuertemente desbalanceado (~95% paga a tiempo)**,
  por lo que la *accuracy* engaña y el foco debe estar en detectar la clase minoritaria.

---

## 🗂️ Estructura del repositorio

```
PI-riesgo-crediticio-M5/
├── Base_de_datos.csv               # Dataset de ejemplo
├── requirements.txt                # Dependencias de desarrollo
├── requirements-api.txt            # Dependencias de la API (las que instala Docker)
├── Dockerfile                      # Imagen del servicio de predicción
├── .dockerignore
├── pyproject.toml                  # Configuración de pytest y coverage
├── sonar-project.properties        # Configuración del análisis de SonarCloud
├── .github/workflows/sonarcloud.yml # CI: tests con cobertura + análisis de calidad
├── readme.md
└── mlops_pipeline/
    └── src/
        ├── Cargar_datos.py                 # Ingesta de datos
        ├── comprension_eda.ipynb           # EDA + análisis del modelado
        ├── ft_engineering.py               # Pipeline de datos (limpieza + preprocesamiento)
        ├── model_training_evaluation.py    # Entrenamiento, selección y serialización
        ├── model_monitoring.py             # Métricas de data drift
        ├── app_monitoreo.py                # App de Streamlit (monitoreo)
        ├── model_deploy.py                 # API REST con FastAPI
        ├── test_model_deploy.py            # Tests automatizados de la API
        └── modelo_riesgo.joblib            # Modelo entrenado (artefacto de despliegue)
```

---

## 🔄 Proceso (pipeline de ML)

| Etapa | Archivo | Qué hace |
|---|---|---|
| **1. Ingesta** | `Cargar_datos.py` | Carga el dataset (simula la lectura del Data Warehouse). |
| **2. EDA** | `comprension_eda.ipynb` | Análisis univariable, bivariable y multivariable. |
| **3. Ingeniería de características** | `ft_engineering.py` | Limpieza, imputación, escalado, codificación y features nuevos. |
| **4. Modelado** | `model_training_evaluation.py` | Compara 4 modelos con validación cruzada, elige el mejor y lo serializa. |
| **5. Monitoreo** | `model_monitoring.py` + `app_monitoreo.py` | Mide data drift en el tiempo y genera alertas. |
| **6. Despliegue** | `model_deploy.py` + `Dockerfile` | Expone el modelo como API REST y la empaqueta en un contenedor. |

### Ingeniería de características
- Limpieza de `tendencia_ingresos` (valores inválidos → `NaN`).
- Imputación (mediana / moda), escalado (`StandardScaler`) y codificación (`OneHotEncoder`)
  vía `ColumnTransformer`, ajustado **solo con el set de entrenamiento** (sin data leakage).
- Features nuevos con sentido de negocio: `ratio_cuota_salario` y `ratio_deuda_ingreso`.
- Partición train/test **estratificada** por el desbalance.
- Las transformaciones que no aprenden de los datos están aisladas en `preparar_features()`,
  que usan **tanto el entrenamiento como la API**. Así se evita el *training/serving skew*:
  que en producción el modelo reciba features calculadas de otra forma.

### Modelado
- Modelos comparados: Regresión Logística, Random Forest, Gradient Boosting y XGBoost.
- Selección por **AUC** y **recall de la clase morosa**, *no* por accuracy (por el desbalance).
- `class_weight="balanced"` para compensar la clase minoritaria.
- El ganador se serializa en `modelo_riesgo.joblib` junto con su metadata de trazabilidad
  (versión, fecha de entrenamiento, métricas, columnas y versiones del entorno).

### Monitoreo de data drift
- Se usa `fecha_prestamo` para dividir la población en una **ventana de referencia**
  (primeros ~6 meses, con la que se entrena el modelo) y **ventanas mensuales** posteriores.
- Métricas por variable:

  | Métrica | Tipo de variable | Qué detecta |
  |---|---|---|
  | **PSI** (Population Stability Index) | numérica y categórica | Magnitud del cambio de distribución |
  | **KS** (Kolmogorov-Smirnov) | numérica | Diferencia entre distribuciones acumuladas |
  | **Jensen-Shannon** | numérica y categórica | Divergencia entre distribuciones |
  | **Chi-cuadrado** | categórica | Cambio en las proporciones de categorías |

- Umbrales de PSI: 🟢 < 0.10 · 🟡 0.10–0.25 · 🔴 > 0.25.

---

## 🌐 API de predicción

`model_deploy.py` levanta un servicio **FastAPI** que carga el modelo serializado una sola
vez al arrancar y lo expone por HTTP. La API recibe los datos **crudos** del solicitante:
todo el preprocesamiento viaja dentro del pipeline serializado.

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Información del servicio y sus rutas. |
| `GET` | `/health` | Estado del servicio (200 = ok · 503 = degradado, sin modelo). |
| `GET` | `/modelo` | Metadata del modelo en producción: versión, fecha, métricas y campos requeridos. |
| `POST` | `/predict` | Predicción individual (JSON). |
| `POST` | `/predict/batch` | **Predicción por lotes**: N solicitudes en una sola llamada. |
| `POST` | `/predict/csv` | Predicción por lotes subiendo un archivo CSV (respuesta en JSON o CSV). |
| `GET` | `/docs` | **Documentación interactiva (Swagger UI)**, generada automáticamente. |

Todos los endpoints de predicción aceptan el parámetro `?umbral=` (0.01–0.99, por defecto
`0.5`): la probabilidad de mora a partir de la cual se rechaza la solicitud. Es una perilla
de **negocio**, no del modelo — bajarla detecta más morosos pero rechaza más buenos clientes.

### Respuesta

Además de la probabilidad, la API devuelve una lectura accionable para el analista de riesgo:

| Condición | `nivel_riesgo` | `decision_sugerida` |
|---|---|---|
| `prob_mora ≥ umbral` | `ALTO` | `RECHAZAR` |
| `70% del umbral ≤ prob_mora < umbral` | `MEDIO` | `REVISAR` (derivación manual) |
| `prob_mora < 70% del umbral` | `BAJO` | `APROBAR` |

```jsonc
{
  "id_solicitud": "8f3c...",        // trazabilidad de cada predicción
  "probabilidad_mora": 0.412,
  "probabilidad_pago": 0.588,
  "prediccion": 1,                  // 1 = paga a tiempo · 0 = riesgo de mora
  "etiqueta": "Paga a tiempo",
  "nivel_riesgo": "MEDIO",
  "decision_sugerida": "REVISAR",
  "umbral": 0.5,
  "version_modelo": "1.3.0",
  "timestamp": "2026-07-27T13:48:02+00:00"
}
```

### Levantar la API localmente

```bash
cd mlops_pipeline/src
uvicorn model_deploy:app --reload --port 8000
# -> http://localhost:8000/docs
```

> ⚠️ La API necesita `modelo_riesgo.joblib`. Si no existe, levanta en modo **degradado**
> (`/health` responde 503). Se genera con `python model_training_evaluation.py`.

### Ejemplos de uso

```bash
# Predicción individual
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"tipo_credito":7,"capital_prestado":2500000,"plazo_meses":12,"edad_cliente":44,
       "tipo_laboral":"Empleado","salario_cliente":3200000,"total_otros_prestamos":5800000,
       "cuota_pactada":245000,"puntaje_datacredito":780,"cant_creditosvigentes":5,
       "huella_consulta":4,"saldo_mora":0,"saldo_total":45000,"saldo_principal":40000,
       "saldo_mora_codeudor":0,"creditos_sectorFinanciero":3,"creditos_sectorCooperativo":0,
       "creditos_sectorReal":1,"promedio_ingresos_datacredito":2000000,
       "tendencia_ingresos":"Estable"}'

# Predicción por lotes desde un CSV, con umbral más estricto
curl -X POST "http://localhost:8000/predict/csv?umbral=0.4&formato=csv" \
  -F "archivo=@solicitudes.csv" -o predicciones.csv
```

### Validación de entradas

- **`/predict` y `/predict/batch`** (consumo online, desde formularios): validación estricta
  con **Pydantic**. Tipos, rangos y categorías permitidas; ante un error responde `422` con
  el detalle del campo, sin llegar a tocar el modelo.
- **`/predict/csv`** (consumo batch, desde el ETL interno): se valida el **esquema**
  (que estén todas las columnas requeridas, con un `400` que las nombra), pero no los rangos.
  Forzarlos haría fallar el lote entero por unos pocos atípicos del histórico — por ejemplo,
  hay 150 registros con edad > 100. Esos valores ya los absorbe la imputación del pipeline.

---

## 🐳 Contenedor Docker

La imagen empaqueta el código, el modelo, las dependencias y el servidor Uvicorn.

```bash
# Construir (desde la raíz del repositorio)
docker build -t riesgo-crediticio-api:1.3.0 .

# Ejecutar
docker run -p 8000:8000 riesgo-crediticio-api:1.3.0
# -> http://localhost:8000/docs

# Verificar el estado
curl http://localhost:8000/health
```

Decisiones de la imagen:

| Decisión | Por qué |
|---|---|
| Base `python:3.12-slim` | Imagen mínima; la versión de Python se fija porque el modelo se deserializa con `pickle`. |
| `requirements-api.txt` en vez de `requirements.txt` | La imagen no necesita Jupyter, Streamlit, XGBoost ni matplotlib: menos peso, build más rápido y menos superficie de ataque. |
| Versiones **clavadas** | `pickle` guarda la estructura interna de los objetos de scikit-learn: si la versión que deserializa no coincide, la carga falla o devuelve resultados distintos. |
| Dependencias antes que el código | Aprovecha la caché de capas: cambiar un `.py` no reinstala todo el stack. |
| Usuario `apiuser` (no root) | Limita el impacto de una eventual vulnerabilidad en el servicio. |
| `HEALTHCHECK` sobre `/health` | Docker marca el contenedor como *unhealthy* si el modelo no cargó. |
| `--host 0.0.0.0` | Sin esto Uvicorn sólo escucharía dentro del contenedor y `-p` no serviría. |

---

## 🧪 Tests

`test_model_deploy.py` cubre los endpoints con el `TestClient` de FastAPI (levanta la app en
memoria: no hace falta el servidor ni Docker).

```bash
# Desde la raíz del repositorio
pytest -v                                  # 30 tests
pytest --cov --cov-report=term             # con reporte de cobertura
```

Qué se verifica: estructura de las respuestas, rechazo de entradas inválidas (campos
faltantes, rangos, categorías desconocidas), coherencia del resumen de los lotes, las bandas
de decisión de negocio, y que **un mismo registro dé el mismo resultado** por `/predict`,
`/predict/batch` y `/predict/csv`.

---

## 🛡️ Calidad de código (SonarCloud)

Cada push a `main`, `developer` o `certification` dispara el workflow
`.github/workflows/sonarcloud.yml`, que ejecuta los tests con cobertura y envía el resultado
a **SonarQube Cloud**, donde se evalúa mantenibilidad, seguridad, cobertura y estilo.

### Puesta en marcha (una sola vez)

1. Entrar a [sonarcloud.io](https://sonarcloud.io) con la cuenta de GitHub e importar el
   repositorio.
2. Copiar el **Project Key** y la **Organization** que genera SonarCloud y verificar que
   coincidan con los de `sonar-project.properties`.
3. En **Administration → Analysis Method**, **desactivar `Automatic Analysis`**. Es
   obligatorio: si queda activo, choca con el análisis por CI y el workflow falla.
4. Generar un token en **My Account → Security** y cargarlo en el repositorio de GitHub como
   secreto `SONAR_TOKEN` (*Settings → Secrets and variables → Actions*).

### Resultados del análisis

| Dimensión | Resultado |
|---|---|
| **Mantenibilidad** (calidad del código) | **A** |
| **Fiabilidad** | **A** |
| **Seguridad** | **B** |
| **Cobertura de pruebas** | **43,5%** |
| **Líneas duplicadas** | **0%** |
| Líneas de código analizadas | 872 |

### Cobertura por módulo

| Módulo | Cobertura | Comentario |
|---|---|---|
| `model_deploy.py` | **86%** | La API, cubierta por los 30 tests. |
| `ft_engineering.py` | 65% | Cubierto por lo que la API ejercita. |
| `Cargar_datos.py` | 33% | Sólo el bloque `__main__` queda sin cubrir. |
| `model_training_evaluation.py` · `model_monitoring.py` | 0% | Scripts batch, sin tests propios todavía. |
| `app_monitoreo.py` | — | Excluido: script de Streamlit que se ejecuta al importarse; se valida con `streamlit.testing.v1.AppTest`, que no genera reporte de cobertura. |

La cobertura está concentrada en el componente que efectivamente se despliega. Los scripts de
entrenamiento y monitoreo son el próximo objetivo natural para subir ese número.

### Hallazgo de seguridad corregido

El primer análisis detectó una vulnerabilidad de **log injection** (CWE-117) en
`model_deploy.py`: el endpoint `/predict/csv` escribía en el log el nombre del archivo subido
sin sanear. Un archivo llamado `x.csv\n2026-07-27 | INFO | Modelo cargado` habría insertado una
entrada falsa en el registro, útil para falsear la auditoría o tapar el rastro de un ataque.

Se corrigió con `_sanitizar_para_log()`, que elimina los saltos de línea y acota el largo antes
de loguear, y se agregaron tests que cubren el caso.

---

## 🔎 Principales hallazgos

1. **Desbalance fuerte (~95% / 5%).** La accuracy no sirve como criterio: un modelo trivial
   que prediga "todos pagan" alcanza 95% sin detectar ningún moroso.
2. **`puntaje` era data leakage.** Correlacionaba 0.92 con el objetivo. Al excluirlo, el poder
   predictivo cae (AUC ~0.67), lo que confirma que **concentraba casi toda la señal** y que no
   estaría disponible en un escenario real de decisión. Se descartó del modelo.
3. **Modelo elegido: Regresión Logística balanceada.** Aunque baja la accuracy (65%), es el
   único que aporta valor al negocio: **mejor AUC (0.672)** y detecta el **~60% de los morosos**,
   frente al 2–5% de los modelos de árbol (que tienen 95% de accuracy pero no sirven para el objetivo).
4. **El data drift crece con el tiempo.** El PSI promedio pasa de **0.077** (mayo 2025) a **0.411**
   (marzo 2026): cuanto más lejos de la ventana de entrenamiento, más se aleja la población reciente.
   Las variables monetarias (`saldo_total`, `salario_cliente`, `capital_prestado`) son las que más
   derivan → **se recomienda reentrenar el modelo periódicamente**.

---

## 🚀 Cómo ejecutar

```bash
# 1. Crear y activar el entorno virtual
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el pipeline (desde mlops_pipeline/src)
cd mlops_pipeline/src
python ft_engineering.py                 # prepara los datos
python model_training_evaluation.py      # entrena, compara y serializa el mejor modelo
python model_monitoring.py               # reporte de drift por consola

# 4. App de monitoreo (Streamlit)
streamlit run app_monitoreo.py           # http://localhost:8501

# 5. API de predicción (FastAPI)
uvicorn model_deploy:app --reload        # http://localhost:8000/docs

# 6. Tests
pytest -v
```

---

## 🏷️ Versionado

| Tag | Contenido |
|---|---|
| `v1.0.0` | Estructura inicial del proyecto |
| `v1.0.1` | Análisis exploratorio de datos (EDA) |
| `v1.1.0` | Ingeniería de características y modelado supervisado |
| `v1.2.0` | Monitoreo de data drift + app de Streamlit |
| `v1.3.0` | Despliegue: API REST con FastAPI + imagen Docker + tests |

Flujo de trabajo con ramas `developer` → `main` mediante Pull Requests.

---

*Proyecto Integrador — Módulo 5 · Carrera de Data Science.*
