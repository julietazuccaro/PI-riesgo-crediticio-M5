# 💳 Modelo de Riesgo Crediticio — Proyecto Integrador (Módulo 5)

Pipeline de MLOps para predecir el **comportamiento de pago** de los clientes de una
financiera: dado un solicitante de crédito, estimar si **pagará a tiempo** o no.
El proyecto cubre el ciclo completo: análisis de datos, ingeniería de características,
entrenamiento y selección de modelos, y **monitoreo de data drift** en producción.

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
├── requirements.txt                # Dependencias
├── readme.md
└── mlops_pipeline/
    └── src/
        ├── Cargar_datos.py                 # Ingesta de datos
        ├── comprension_eda.ipynb           # EDA + análisis del modelado
        ├── ft_engineering.py               # Pipeline de datos (limpieza + preprocesamiento)
        ├── model_training_evaluation.py    # Entrenamiento y selección de modelos
        ├── model_monitoring.py             # Métricas de data drift
        ├── app_monitoreo.py                # App de Streamlit (monitoreo)
        └── model_deploy.py                 # Despliegue (avance siguiente)
```

---

## 🔄 Proceso (pipeline de ML)

| Etapa | Archivo | Qué hace |
|---|---|---|
| **1. Ingesta** | `Cargar_datos.py` | Carga el dataset (simula la lectura del Data Warehouse). |
| **2. EDA** | `comprension_eda.ipynb` | Análisis univariable, bivariable y multivariable. |
| **3. Ingeniería de características** | `ft_engineering.py` | Limpieza, imputación, escalado, codificación y features nuevos. |
| **4. Modelado** | `model_training_evaluation.py` | Compara 4 modelos con validación cruzada y elige el mejor. |
| **5. Monitoreo** | `model_monitoring.py` + `app_monitoreo.py` | Mide data drift en el tiempo y genera alertas. |

### Ingeniería de características
- Limpieza de `tendencia_ingresos` (valores inválidos → `NaN`).
- Imputación (mediana / moda), escalado (`StandardScaler`) y codificación (`OneHotEncoder`)
  vía `ColumnTransformer`, ajustado **solo con el set de entrenamiento** (sin data leakage).
- Features nuevos con sentido de negocio: `ratio_cuota_salario` y `ratio_deuda_ingreso`.
- Partición train/test **estratificada** por el desbalance.

### Modelado
- Modelos comparados: Regresión Logística, Random Forest, Gradient Boosting y XGBoost.
- Selección por **AUC** y **recall de la clase morosa**, *no* por accuracy (por el desbalance).
- `class_weight="balanced"` para compensar la clase minoritaria.

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
# 1. Crear y activar el entorno virtual (Python 3.11)
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar los scripts (desde mlops_pipeline/src)
cd mlops_pipeline/src
python ft_engineering.py                 # prepara los datos
python model_training_evaluation.py      # entrena y compara modelos
python model_monitoring.py               # reporte de drift por consola

# 4. Levantar la app de monitoreo
streamlit run app_monitoreo.py           # abre http://localhost:8501
```

---

## 🏷️ Versionado

| Tag | Contenido |
|---|---|
| `v1.0.0` | Estructura inicial del proyecto |
| `v1.0.1` | Análisis exploratorio de datos (EDA) |
| `v1.1.0` | Ingeniería de características y modelado supervisado |
| `v1.2.0` | Monitoreo de data drift + app de Streamlit |

Flujo de trabajo con ramas `developer` → `main` mediante Pull Requests.

---

*Proyecto Integrador — Módulo 5 · Carrera de Data Science.*
