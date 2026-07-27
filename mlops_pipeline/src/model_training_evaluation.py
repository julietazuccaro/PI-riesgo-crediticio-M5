"""
Entrenamiento y evaluación de modelos de riesgo crediticio.

Compara varios modelos supervisados, los evalúa con validación cruzada y
selecciona el de mejor performance. Genera una tabla resumen, un gráfico
comparativo y **serializa el mejor modelo** en `modelo_riesgo.joblib`, que es
el artefacto que después levanta la API (`model_deploy.py`).

Ejecutar desde `mlops_pipeline/src`:
    python model_training_evaluation.py
"""

import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from ft_engineering import preparar_datos

# Versión del modelo que se despliega (se guarda dentro del artefacto y la
# expone la API en /modelo, para poder trazar qué versión hizo cada predicción).
VERSION_MODELO = "1.3.0"

# El artefacto se guarda junto al código, en mlops_pipeline/src/. Se usa una ruta
# relativa al archivo (no al directorio de trabajo) para que funcione igual al
# ejecutarlo localmente o dentro del contenedor Docker.
RUTA_ARTEFACTO = Path(__file__).resolve().parent / "modelo_riesgo.joblib"


def build_models() -> list:
    """Devuelve la lista de modelos candidatos a comparar (nombre, instancia)."""
    return [
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("RandomForest", RandomForestClassifier(random_state=42, class_weight="balanced")),
        ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
        ("XGBoost", XGBClassifier(random_state=42, eval_metric="logloss"))
    ]

def summarize_classification(y_true, y_pred, y_proba) -> dict:
    """Calcula las métricas de clasificación y las devuelve en un diccionario.

    Dado el fuerte desbalance (~95% paga a tiempo), las métricas sobre la clase
    mayoritaria (1 = "paga") resultan engañosas: un modelo que prediga "todos pagan"
    obtiene recall/f1 ~0.97 sin detectar ningún moroso. Por ese motivo se calculan
    precision/recall/f1 sobre la clase morosa (0), relevante en riesgo crediticio,
    y se agregan métricas robustas al desbalance (f1_macro, auc).

    Parámetros
    ----------
    y_true : valores reales
    y_pred : predicciones (0/1)
    y_proba : probabilidad de la clase 1 (necesaria para el AUC)

    Retorna
    -------
    dict con accuracy, métricas de la clase morosa, f1_macro y auc
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        # --- Clase MOROSA (0): lo que el negocio necesita detectar ---
        "precision_moroso": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_moroso": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_moroso": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        # --- Métricas robustas al desbalance (promedian ambas clases / independientes del umbral) ---
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
    }


def train_and_select_model(X_train, y_train, X_test, y_test, preprocesador):
    """Entrena y evalúa cada modelo candidato, y selecciona el mejor.

    Devuelve la tabla resumen (una fila por modelo) y el mejor modelo (por AUC en test).
    """
    resultados = []
    modelos_entrenados = {}

    for nombre, modelo in build_models():
        # 1. Pipeline: preprocesamiento + modelo (clone = preprocesador fresco por modelo)
        pipe = Pipeline([
            ("preprocesador", clone(preprocesador)),
            ("clf", modelo),
        ])

        # 2. Validación cruzada (AUC) en train -> mide estabilidad entre folds.
        #    Se usa AUC (no F1 de la clase mayoritaria) para no premiar al modelo "todos pagan".
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")

        # 3. Entrenar con todo el train y predecir en test
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        # 4. Métricas en test + info de la CV
        metricas = summarize_classification(y_test, y_pred, y_proba)
        metricas["modelo"] = nombre
        metricas["cv_auc_mean"] = cv_scores.mean()
        metricas["cv_auc_std"] = cv_scores.std()

        resultados.append(metricas)
        modelos_entrenados[nombre] = pipe

    # Tabla resumen ordenada por AUC en test (mejor arriba): mide la capacidad real
    # de separar morosos de buenos pagadores, sin verse afectada por el desbalance.
    resultados = pd.DataFrame(resultados).set_index("modelo").sort_values("auc", ascending=False)

    # El mejor = el de mayor AUC en test
    mejor_nombre = resultados.index[0]
    mejor_modelo = modelos_entrenados[mejor_nombre]

    return resultados, mejor_modelo


def guardar_modelo(pipeline, nombre: str, metricas: dict, columnas,
                   ruta: Path = RUTA_ARTEFACTO) -> Path:
    """Serializa el pipeline entrenado junto con su metadata (formato .joblib).

    Se guarda el **pipeline completo** (preprocesador + clasificador), no sólo el
    clasificador: así la API aplica exactamente las mismas imputaciones, escalados
    y codificaciones que se usaron al entrenar.

    Además del modelo se persiste metadata de trazabilidad, un principio básico de
    MLOps: sin ella no se puede saber qué versión respondió una predicción ni
    reproducir el entorno que la generó.

    Parámetros
    ----------
    pipeline : Pipeline de sklearn ya entrenado (preprocesador + clf).
    nombre : nombre del algoritmo ganador (p. ej. "LogisticRegression").
    metricas : métricas de evaluación en test del modelo elegido.
    columnas : columnas de entrada (en orden) con las que se entrenó.
    ruta : destino del archivo .joblib.

    Retorna
    -------
    Path del archivo generado.
    """
    artefacto = {
        "modelo": pipeline,
        "nombre": nombre,
        "version": VERSION_MODELO,
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        # Orden exacto de las columnas de entrada: la API reindexa con esta lista
        # para no depender del orden en que lleguen los campos del JSON.
        "columnas": list(columnas),
        # np.float64 no es serializable a JSON -> se castea a float nativo.
        "metricas": {k: float(v) for k, v in metricas.items()},
        # Versiones del entorno: si no coinciden al deserializar, sklearn avisa.
        "entorno": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    joblib.dump(artefacto, ruta)
    return ruta


if __name__ == "__main__":
    # 1. Preparar datos (reutiliza ft_engineering)
    X_train, X_test, y_train, y_test, preprocesador = preparar_datos()

    # 2. Entrenar y seleccionar el mejor modelo
    resultados, mejor_modelo = train_and_select_model(
        X_train, y_train, X_test, y_test, preprocesador
    )

    # 3. Tabla resumen
    print("=== Tabla resumen de modelos ===")
    print(resultados.round(3))
    print("\nMejor modelo:", mejor_modelo.named_steps["clf"].__class__.__name__)

    # 4. Serializar el modelo ganador: es el artefacto que consume la API.
    mejor_nombre = resultados.index[0]
    ruta = guardar_modelo(
        pipeline=mejor_modelo,
        nombre=mejor_nombre,
        metricas=resultados.loc[mejor_nombre].to_dict(),
        columnas=X_train.columns,
    )
    print(f"Modelo serializado en {ruta}")

    # 5. Gráfico comparativo (recall de morosos y AUC por modelo):
    #    las dos métricas que muestran si el modelo sirve para el negocio.
    resultados[["recall_moroso", "auc"]].plot(kind="bar", figsize=(8, 5))
    plt.title("Comparación de modelos: recall de morosos y AUC")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("comparacion_modelos.png", dpi=120)
    plt.show()
    print("Gráfico guardado en comparacion_modelos.png")