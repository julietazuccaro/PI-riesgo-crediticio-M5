"""
Entrenamiento y evaluación de modelos de riesgo crediticio.

Compara varios modelos supervisados, los evalúa con validación cruzada y
selecciona el de mejor performance. Genera una tabla resumen y un gráfico comparativo.
"""

import pandas as pd
import matplotlib.pyplot as plt

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

    # 4. Gráfico comparativo (recall de morosos y AUC por modelo):
    #    las dos métricas que muestran si el modelo sirve para el negocio.
    resultados[["recall_moroso", "auc"]].plot(kind="bar", figsize=(8, 5))
    plt.title("Comparación de modelos: recall de morosos y AUC")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("comparacion_modelos.png", dpi=120)
    plt.show()
    print("Gráfico guardado en comparacion_modelos.png")