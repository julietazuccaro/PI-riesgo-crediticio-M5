"""
Ingeniería de características del proyecto de riesgo crediticio.

Carga los datos, los limpia, arma el pipeline de preprocesamiento y devuelve
los conjuntos de entrenamiento y evaluación listos para entrenar modelos.

Las transformaciones que NO aprenden nada de los datos (limpieza de valores
inválidos y creación de ratios) se aíslan en `preparar_features()`, de modo que
el **mismo código** se use al entrenar (`model_training_evaluation.py`) y al
servir predicciones (`model_deploy.py`). Esto evita el *training/serving skew*:
que en producción el modelo reciba features calculadas de una forma distinta a
la del entrenamiento, y por lo tanto prediga sobre datos que no reconoce.

Las transformaciones que SÍ aprenden de los datos (imputación, escalado,
codificación) viven dentro del `ColumnTransformer`, que se ajusta únicamente
con el set de entrenamiento y viaja serializado junto al modelo.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Cargar_datos import cargar_datos

# --- Contrato de datos: usado por el entrenamiento y por la API ---------------
OBJETIVO = "Pago_atiempo"

# Columnas tratadas como categóricas (el resto se considera numérico).
COLUMNAS_CATEGORICAS = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]

# Columnas que se descartan del modelo:
#   - puntaje: correlaciona 0.92 con el objetivo -> data leakage (ver EDA).
#   - fecha_prestamo: identificador temporal, se reserva para el monitoreo.
COLUMNAS_DESCARTADAS = ["fecha_prestamo", "puntaje"]

# Únicos valores válidos de `tendencia_ingresos`; el EDA detectó 58 registros
# con montos numéricos cargados por error en esta columna de texto.
TENDENCIAS_VALIDAS = ("Creciente", "Decreciente", "Estable")

# Ratios derivados que se agregan como features.
COLUMNAS_RATIOS = ["ratio_cuota_salario", "ratio_deuda_ingreso"]

# Campos que la API debe recibir sí o sí para poder predecir (los ratios no se
# piden: se calculan). Se deriva de las columnas del dataset original.
COLUMNAS_REQUERIDAS = [
    "tipo_credito",
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "tipo_laboral",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    "tendencia_ingresos",
]


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza los valores inválidos detectados en el EDA.

    `tendencia_ingresos` es una variable de texto ("Creciente", "Decreciente",
    "Estable") pero trae registros con importes numéricos cargados por error.
    Todo lo que no sea una categoría válida se convierte en nulo para que
    después lo impute el pipeline (moda), en lugar de generar categorías basura
    en el One-Hot Encoding.
    """
    df = df.copy()
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(TENDENCIAS_VALIDAS), np.nan
    )
    return df


def agregar_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega los dos ratios con sentido de negocio usados como features.

    - `ratio_cuota_salario`: qué proporción del salario se lleva la cuota
      (capacidad de pago).
    - `ratio_deuda_ingreso`: nivel de endeudamiento previo respecto del ingreso.

    Ambos dividen por `salario_cliente`, que puede ser 0: esas divisiones dan
    infinito y se convierten en nulo para que las impute el pipeline.
    """
    df = df.copy()
    df["ratio_cuota_salario"] = df["cuota_pactada"] / df["salario_cliente"]
    df["ratio_deuda_ingreso"] = df["total_otros_prestamos"] / df["salario_cliente"]
    df[COLUMNAS_RATIOS] = df[COLUMNAS_RATIOS].replace([np.inf, -np.inf], np.nan)
    return df


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la ingeniería de características que NO depende del entrenamiento.

    Es el punto de entrada compartido entre el entrenamiento y la API: limpia,
    descarta las columnas que no van al modelo y agrega los ratios derivados.
    Las columnas a descartar se eliminan con `errors="ignore"` porque la API
    recibe solicitudes que directamente no las traen.
    """
    df = limpiar_datos(df)
    df = df.drop(columns=COLUMNAS_DESCARTADAS, errors="ignore")
    return agregar_ratios(df)


def construir_preprocesador(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Arma el pipeline de preprocesamiento: imputación + escalado + codificación."""
    # Ruta numérica: imputar nulos con la mediana + escalar
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Ruta categórica: imputar con el más frecuente + codificar (one-hot)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Combinar las dos rutas
    preprocesador = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return preprocesador


def preparar_datos(test_size: float = 0.2, random_state: int = 42):
    """Carga los datos, los limpia, define features y devuelve train/test + el preprocesador."""
    # --- Limpieza + features derivados (mismo código que usa la API) ---
    df = preparar_features(cargar_datos())

    # --- Definir columnas por tipo ---
    cat_cols = COLUMNAS_CATEGORICAS
    num_cols = [c for c in df.columns if c not in cat_cols + [OBJETIVO]]

    # --- Separar X / y ---
    X = df.drop(columns=[OBJETIVO])
    y = df[OBJETIVO]

    # --- Split train/test (estratificado por el desbalance) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # --- Armar el preprocesador ---
    preprocesador = construir_preprocesador(num_cols, cat_cols)

    return X_train, X_test, y_train, y_test, preprocesador


if __name__ == "__main__":
    # Prueba rápida: que la función devuelva los datos y el preprocesador funcione
    X_train, X_test, y_train, y_test, preprocesador = preparar_datos()
    print("Train:", X_train.shape, "| Test:", X_test.shape)
    print("Proporción de pago a tiempo (train):", round(y_train.mean(), 3))

    # probar que el preprocesador transforma sin errores
    X_train_proc = preprocesador.fit_transform(X_train)
    print("Shape después de preprocesar:", X_train_proc.shape)
