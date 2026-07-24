"""
Ingeniería de características del proyecto de riesgo crediticio.

Carga los datos, los limpia, arma el pipeline de preprocesamiento y devuelve
los conjuntos de entrenamiento y evaluación listos para entrenar modelos.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from Cargar_datos import cargar_datos 


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
      df = cargar_datos()

      # --- Limpieza (según los hallazgos del EDA) ---
      validas = ["Creciente", "Decreciente", "Estable"]
      df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
          df["tendencia_ingresos"].isin(validas), np.nan
      )
      df = df.drop(columns=["fecha_prestamo", "puntaje"])   # puntaje: se descarta por data leakage

      # --- Features nuevos (ratios con sentido de negocio) ---
      df["ratio_cuota_salario"] = df["cuota_pactada"] / df["salario_cliente"]
      df["ratio_deuda_ingreso"] = df["total_otros_prestamos"] / df["salario_cliente"]
      df = df.replace([np.inf, -np.inf], np.nan)   # división por 0 -> NaN

      # --- Definir columnas por tipo ---
      cat_cols = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]
      num_cols = [c for c in df.columns if c not in cat_cols + ["Pago_atiempo"]]

      # --- Separar X / y ---
      X = df.drop(columns=["Pago_atiempo"])
      y = df["Pago_atiempo"]

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