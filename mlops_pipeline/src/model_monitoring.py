"""
Monitoreo del modelo de riesgo crediticio y detección de data drift.

Este módulo simula el trabajo de monitoreo: toma los datos junto con los
pronósticos del modelo y, dividiéndolos en ventanas temporales según
`fecha_prestamo`, mide cuánto se aleja la población reciente ("actual") de la
población con la que se entrenó el modelo ("referencia"). Ese alejamiento es el
*data drift*: si es grande, el modelo puede degradarse y conviene reentrenar.

Métricas implementadas:
- PSI (Population Stability Index): numéricas y categóricas.
- KS (Kolmogorov-Smirnov): numéricas.
- Jensen-Shannon: numéricas y categóricas.
- Chi-cuadrado: categóricas.

Expone funciones reutilizables por la app de Streamlit (`app_monitoreo.py`).
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from Cargar_datos import cargar_datos
from ft_engineering import construir_preprocesador


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
# Fecha que separa la población de referencia (histórica) de la de monitoreo.
# Referencia = préstamos anteriores a esta fecha (primeros ~6 meses).
FECHA_CORTE = "2025-05-01"

# Umbrales estándar de PSI para clasificar la severidad del drift.
PSI_MODERADO = 0.10   # 0.10 - 0.25 -> cambio moderado (amarillo)
PSI_ALTO = 0.25       # > 0.25      -> cambio significativo (rojo)

# Nivel de significancia para los tests de hipótesis (KS y chi-cuadrado).
PVALOR_ALERTA = 0.05

# Mínimo de filas en una ventana para que las métricas sean confiables.
MIN_FILAS = 30

# Variables a monitorear (las que usa el modelo + el score del modelo).
NUM_COLS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "saldo_total", "promedio_ingresos_datacredito",
    "prob_moroso",
]
CAT_COLS = ["tipo_credito", "tipo_laboral", "tendencia_ingresos", "pred_moroso"]


# ---------------------------------------------------------------------------
# 1. Cargar datos con los pronósticos del modelo
# ---------------------------------------------------------------------------
def cargar_datos_monitoreo() -> pd.DataFrame:
    """Carga los datos y agrega los pronósticos del modelo (score y clase).

    Para simular un entorno productivo, el modelo se entrena solo con la
    población de referencia (los primeros meses) y luego puntúa TODAS las filas.
    Así, las ventanas posteriores representan datos "nuevos" que el modelo nunca
    vio, igual que en producción.

    Devuelve el DataFrame original más dos columnas:
    - `prob_moroso`: probabilidad estimada de no pagar a tiempo (clase 0).
    - `pred_moroso`: clase predicha (1 = el modelo lo marca como moroso).
    """
    df = cargar_datos()
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])

    # Limpieza mínima de tendencia_ingresos (misma regla que en ft_engineering).
    validas = ["Creciente", "Decreciente", "Estable"]
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(validas), np.nan
    )

    # Features del modelo (se descarta `puntaje` por leakage, igual que en el
    # entrenamiento) y se separan por tipo para el preprocesador.
    cat_modelo = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]
    excluir = ["Pago_atiempo", "fecha_prestamo", "puntaje"]
    num_modelo = [
        c for c in df.columns
        if c not in cat_modelo + excluir and df[c].dtype != "object"
    ]

    X = df[num_modelo + cat_modelo]
    y = df["Pago_atiempo"]

    # Entrenar el modelo elegido en el Avance 2 SOLO con la referencia.
    ref_mask = df["fecha_prestamo"] < FECHA_CORTE
    modelo = Pipeline([
        ("preprocesador", construir_preprocesador(num_modelo, cat_modelo)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    modelo.fit(X[ref_mask], y[ref_mask])

    # Puntuar todas las filas. La clase 0 es "moroso": su probabilidad está en
    # la columna 0 de predict_proba.
    df["prob_moroso"] = modelo.predict_proba(X)[:, 0]
    df["pred_moroso"] = (modelo.predict(X) == 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Métricas de drift
# ---------------------------------------------------------------------------
def _psi_desde_proporciones(ref_pct: np.ndarray, act_pct: np.ndarray) -> float:
    """PSI a partir de dos vectores de proporciones ya alineados."""
    # Se acota para evitar log(0) o división por cero en bins vacíos.
    ref_pct = np.clip(ref_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)
    return float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))


def psi_numerico(referencia: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """PSI para una variable numérica, usando bins por cuantiles de la referencia."""
    bordes = np.quantile(referencia, np.linspace(0, 1, n_bins + 1))
    bordes = np.unique(bordes)               # evita bins repetidos si hay muchos valores iguales
    bordes[0], bordes[-1] = -np.inf, np.inf  # cubrir valores fuera del rango de referencia
    ref_pct = np.histogram(referencia, bins=bordes)[0] / len(referencia)
    act_pct = np.histogram(actual, bins=bordes)[0] / len(actual)
    return _psi_desde_proporciones(ref_pct, act_pct)


def psi_categorico(referencia: pd.Series, actual: pd.Series) -> float:
    """PSI para una variable categórica, comparando la proporción de cada categoría."""
    categorias = referencia.value_counts(normalize=True).index.union(
        actual.value_counts(normalize=True).index
    )
    ref_pct = referencia.value_counts(normalize=True).reindex(categorias).fillna(0).to_numpy()
    act_pct = actual.value_counts(normalize=True).reindex(categorias).fillna(0).to_numpy()
    return _psi_desde_proporciones(ref_pct, act_pct)


def js_numerico(referencia: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """Distancia de Jensen-Shannon (0 a 1) para una variable numérica."""
    bordes = np.unique(np.quantile(referencia, np.linspace(0, 1, n_bins + 1)))
    ref_pct = np.histogram(referencia, bins=bordes)[0] / len(referencia)
    act_pct = np.histogram(actual, bins=bordes)[0] / len(actual)
    return float(jensenshannon(ref_pct, act_pct, base=2))


def js_categorico(referencia: pd.Series, actual: pd.Series) -> float:
    """Distancia de Jensen-Shannon (0 a 1) para una variable categórica."""
    categorias = referencia.value_counts(normalize=True).index.union(
        actual.value_counts(normalize=True).index
    )
    ref_pct = referencia.value_counts(normalize=True).reindex(categorias).fillna(0).to_numpy()
    act_pct = actual.value_counts(normalize=True).reindex(categorias).fillna(0).to_numpy()
    return float(jensenshannon(ref_pct, act_pct, base=2))


def chi2_categorico(referencia: pd.Series, actual: pd.Series) -> tuple[float, float]:
    """Chi-cuadrado de independencia entre referencia y actual. Devuelve (estadístico, p-valor)."""
    categorias = referencia.value_counts().index.union(actual.value_counts().index)
    tabla = pd.DataFrame({
        "referencia": referencia.value_counts().reindex(categorias).fillna(0),
        "actual": actual.value_counts().reindex(categorias).fillna(0),
    })
    # Se descartan categorías sin observaciones en ninguna de las dos ventanas.
    tabla = tabla[tabla.sum(axis=1) > 0]
    estadistico, pvalor, _, _ = chi2_contingency(tabla.to_numpy())
    return float(estadistico), float(pvalor)


# ---------------------------------------------------------------------------
# 3. Clasificación de severidad
# ---------------------------------------------------------------------------
def clasificar_severidad(psi: float) -> str:
    """Traduce el PSI a una etiqueta de severidad (semáforo)."""
    if psi < PSI_MODERADO:
        return "🟢 Estable"
    if psi < PSI_ALTO:
        return "🟡 Moderado"
    return "🔴 Alto"


# ---------------------------------------------------------------------------
# 4. Reporte de drift por variable (referencia vs una ventana actual)
# ---------------------------------------------------------------------------
def reporte_drift(df_ref: pd.DataFrame, df_act: pd.DataFrame,
                  num_cols: list = None, cat_cols: list = None) -> pd.DataFrame:
    """Compara dos poblaciones y devuelve una tabla de drift, una fila por variable."""
    num_cols = num_cols if num_cols is not None else NUM_COLS
    cat_cols = cat_cols if cat_cols is not None else CAT_COLS
    filas = []

    for col in num_cols:
        ref = df_ref[col].dropna()
        act = df_act[col].dropna()
        if len(ref) < MIN_FILAS or len(act) < MIN_FILAS:
            continue
        psi = psi_numerico(ref, act)
        ks_stat, ks_p = ks_2samp(ref, act)
        filas.append({
            "variable": col, "tipo": "numérica",
            "psi": psi, "ks_stat": float(ks_stat), "ks_pvalor": float(ks_p),
            "js": js_numerico(ref, act), "severidad": clasificar_severidad(psi),
        })

    for col in cat_cols:
        ref = df_ref[col].dropna().astype(str)
        act = df_act[col].dropna().astype(str)
        if len(ref) < MIN_FILAS or len(act) < MIN_FILAS:
            continue
        psi = psi_categorico(ref, act)
        chi2, chi2_p = chi2_categorico(ref, act)
        filas.append({
            "variable": col, "tipo": "categórica",
            "psi": psi, "chi2": chi2, "chi2_pvalor": chi2_p,
            "js": js_categorico(ref, act), "severidad": clasificar_severidad(psi),
        })

    columnas = ["variable", "tipo", "psi", "ks_stat", "ks_pvalor",
                "chi2", "chi2_pvalor", "js", "severidad"]
    if not filas:  # ninguna ventana tuvo filas suficientes
        return pd.DataFrame(columns=columnas)
    return pd.DataFrame(filas).sort_values("psi", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Análisis temporal: evolución del drift mes a mes
# ---------------------------------------------------------------------------
def drift_temporal(df: pd.DataFrame, num_cols: list = None, cat_cols: list = None,
                   fecha_corte: str = FECHA_CORTE) -> pd.DataFrame:
    """Calcula un resumen de drift por mes (cada mes vs la referencia).

    Para cada mes posterior a `fecha_corte` devuelve el PSI promedio y máximo, la
    cantidad de variables en alerta y el número de préstamos de ese mes. Sirve
    para graficar la evolución del drift y detectar cambios abruptos.
    """
    num_cols = num_cols if num_cols is not None else NUM_COLS
    cat_cols = cat_cols if cat_cols is not None else CAT_COLS

    df_ref = df[df["fecha_prestamo"] < fecha_corte]
    df_post = df[df["fecha_prestamo"] >= fecha_corte].copy()
    df_post["mes"] = df_post["fecha_prestamo"].dt.to_period("M").astype(str)

    filas = []
    for mes, grupo in df_post.groupby("mes"):
        if len(grupo) < MIN_FILAS:
            continue
        rep = reporte_drift(df_ref, grupo, num_cols, cat_cols)
        filas.append({
            "mes": mes,
            "n_prestamos": len(grupo),
            "psi_promedio": rep["psi"].mean(),
            "psi_maximo": rep["psi"].max(),
            "vars_en_alerta": int((rep["psi"] >= PSI_MODERADO).sum()),
            "vars_criticas": int((rep["psi"] >= PSI_ALTO).sum()),
        })

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 6. Recomendaciones automáticas
# ---------------------------------------------------------------------------
def generar_recomendaciones(reporte: pd.DataFrame) -> list[str]:
    """Genera mensajes de alerta y sugerencias según el nivel de drift detectado."""
    mensajes = []
    criticas = reporte[reporte["psi"] >= PSI_ALTO]["variable"].tolist()
    moderadas = reporte[(reporte["psi"] >= PSI_MODERADO) &
                        (reporte["psi"] < PSI_ALTO)]["variable"].tolist()

    if criticas:
        mensajes.append(
            f"🔴 Drift SIGNIFICATIVO en: {', '.join(criticas)}. "
            "Se recomienda reentrenar el modelo con datos recientes."
        )
    if moderadas:
        mensajes.append(
            f"🟡 Drift moderado en: {', '.join(moderadas)}. "
            "Conviene revisar estas variables y monitorearlas de cerca."
        )
    if not criticas and not moderadas:
        mensajes.append("🟢 Sin drift relevante. El modelo opera dentro de lo esperado.")

    if "prob_moroso" in criticas:
        mensajes.append(
            "⚠️ El drift afecta al score del modelo (prob_moroso): "
            "es la señal más directa de posible degradación del desempeño."
        )
    return mensajes


# ---------------------------------------------------------------------------
# Ejecución directa: reporte de ejemplo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = cargar_datos_monitoreo()
    df_ref = df[df["fecha_prestamo"] < FECHA_CORTE]
    post = df[df["fecha_prestamo"] >= FECHA_CORTE].copy()
    post["mes"] = post["fecha_prestamo"].dt.to_period("M").astype(str)
    # Último mes con datos suficientes (los meses muy chicos se descartan).
    meses_validos = [m for m, g in post.groupby("mes") if len(g) >= MIN_FILAS]
    mes_reciente = sorted(meses_validos)[-1]
    df_act = post[post["mes"] == mes_reciente]

    print(f"Referencia: {len(df_ref)} préstamos (< {FECHA_CORTE})")
    print(f"Ventana actual: {mes_reciente} ({len(df_act)} préstamos)\n")

    print(f"=== Drift por variable (referencia vs {mes_reciente}) ===")
    print(reporte_drift(df_ref, df_act).round(3).to_string(index=False))

    print("\n=== Evolución temporal del drift ===")
    print(drift_temporal(df).round(3).to_string(index=False))

    print("\n=== Recomendaciones ===")
    for m in generar_recomendaciones(reporte_drift(df_ref, df_act)):
        print("-", m)
