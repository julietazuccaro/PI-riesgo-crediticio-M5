"""
App de Streamlit para el monitoreo de data drift del modelo de riesgo crediticio.

Reutiliza las funciones de `model_monitoring.py` y presenta:
1. Visualización de métricas: tabla de drift por variable, semáforo, barras de
   riesgo y comparación de distribuciones (histórica vs actual).
2. Análisis temporal: evolución del drift mes a mes y detección de cambios abruptos.
3. Recomendaciones automáticas: alertas y sugerencias de reentrenamiento.

Ejecutar desde la carpeta `mlops_pipeline/src`:
    streamlit run app_monitoreo.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import model_monitoring as mm

st.set_page_config(page_title="Monitoreo de Data Drift", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Carga de datos cacheada (entrena el modelo una sola vez)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar():
    return mm.cargar_datos_monitoreo()


@st.cache_data
def evolucion_temporal():
    return mm.drift_temporal(cargar())


df = cargar()
df_ref = df[df["fecha_prestamo"] < mm.FECHA_CORTE]

# Meses de monitoreo con datos suficientes.
post = df[df["fecha_prestamo"] >= mm.FECHA_CORTE].copy()
post["mes"] = post["fecha_prestamo"].dt.to_period("M").astype(str)
meses = sorted(m for m, g in post.groupby("mes") if len(g) >= mm.MIN_FILAS)


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")
st.sidebar.markdown(
    f"**Población de referencia:** préstamos anteriores a "
    f"`{mm.FECHA_CORTE}` ({len(df_ref):,} registros)."
)
mes_sel = st.sidebar.selectbox(
    "Mes a monitorear (vs referencia):", meses, index=len(meses) - 1
)
st.sidebar.markdown(
    f"**Umbrales PSI:** 🟢 < {mm.PSI_MODERADO} · "
    f"🟡 {mm.PSI_MODERADO}–{mm.PSI_ALTO} · 🔴 > {mm.PSI_ALTO}"
)

df_act = post[post["mes"] == mes_sel]
reporte = mm.reporte_drift(df_ref, df_act)


# ---------------------------------------------------------------------------
# Encabezado y KPIs
# ---------------------------------------------------------------------------
st.title("📊 Monitoreo de Data Drift — Riesgo Crediticio")
st.caption(
    "Detecta si la población de nuevos préstamos se aleja de la población con la "
    "que se entrenó el modelo. Un drift alto anticipa una posible degradación del modelo."
)

psi_prom = reporte["psi"].mean()
n_alerta = int((reporte["psi"] >= mm.PSI_MODERADO).sum())
n_criticas = int((reporte["psi"] >= mm.PSI_ALTO).sum())

# Semáforo global según la peor variable del mes.
if n_criticas > 0:
    estado, color = "🔴 Drift Alto", "red"
elif n_alerta > 0:
    estado, color = "🟡 Drift Moderado", "orange"
else:
    estado, color = "🟢 Estable", "green"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Préstamos en el mes", f"{len(df_act):,}")
c2.metric("PSI promedio", f"{psi_prom:.3f}")
c3.metric("Variables en alerta", f"{n_alerta} / {len(reporte)}")
c4.metric("Variables críticas", n_criticas)
st.markdown(f"### Estado general del mes {mes_sel}: :{color}[{estado}]")

st.divider()


# ---------------------------------------------------------------------------
# 1. Visualización de métricas
# ---------------------------------------------------------------------------
st.header("1️⃣ Métricas de drift por variable")

col_tabla, col_barras = st.columns([1.1, 1])

with col_tabla:
    st.subheader("Tabla de drift")
    st.dataframe(
        reporte[["variable", "tipo", "psi", "js", "ks_pvalor", "chi2_pvalor", "severidad"]]
        .style.format({"psi": "{:.3f}", "js": "{:.3f}",
                       "ks_pvalor": "{:.3f}", "chi2_pvalor": "{:.3f}"}),
        width="stretch", height=500,
    )

with col_barras:
    st.subheader("Barras de riesgo (PSI)")
    colores = reporte["psi"].apply(
        lambda p: "red" if p >= mm.PSI_ALTO else ("orange" if p >= mm.PSI_MODERADO else "green")
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(reporte["variable"], reporte["psi"], color=colores)
    ax.axvline(mm.PSI_MODERADO, color="orange", ls="--", lw=1, label="Moderado (0.10)")
    ax.axvline(mm.PSI_ALTO, color="red", ls="--", lw=1, label="Alto (0.25)")
    ax.invert_yaxis()
    ax.set_xlabel("PSI")
    ax.legend(fontsize=8)
    st.pyplot(fig)


# --- Comparación de distribuciones histórica vs actual ---
st.subheader("Distribución histórica vs actual")
variable = st.selectbox("Elegí una variable para comparar:", list(reporte["variable"]))
tipo_var = reporte.set_index("variable").loc[variable, "tipo"]

fig2, ax2 = plt.subplots(figsize=(9, 4))
if tipo_var == "numérica":
    ax2.hist(df_ref[variable].dropna(), bins=40, density=True, alpha=0.5,
             label="Referencia (histórica)", color="steelblue")
    ax2.hist(df_act[variable].dropna(), bins=40, density=True, alpha=0.5,
             label=f"Actual ({mes_sel})", color="darkorange")
    ax2.set_xlabel(variable)
    ax2.set_ylabel("Densidad")
else:
    ref_pct = df_ref[variable].astype(str).value_counts(normalize=True)
    act_pct = df_act[variable].astype(str).value_counts(normalize=True)
    comp = pd.DataFrame({"Referencia": ref_pct, f"Actual ({mes_sel})": act_pct}).fillna(0)
    comp.plot(kind="bar", ax=ax2, color=["steelblue", "darkorange"])
    ax2.set_ylabel("Proporción")
ax2.legend()
ax2.set_title(f"{variable}: referencia vs {mes_sel}")
st.pyplot(fig2)

st.divider()


# ---------------------------------------------------------------------------
# 2. Análisis temporal
# ---------------------------------------------------------------------------
st.header("2️⃣ Análisis temporal del drift")

evol = evolucion_temporal()

col_ev1, col_ev2 = st.columns(2)
with col_ev1:
    st.subheader("Evolución del PSI")
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(evol["mes"], evol["psi_promedio"], marker="o", label="PSI promedio")
    ax3.plot(evol["mes"], evol["psi_maximo"], marker="s", ls="--", label="PSI máximo")
    ax3.axhline(mm.PSI_MODERADO, color="orange", ls=":", lw=1)
    ax3.axhline(mm.PSI_ALTO, color="red", ls=":", lw=1)
    ax3.set_ylabel("PSI")
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    ax3.legend()
    st.pyplot(fig3)

with col_ev2:
    st.subheader("Variables en alerta por mes")
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.bar(evol["mes"], evol["vars_en_alerta"], color="orange", alpha=0.7, label="En alerta")
    ax4.bar(evol["mes"], evol["vars_criticas"], color="red", label="Críticas")
    ax4.set_ylabel("Cantidad de variables")
    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
    ax4.legend()
    st.pyplot(fig4)

# Detección de cambios abruptos: mayor salto de PSI promedio entre meses.
evol_salto = evol.copy()
evol_salto["salto"] = evol_salto["psi_promedio"].diff()
if len(evol_salto) > 1 and evol_salto["salto"].max() > 0:
    peor = evol_salto.loc[evol_salto["salto"].idxmax()]
    st.info(
        f"📈 **Cambio más abrupto:** en **{peor['mes']}** el PSI promedio saltó "
        f"+{peor['salto']:.3f} respecto del mes anterior."
    )

st.dataframe(evol.style.format({"psi_promedio": "{:.3f}", "psi_maximo": "{:.3f}"}),
             width="stretch")

st.divider()


# ---------------------------------------------------------------------------
# 3. Recomendaciones automáticas
# ---------------------------------------------------------------------------
st.header("3️⃣ Recomendaciones automáticas")

for msg in mm.generar_recomendaciones(reporte):
    if msg.startswith("🔴") or msg.startswith("⚠️"):
        st.error(msg)
    elif msg.startswith("🟡"):
        st.warning(msg)
    else:
        st.success(msg)
