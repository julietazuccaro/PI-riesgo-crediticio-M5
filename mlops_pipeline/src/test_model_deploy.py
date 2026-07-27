"""
Tests automatizados de la API de riesgo crediticio (`model_deploy.py`).

Se usa `TestClient` de FastAPI, que levanta la aplicación en memoria y le manda
solicitudes HTTP reales: no hace falta tener el servidor corriendo ni Docker.
El bloque `with TestClient(app)` es importante porque dispara el `lifespan`, es
decir, la carga del modelo serializado.

Qué se cubre:
- Endpoints de estado (`/`, `/health`, `/modelo`).
- Predicción individual, por lotes y desde CSV (el "camino feliz").
- Validación de entradas inválidas: el contrato de la API es parte del producto,
  y una API que acepta basura silenciosamente es peor que una que falla.
- Coherencia entre modos: el mismo registro debe dar el mismo resultado por
  `/predict` y por `/predict/batch`.

Ejecutar desde `mlops_pipeline/src`:
    pytest -v
"""

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ft_engineering import COLUMNAS_REQUERIDAS
from model_deploy import app, clasificar_riesgo

# Solicitud de referencia: un cliente "promedio" del dataset. Se usa como base
# en casi todos los tests y se modifica puntualmente cuando hace falta.
SOLICITUD_VALIDA = {
    "tipo_credito": 7,
    "capital_prestado": 2_500_000,
    "plazo_meses": 12,
    "edad_cliente": 44,
    "tipo_laboral": "Empleado",
    "salario_cliente": 3_200_000,
    "total_otros_prestamos": 5_800_000,
    "cuota_pactada": 245_000,
    "puntaje_datacredito": 780,
    "cant_creditosvigentes": 5,
    "huella_consulta": 4,
    "saldo_mora": 0,
    "saldo_total": 45_000,
    "saldo_principal": 40_000,
    "saldo_mora_codeudor": 0,
    "creditos_sectorFinanciero": 3,
    "creditos_sectorCooperativo": 0,
    "creditos_sectorReal": 1,
    "promedio_ingresos_datacredito": 2_000_000,
    "tendencia_ingresos": "Estable",
}


@pytest.fixture(scope="module")
def client():
    """Cliente de prueba compartido por todos los tests (carga el modelo una vez)."""
    with TestClient(app) as c:
        yield c


def csv_en_memoria(filas: int = 3, columnas_extra: bool = True) -> bytes:
    """Arma un CSV de prueba en memoria, sin tocar el disco.

    Por defecto agrega `fecha_prestamo` y `puntaje`, que existen en el dataset
    original pero NO forman parte del modelo: sirve para verificar que la API
    las descarta sin romperse.
    """
    df = pd.DataFrame([SOLICITUD_VALIDA] * filas)
    if columnas_extra:
        df["fecha_prestamo"] = "2025-06-15"
        df["puntaje"] = 91.2
    return df.to_csv(index=False).encode("utf-8")


# --- Endpoints de estado -----------------------------------------------------
def test_raiz_lista_los_endpoints(client):
    respuesta = client.get("/")
    assert respuesta.status_code == 200
    assert "endpoints" in respuesta.json()


def test_health_confirma_que_el_modelo_esta_cargado(client):
    respuesta = client.get("/health")
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["estado"] == "ok"
    assert datos["modelo_cargado"] is True


def test_modelo_expone_metadata_de_trazabilidad(client):
    respuesta = client.get("/modelo")
    assert respuesta.status_code == 200
    datos = respuesta.json()
    # Sin estos campos no se puede auditar qué versión generó una predicción.
    for campo in ("nombre", "version", "fecha_entrenamiento", "metricas_test"):
        assert campo in datos
    assert "auc" in datos["metricas_test"]


# --- Predicción individual ---------------------------------------------------
def test_predict_devuelve_la_estructura_completa(client):
    respuesta = client.post("/predict", json=SOLICITUD_VALIDA)
    assert respuesta.status_code == 200
    datos = respuesta.json()

    campos = {
        "id_solicitud", "probabilidad_mora", "probabilidad_pago", "prediccion",
        "etiqueta", "nivel_riesgo", "decision_sugerida", "umbral",
        "version_modelo", "timestamp",
    }
    assert campos.issubset(datos.keys())
    assert 0.0 <= datos["probabilidad_mora"] <= 1.0
    assert datos["prediccion"] in (0, 1)
    assert datos["nivel_riesgo"] in ("BAJO", "MEDIO", "ALTO")


def test_predict_probabilidades_suman_uno(client):
    datos = client.post("/predict", json=SOLICITUD_VALIDA).json()
    assert datos["probabilidad_mora"] + datos["probabilidad_pago"] == pytest.approx(1.0, abs=1e-6)


def test_predict_acepta_tendencia_ingresos_nula(client):
    """El único campo opcional: si el buró no lo reporta, lo imputa el pipeline."""
    solicitud = {k: v for k, v in SOLICITUD_VALIDA.items() if k != "tendencia_ingresos"}
    assert client.post("/predict", json=solicitud).status_code == 200


def test_predict_rechaza_edad_fuera_de_rango(client):
    solicitud = SOLICITUD_VALIDA | {"edad_cliente": 15}
    assert client.post("/predict", json=solicitud).status_code == 422


def test_predict_rechaza_campo_faltante(client):
    solicitud = {k: v for k, v in SOLICITUD_VALIDA.items() if k != "salario_cliente"}
    assert client.post("/predict", json=solicitud).status_code == 422


def test_predict_rechaza_categoria_desconocida(client):
    """`tipo_laboral` es un Literal: sólo admite las categorías del entrenamiento."""
    solicitud = SOLICITUD_VALIDA | {"tipo_laboral": "Jubilado"}
    assert client.post("/predict", json=solicitud).status_code == 422


def test_predict_rechaza_umbral_invalido(client):
    assert client.post("/predict?umbral=1.5", json=SOLICITUD_VALIDA).status_code == 422


def test_umbral_no_cambia_la_probabilidad_pero_si_la_decision(client):
    """El umbral es una perilla de negocio: mueve la decisión, no el modelo."""
    laxo = client.post("/predict?umbral=0.95", json=SOLICITUD_VALIDA).json()
    estricto = client.post("/predict?umbral=0.01", json=SOLICITUD_VALIDA).json()

    assert laxo["probabilidad_mora"] == estricto["probabilidad_mora"]
    # Con un umbral de 0.01 prácticamente cualquier solicitud queda por encima.
    assert estricto["decision_sugerida"] == "RECHAZAR"
    assert estricto["prediccion"] == 0


# --- Predicción por lotes (JSON) ---------------------------------------------
def test_batch_devuelve_una_prediccion_por_registro(client):
    lote = {"solicitudes": [SOLICITUD_VALIDA] * 3}
    respuesta = client.post("/predict/batch", json=lote)
    assert respuesta.status_code == 200

    datos = respuesta.json()
    assert len(datos["predicciones"]) == 3
    assert datos["resumen"]["total_registros"] == 3


def test_batch_resumen_es_coherente_con_las_predicciones(client):
    lote = {"solicitudes": [SOLICITUD_VALIDA] * 5}
    datos = client.post("/predict/batch", json=lote).json()
    resumen = datos["resumen"]

    # Los tres niveles de riesgo tienen que cubrir todos los registros.
    assert resumen["riesgo_alto"] + resumen["riesgo_medio"] + resumen["riesgo_bajo"] == 5

    promedio = sum(p["probabilidad_mora"] for p in datos["predicciones"]) / 5
    assert resumen["probabilidad_mora_promedio"] == pytest.approx(promedio, abs=1e-6)


def test_batch_rechaza_lote_vacio(client):
    assert client.post("/predict/batch", json={"solicitudes": []}).status_code == 422


def test_batch_coincide_con_la_prediccion_individual(client):
    """El resultado no puede depender del modo de consumo de la API."""
    individual = client.post("/predict", json=SOLICITUD_VALIDA).json()
    lote = client.post("/predict/batch", json={"solicitudes": [SOLICITUD_VALIDA]}).json()

    assert lote["predicciones"][0]["probabilidad_mora"] == pytest.approx(
        individual["probabilidad_mora"], abs=1e-9
    )


# --- Predicción por lotes (CSV) ----------------------------------------------
def test_csv_predice_todas_las_filas(client):
    archivos = {"archivo": ("solicitudes.csv", csv_en_memoria(4), "text/csv")}
    respuesta = client.post("/predict/csv", files=archivos)
    assert respuesta.status_code == 200
    assert respuesta.json()["resumen"]["total_registros"] == 4


def test_csv_coincide_con_la_prediccion_individual(client):
    """Las columnas extra (`fecha_prestamo`, `puntaje`) no deben alterar el score."""
    individual = client.post("/predict", json=SOLICITUD_VALIDA).json()
    archivos = {"archivo": ("solicitudes.csv", csv_en_memoria(1), "text/csv")}
    desde_csv = client.post("/predict/csv", files=archivos).json()

    assert desde_csv["predicciones"][0]["probabilidad_mora"] == pytest.approx(
        individual["probabilidad_mora"], abs=1e-9
    )


def test_csv_puede_devolver_un_archivo_descargable(client):
    archivos = {"archivo": ("solicitudes.csv", csv_en_memoria(3), "text/csv")}
    respuesta = client.post("/predict/csv?formato=csv", files=archivos)

    assert respuesta.status_code == 200
    assert respuesta.headers["content-type"].startswith("text/csv")

    salida = pd.read_csv(io.StringIO(respuesta.text))
    assert len(salida) == 3
    # El CSV de salida conserva los datos originales y suma el resultado.
    assert {"probabilidad_mora", "nivel_riesgo", "decision_sugerida"}.issubset(salida.columns)


def test_csv_avisa_que_columnas_faltan(client):
    df = pd.DataFrame([SOLICITUD_VALIDA]).drop(columns=["saldo_mora", "huella_consulta"])
    archivos = {"archivo": ("incompleto.csv", df.to_csv(index=False).encode(), "text/csv")}

    respuesta = client.post("/predict/csv", files=archivos)
    assert respuesta.status_code == 400
    # El mensaje tiene que nombrar las columnas: un 400 genérico no ayuda a nadie.
    assert "saldo_mora" in respuesta.json()["detail"]


def test_csv_rechaza_extension_invalida(client):
    archivos = {"archivo": ("datos.txt", b"no soy un csv", "text/plain")}
    assert client.post("/predict/csv", files=archivos).status_code == 400


# --- Reglas de negocio (test unitario, sin HTTP) ------------------------------
@pytest.mark.parametrize(
    "probabilidad, esperado",
    [
        (0.90, ("ALTO", "RECHAZAR")),    # por encima del umbral
        (0.50, ("ALTO", "RECHAZAR")),    # justo en el umbral -> se rechaza
        (0.40, ("MEDIO", "REVISAR")),    # zona gris (>= 70% del umbral)
        (0.10, ("BAJO", "APROBAR")),     # claramente por debajo
    ],
)
def test_clasificar_riesgo_aplica_las_bandas_de_decision(probabilidad, esperado):
    assert clasificar_riesgo(probabilidad, umbral=0.5) == esperado


def test_columnas_requeridas_coinciden_con_el_esquema_de_la_api():
    """El contrato de `/predict` y el del CSV tienen que ser el mismo."""
    from model_deploy import SolicitudCredito

    assert set(SolicitudCredito.model_fields) == set(COLUMNAS_REQUERIDAS)
