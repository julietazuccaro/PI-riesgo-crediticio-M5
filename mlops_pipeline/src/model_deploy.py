"""
Despliegue del modelo de riesgo crediticio como API REST (FastAPI).

Toma el mejor modelo seleccionado en `model_training_evaluation.py` (serializado
en `modelo_riesgo.joblib`) y lo expone como un servicio HTTP que cualquier
sistema de la financiera puede consumir: formularios digitales, el CRM o un
proceso batch nocturno.

Endpoints
---------
GET  /               Información general del servicio.
GET  /health         Estado del servicio (para Docker / orquestadores).
GET  /modelo         Metadata del modelo en producción (versión, métricas).
POST /predict        Predicción individual (JSON).
POST /predict/batch  Predicción por lotes (lista de JSON en una sola solicitud).
POST /predict/csv    Predicción por lotes subiendo un archivo CSV.

La documentación interactiva (Swagger UI) queda disponible en /docs.

Ejecutar localmente desde `mlops_pipeline/src`:
    uvicorn model_deploy:app --reload --port 8000
    # o bien:  python model_deploy.py
"""

import io
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile, status
from pydantic import BaseModel, Field

from ft_engineering import COLUMNAS_REQUERIDAS, preparar_features

# --- Configuración -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("api-riesgo-crediticio")

# Ruta del artefacto. Se puede sobrescribir con la variable de entorno
# RUTA_MODELO (útil en Docker o para probar otra versión del modelo sin tocar
# el código: es configuración, no lógica).
RUTA_ARTEFACTO = Path(
    os.getenv("RUTA_MODELO", Path(__file__).resolve().parent / "modelo_riesgo.joblib")
)

# Umbral por defecto sobre la probabilidad de mora. Es una decisión de negocio,
# no del modelo: cuanto más bajo, más solicitudes se frenan (se detectan más
# morosos, pero también se rechazan más buenos clientes).
UMBRAL_DEFECTO = 0.5

# Tope de registros por lote: evita que una sola solicitud tumbe el servicio.
MAX_REGISTROS_LOTE = 5_000

# Estado del servicio: el artefacto se carga UNA sola vez al arrancar y queda en
# memoria. Cargarlo en cada request multiplicaría la latencia innecesariamente.
ARTEFACTO: Optional[dict] = None


def cargar_artefacto() -> None:
    """Carga el modelo serializado en memoria. Se ejecuta al iniciar la API."""
    global ARTEFACTO
    if not RUTA_ARTEFACTO.exists():
        # No se corta el arranque: la API levanta en modo "degradado" y lo
        # informa en /health. Así el contenedor no entra en un loop de reinicios
        # y el problema queda visible para quien monitorea.
        logger.error(
            "No se encontró el modelo en %s. Ejecutá "
            "'python model_training_evaluation.py' para generarlo.",
            RUTA_ARTEFACTO,
        )
        ARTEFACTO = None
        return

    ARTEFACTO = joblib.load(RUTA_ARTEFACTO)
    logger.info(
        "Modelo cargado: %s v%s (entrenado el %s)",
        ARTEFACTO["nombre"], ARTEFACTO["version"], ARTEFACTO["fecha_entrenamiento"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación: carga el modelo antes de aceptar tráfico."""
    cargar_artefacto()
    yield
    logger.info("API detenida.")


app = FastAPI(
    title="API de Riesgo Crediticio",
    description=(
        "Servicio de scoring crediticio: estima la **probabilidad de mora** de una "
        "solicitud de crédito y sugiere una decisión.\n\n"
        "El modelo en producción es el ganador de la comparación hecha en "
        "`model_training_evaluation.py` y viaja con todo su pipeline de "
        "preprocesamiento, por lo que la API recibe los datos **crudos** del "
        "solicitante (sin transformar).\n\n"
        "Modos de uso: predicción individual (`/predict`), por lotes vía JSON "
        "(`/predict/batch`) y por lotes vía archivo CSV (`/predict/csv`)."
    ),
    version="1.3.0",
    lifespan=lifespan,
)


# --- Esquemas de entrada y salida (Pydantic) ---------------------------------
# Pydantic valida automáticamente tipos y rangos: si un campo falta o viene mal,
# FastAPI responde 422 con el detalle del error antes de tocar el modelo.
class SolicitudCredito(BaseModel):
    """Datos crudos de una solicitud de crédito, tal como los toma el formulario."""

    tipo_credito: int = Field(..., ge=0, description="Código del producto crediticio")
    capital_prestado: float = Field(..., gt=0, description="Monto solicitado")
    plazo_meses: int = Field(..., ge=1, le=120, description="Plazo del crédito en meses")
    edad_cliente: int = Field(..., ge=18, le=100, description="Edad del solicitante")
    tipo_laboral: Literal["Empleado", "Independiente"]
    salario_cliente: float = Field(..., ge=0, description="Ingreso mensual declarado")
    total_otros_prestamos: float = Field(..., ge=0, description="Deuda vigente en otras entidades")
    cuota_pactada: float = Field(..., gt=0, description="Cuota mensual del crédito solicitado")
    puntaje_datacredito: float = Field(..., ge=-100, le=1000, description="Score del buró de crédito")
    cant_creditosvigentes: int = Field(..., ge=0, description="Cantidad de créditos activos")
    huella_consulta: int = Field(..., ge=0, description="Consultas recientes al buró")
    saldo_mora: float = Field(..., ge=0, description="Saldo actualmente en mora")
    saldo_total: float = Field(..., ge=0, description="Saldo total adeudado")
    saldo_principal: float = Field(..., ge=0, description="Saldo de capital adeudado")
    saldo_mora_codeudor: float = Field(..., ge=0, description="Saldo en mora del codeudor")
    creditos_sectorFinanciero: int = Field(..., ge=0)
    creditos_sectorCooperativo: int = Field(..., ge=0)
    creditos_sectorReal: int = Field(..., ge=0)
    promedio_ingresos_datacredito: float = Field(..., ge=0, description="Ingreso promedio según el buró")
    # Único campo opcional: si el buró no lo reporta, el pipeline lo imputa.
    tendencia_ingresos: Optional[Literal["Creciente", "Decreciente", "Estable"]] = Field(
        None, description="Tendencia de ingresos reportada por el buró"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class LoteSolicitudes(BaseModel):
    """Varias solicitudes enviadas en una sola llamada (predicción por lotes)."""

    solicitudes: list[SolicitudCredito] = Field(
        ..., min_length=1, max_length=MAX_REGISTROS_LOTE,
        description="Lista de solicitudes a evaluar",
    )


class RespuestaPrediccion(BaseModel):
    """Resultado del scoring de una solicitud."""

    id_solicitud: str = Field(..., description="Identificador único de la predicción (trazabilidad)")
    probabilidad_mora: float = Field(..., description="Probabilidad de NO pagar a tiempo (clase 0)")
    probabilidad_pago: float = Field(..., description="Probabilidad de pagar a tiempo (clase 1)")
    prediccion: int = Field(..., description="1 = paga a tiempo · 0 = riesgo de mora")
    etiqueta: str = Field(..., description="Lectura en texto de la predicción")
    nivel_riesgo: str = Field(..., description="BAJO · MEDIO · ALTO")
    decision_sugerida: str = Field(..., description="APROBAR · REVISAR · RECHAZAR")
    umbral: float = Field(..., description="Umbral de corte aplicado")
    version_modelo: str
    timestamp: str


class ResumenLote(BaseModel):
    """Agregados del lote, para no tener que recorrer las predicciones una por una."""

    total_registros: int
    riesgo_alto: int
    riesgo_medio: int
    riesgo_bajo: int
    probabilidad_mora_promedio: float


class RespuestaLote(BaseModel):
    """Resultado de una predicción por lotes."""

    id_lote: str
    version_modelo: str
    umbral: float
    resumen: ResumenLote
    predicciones: list[RespuestaPrediccion]
    timestamp: str


# --- Lógica de predicción ----------------------------------------------------
def obtener_artefacto() -> dict:
    """Devuelve el artefacto cargado o corta con 503 si el servicio está degradado."""
    if ARTEFACTO is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El modelo no está disponible. Generá el artefacto ejecutando "
                "'python model_training_evaluation.py'."
            ),
        )
    return ARTEFACTO


def clasificar_riesgo(probabilidad_mora: float, umbral: float) -> tuple[str, str]:
    """Traduce una probabilidad a un nivel de riesgo y una decisión de negocio.

    La probabilidad sola no le sirve a un analista de crédito: necesita una
    recomendación accionable. Se define una banda intermedia (a partir del 70%
    del umbral) para los casos limítrofes, que se derivan a revisión manual en
    vez de aprobarse o rechazarse automáticamente.
    """
    if probabilidad_mora >= umbral:
        return "ALTO", "RECHAZAR"
    if probabilidad_mora >= umbral * 0.7:
        return "MEDIO", "REVISAR"
    return "BAJO", "APROBAR"


def _ahora() -> str:
    """Marca temporal en UTC ISO-8601 (trazabilidad de cada respuesta)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def predecir(df_crudo: pd.DataFrame, umbral: float) -> list[RespuestaPrediccion]:
    """Aplica el pipeline completo sobre datos crudos y arma las respuestas.

    Pasos:
    1. `preparar_features()`: la MISMA función que se usa al entrenar (limpieza de
       `tendencia_ingresos` + ratios derivados) -> evita el training/serving skew.
    2. Reindexado al orden exacto de columnas del entrenamiento: así da igual en
       qué orden lleguen los campos del JSON o del CSV.
    3. `predict_proba` del pipeline: internamente imputa, escala y codifica igual
       que en el entrenamiento, porque el preprocesador viaja dentro del modelo.
    """
    artefacto = obtener_artefacto()
    modelo = artefacto["modelo"]

    X = preparar_features(df_crudo)
    X = X.reindex(columns=artefacto["columnas"])

    # `classes_` indica el orden de las columnas de predict_proba. Se busca la
    # posición de la clase 0 (moroso) en lugar de asumir que es la primera:
    # es la que le interesa al negocio y evita un error silencioso de signo.
    probabilidades = modelo.predict_proba(X)
    col_moroso = list(modelo.classes_).index(0)

    respuestas = []
    for prob in probabilidades[:, col_moroso]:
        prob_mora = float(prob)
        nivel, decision = clasificar_riesgo(prob_mora, umbral)
        respuestas.append(
            RespuestaPrediccion(
                id_solicitud=str(uuid4()),
                probabilidad_mora=round(prob_mora, 6),
                probabilidad_pago=round(1 - prob_mora, 6),
                # La predicción se deriva del umbral de negocio, no del 0.5 fijo
                # que usaría `modelo.predict()`.
                prediccion=0 if prob_mora >= umbral else 1,
                etiqueta="Riesgo de mora" if prob_mora >= umbral else "Paga a tiempo",
                nivel_riesgo=nivel,
                decision_sugerida=decision,
                umbral=umbral,
                version_modelo=artefacto["version"],
                timestamp=_ahora(),
            )
        )
    return respuestas


def armar_respuesta_lote(predicciones: list[RespuestaPrediccion], umbral: float) -> RespuestaLote:
    """Agrupa las predicciones de un lote y calcula el resumen agregado."""
    niveles = [p.nivel_riesgo for p in predicciones]
    promedio = sum(p.probabilidad_mora for p in predicciones) / len(predicciones)

    return RespuestaLote(
        id_lote=str(uuid4()),
        version_modelo=obtener_artefacto()["version"],
        umbral=umbral,
        resumen=ResumenLote(
            total_registros=len(predicciones),
            riesgo_alto=niveles.count("ALTO"),
            riesgo_medio=niveles.count("MEDIO"),
            riesgo_bajo=niveles.count("BAJO"),
            probabilidad_mora_promedio=round(promedio, 6),
        ),
        predicciones=predicciones,
        timestamp=_ahora(),
    )


# --- Endpoints de estado -----------------------------------------------------
@app.get("/", tags=["Estado"], summary="Información del servicio")
def raiz() -> dict:
    """Presenta el servicio y sus rutas principales."""
    return {
        "servicio": "API de Riesgo Crediticio",
        "version_api": app.version,
        "documentacion": "/docs",
        "endpoints": {
            "estado": "/health",
            "modelo": "/modelo",
            "prediccion_individual": "POST /predict",
            "prediccion_por_lotes": "POST /predict/batch",
            "prediccion_desde_csv": "POST /predict/csv",
        },
    }


@app.get("/health", tags=["Estado"], summary="Estado del servicio")
def health(response: Response) -> dict:
    """Chequeo de salud para Docker y orquestadores.

    Devuelve 200 si el modelo está cargado y 503 si la API está degradada
    (levantada pero sin modelo), para que el orquestador no le mande tráfico.
    """
    cargado = ARTEFACTO is not None
    if not cargado:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "estado": "ok" if cargado else "degradado",
        "modelo_cargado": cargado,
        "version_modelo": ARTEFACTO["version"] if cargado else None,
        "timestamp": _ahora(),
    }


@app.get("/modelo", tags=["Estado"], summary="Metadata del modelo en producción")
def info_modelo() -> dict:
    """Devuelve qué modelo está sirviendo, cuándo se entrenó y con qué métricas.

    Es la contracara de la trazabilidad: permite auditar una predicción pasada
    sabiendo exactamente qué versión del modelo la produjo.
    """
    artefacto = obtener_artefacto()
    return {
        "nombre": artefacto["nombre"],
        "version": artefacto["version"],
        "fecha_entrenamiento": artefacto["fecha_entrenamiento"],
        "metricas_test": artefacto["metricas"],
        "entorno_entrenamiento": artefacto["entorno"],
        "columnas_modelo": artefacto["columnas"],
        "campos_requeridos": COLUMNAS_REQUERIDAS,
        "umbral_por_defecto": UMBRAL_DEFECTO,
    }


# --- Endpoints de predicción -------------------------------------------------
@app.post(
    "/predict",
    tags=["Predicción"],
    response_model=RespuestaPrediccion,
    summary="Predicción individual",
)
def predict(
    solicitud: SolicitudCredito,
    umbral: float = Query(UMBRAL_DEFECTO, ge=0.01, le=0.99,
                          description="Probabilidad de mora a partir de la cual se rechaza"),
) -> RespuestaPrediccion:
    """Evalúa una solicitud de crédito y devuelve su probabilidad de mora.

    Pensado para el consumo online desde los formularios digitales: un registro
    por llamada, con validación estricta de los datos de entrada.
    """
    try:
        df = pd.DataFrame([solicitud.model_dump()])
        resultado = predecir(df, umbral)[0]
    except HTTPException:
        # Los errores esperados (503 sin modelo) se propagan tal cual.
        raise
    except Exception as exc:  # noqa: BLE001 - se registra y se traduce a 500
        logger.exception("Error al predecir: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al generar la predicción.",
        ) from exc

    logger.info(
        "Predicción %s | prob_mora=%.4f | %s",
        resultado.id_solicitud, resultado.probabilidad_mora, resultado.decision_sugerida,
    )
    return resultado


@app.post(
    "/predict/batch",
    tags=["Predicción"],
    response_model=RespuestaLote,
    summary="Predicción por lotes (JSON)",
)
def predict_batch(
    lote: LoteSolicitudes,
    umbral: float = Query(UMBRAL_DEFECTO, ge=0.01, le=0.99,
                          description="Probabilidad de mora a partir de la cual se rechaza"),
) -> RespuestaLote:
    """Evalúa varias solicitudes en una sola llamada.

    Es mucho más eficiente que llamar N veces a `/predict`: se paga una sola vez
    el costo de red y el modelo vectoriza el cálculo sobre todo el lote.
    """
    try:
        df = pd.DataFrame([s.model_dump() for s in lote.solicitudes])
        predicciones = predecir(df, umbral)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error al predecir el lote: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al generar las predicciones.",
        ) from exc

    respuesta = armar_respuesta_lote(predicciones, umbral)
    logger.info(
        "Lote %s | %d registros | %d de riesgo alto",
        respuesta.id_lote, respuesta.resumen.total_registros, respuesta.resumen.riesgo_alto,
    )
    return respuesta


@app.post(
    "/predict/csv",
    tags=["Predicción"],
    summary="Predicción por lotes (archivo CSV)",
)
def predict_csv(
    archivo: UploadFile = File(..., description="CSV con una solicitud por fila"),
    umbral: float = Query(UMBRAL_DEFECTO, ge=0.01, le=0.99),
    formato: Literal["json", "csv"] = Query("json", description="Formato de la respuesta"),
):
    """Puntúa un archivo CSV completo (proceso batch).

    A diferencia de `/predict`, acá **no** se aplican las validaciones de rango de
    Pydantic: el CSV viene del ETL interno, no de un usuario final, y forzar los
    rangos haría fallar el lote entero por unos pocos registros atípicos (el
    dataset histórico tiene, por ejemplo, 150 clientes con edad > 100). Lo que sí
    se valida es el **esquema**: que estén todas las columnas que el modelo
    espera. Los valores atípicos ya los resuelve el pipeline (imputación).

    Con `formato=csv` devuelve el archivo original más las columnas de predicción,
    listo para descargar.
    """
    if not archivo.filename or not archivo.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo debe tener extensión .csv",
        )

    try:
        contenido = archivo.file.read()
        df = pd.read_csv(io.BytesIO(contenido))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No se pudo leer el CSV: {exc}",
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo CSV está vacío.",
        )

    if len(df) > MAX_REGISTROS_LOTE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"El lote supera el máximo de {MAX_REGISTROS_LOTE} registros.",
        )

    # Validación de esquema: se avisa exactamente qué falta, en vez de fallar
    # más adelante con un error críptico de sklearn.
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Faltan columnas en el CSV: {', '.join(faltantes)}",
        )

    try:
        predicciones = predecir(df, umbral)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error al predecir el CSV: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al generar las predicciones.",
        ) from exc

    respuesta = armar_respuesta_lote(predicciones, umbral)
    logger.info(
        "CSV '%s' | %d registros | %d de riesgo alto",
        archivo.filename, respuesta.resumen.total_registros, respuesta.resumen.riesgo_alto,
    )

    if formato == "csv":
        # Se devuelven las columnas originales + el resultado, para que el área
        # de riesgo pueda abrirlo directamente en una planilla.
        salida = df.copy()
        salida["probabilidad_mora"] = [p.probabilidad_mora for p in predicciones]
        salida["prediccion"] = [p.prediccion for p in predicciones]
        salida["nivel_riesgo"] = [p.nivel_riesgo for p in predicciones]
        salida["decision_sugerida"] = [p.decision_sugerida for p in predicciones]
        return Response(
            content=salida.to_csv(index=False),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="predicciones.csv"'},
        )

    return respuesta


if __name__ == "__main__":
    # Atajo para desarrollo local: `python model_deploy.py`.
    # En producción / Docker el servidor lo levanta Uvicorn directamente.
    import uvicorn

    uvicorn.run("model_deploy:app", host="127.0.0.1", port=8000, reload=True)
