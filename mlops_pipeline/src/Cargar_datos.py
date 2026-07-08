"""
Carga de datos del proyecto de riesgo crediticio.

Este módulo simula la ingesta de datos. En un entorno productivo, los datos
vendrían del Data Warehouse / Datalake de la empresa (resultado de un proceso
de ETL). Para este caso de ejemplo, se carga un dataset no productivo en .csv.
"""

import pandas as pd


def cargar_datos(ruta: str = "../../Base_de_datos.csv") -> pd.DataFrame:
    """Carga el dataset de créditos desde un CSV y lo devuelve como DataFrame.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo .csv con los datos.

    Retorna
    -------
    pd.DataFrame
        Tabla con los datos cargados.
    """
    df = pd.read_csv(ruta)
    return df


if __name__ == "__main__":
    # Prueba rápida: cargar y verificar que la data se leyó bien
    df = cargar_datos()
    print("Dimensiones:", df.shape)
    print(df.head())
    df.info()
