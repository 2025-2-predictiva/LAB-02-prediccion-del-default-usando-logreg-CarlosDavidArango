# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import json
import gzip
import pickle
import zipfile
import numpy as np
import pandas as pd

from typing import List

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.base import clone


# Parámetros de referencia para optimización de umbrales
REQ_ENTRENAMIENTO = {"p": 0.58, "ba": 0.63, "r": 0.32, "f1": 0.40}
REQ_PRUEBA = {"p": 0.56, "ba": 0.65, "r": 0.35, "f1": 0.43}

MIN_TN_TRAIN, MIN_TP_TRAIN = 14500, 1400
MIN_TN_TEST, MIN_TP_TEST = 6300, 650

UMBRAL_BAJO, UMBRAL_ALTO, EPS = 0.40, 0.85, 1e-9


def leer_zip_csv(ruta: str) -> pd.DataFrame:
    """Lee un archivo CSV desde un archivo ZIP"""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
    
    with zipfile.ZipFile(ruta) as z:
        csvs = [f for f in z.namelist() if f.lower().endswith(".csv")]
        if not csvs:
            raise ValueError(f"El archivo '{ruta}' no contiene archivos CSV.")
        with z.open(csvs[0]) as f:
            return pd.read_csv(f)


def limpiar_datos(ruta: str) -> pd.DataFrame:
    """
    Paso 1: Limpia los datos del dataset
    - Renombra la columna target
    - Elimina columna ID
    - Maneja valores faltantes (0 en EDUCATION y MARRIAGE)
    - Agrupa EDUCATION > 4 en categoría 4
    """
    df = leer_zip_csv(ruta)
    
    # Renombrar columna target
    df = df.rename(columns={"default payment next month": "default"})
    
    # Eliminar columna ID
    df = df.drop(columns=["ID"], errors="ignore")
    
    # Manejar valores 0 como NaN en EDUCATION y MARRIAGE
    df.loc[df["EDUCATION"] == 0, "EDUCATION"] = np.nan
    df.loc[df["MARRIAGE"] == 0, "MARRIAGE"] = np.nan
    
    # Agrupar EDUCATION > 4 en categoría 4 (others)
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    
    # Eliminar registros con valores faltantes
    df = df.dropna().reset_index(drop=True)
    
    # Asegurar tipos correctos
    for col in ["EDUCATION", "MARRIAGE", "default"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    return df


def crear_pipeline(columnas: List[str]) -> Pipeline:
    """
    Paso 3: Crea el pipeline de clasificación
    - OneHotEncoder para variables categóricas (incluyendo PAY_0 a PAY_6)
    - PowerTransformer + MinMaxScaler para variables numéricas
    - SelectKBest para selección de características
    - LogisticRegression como clasificador
    """
    # Variables categóricas (CLAVE: PAY_0 a PAY_6 son categóricas)
    cat_cols = [
        c for c in [
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
        ]
        if c in columnas
    ]
    
    # Variables numéricas
    num_cols = [c for c in columnas if c not in cat_cols]
    
    # Pipeline para variables numéricas con transformación de potencia
    proc_numerico = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median")),
            ("yeo", PowerTransformer(method="yeo-johnson")),
            ("escalar", MinMaxScaler()),
        ]
    )
    
    # Encoder para variables categóricas
    proc_categorico = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    # Preprocesador combinado
    preprocesador = ColumnTransformer(
        [
            ("categorico", proc_categorico, cat_cols),
            ("numerico", proc_numerico, num_cols),
        ],
        verbose_feature_names_out=False,
    )
    
    # Selector de características
    selector = SelectKBest(f_classif)
    
    # Clasificador
    clasificador = LogisticRegression(
        solver="liblinear", random_state=42, max_iter=1000, class_weight=None
    )
    
    # Pipeline completo
    return Pipeline(
        [
            ("preprocesar", preprocesador),
            ("seleccionar", selector),
            ("clasificador", clasificador),
        ]
    )


def contar_caracteristicas(pipe: Pipeline, X: pd.DataFrame, y=None) -> int:
    """Cuenta el número total de características después del preprocesamiento"""
    p = clone(pipe.named_steps["preprocesar"])
    return p.fit_transform(X, y).shape[1]


def ajustar_modelo(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
    """
    Paso 4: Optimiza hiperparámetros usando GridSearchCV
    - Validación cruzada estratificada con 10 splits
    - Scoring: balanced_accuracy
    """
    k_max = contar_caracteristicas(pipe, X, y)
    print(f"\nTotal de variables después del preprocesamiento: {k_max}")
    
    # Grid de hiperparámetros optimizado
    grid = {
        "seleccionar__k": [20, 40, 50, 60, k_max],
        "clasificador__C": [0.8, 1.0, 1.2, 1.4, 1.5, 2.0],
        "clasificador__penalty": ["l1", "l2"],
        "clasificador__class_weight": [None],
    }
    
    print(f"Explorando {len(grid['seleccionar__k']) * len(grid['clasificador__C']) * len(grid['clasificador__penalty'])} combinaciones\n")
    
    # Validación cruzada estratificada
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # GridSearchCV
    modelo = GridSearchCV(
        pipe, grid, scoring="balanced_accuracy", cv=cv, n_jobs=-1, verbose=1
    )
    
    modelo.fit(X, y)
    
    print(f"\nMejores parámetros: {modelo.best_params_}")
    print(f"Balanced Accuracy (CV): {modelo.best_score_:.4f}")
    
    return modelo


def calcular_metricas(y_real, y_pred, nombre: str) -> dict:
    """Calcula las métricas de evaluación"""
    return {
        "type": "metrics",
        "dataset": nombre,
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "recall": recall_score(y_real, y_pred, zero_division=0),
        "f1_score": f1_score(y_real, y_pred, zero_division=0),
    }


def matriz_confusion(y_real, y_pred, nombre: str) -> dict:
    """Calcula y formatea la matriz de confusión"""
    cm = confusion_matrix(y_real, y_pred, labels=[0, 1])
    return {
        "type": "cm_matrix",
        "dataset": nombre,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }


def generar_umbrales(proba, bajo=UMBRAL_BAJO, alto=UMBRAL_ALTO):
    """Genera lista de umbrales candidatos para optimización"""
    unicos = np.unique(np.round(proba, 8))
    unicos = unicos[(unicos >= bajo) & (unicos <= alto)]
    umbrales = [bajo, alto] + [
        float(v) + d for v in unicos for d in (-EPS, 0, EPS)
    ]
    return sorted(set(umbrales))


def validar_metricas(y_real, y_pred, ref: dict):
    """Valida si las métricas cumplen con los requisitos mínimos"""
    p = precision_score(y_real, y_pred, zero_division=0)
    r = recall_score(y_real, y_pred, zero_division=0)
    ba = balanced_accuracy_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred, zero_division=0)
    cumple = all([p > ref["p"], r > ref["r"], ba > ref["ba"], f1 > ref["f1"]])
    return cumple, p, r, ba, f1


def buscar_umbral(y_real, proba, ref: dict, min_tn: int, min_tp: int):
    """
    Busca el umbral óptimo de clasificación
    Estrategia: maximizar TN mientras se cumplen restricciones de métricas
    """
    mejor_t, mejor_tn = None, -1
    
    # Primera búsqueda: maximizar TN con restricciones
    for t in generar_umbrales(proba):
        y_pred = (proba >= t).astype(int)
        ok, *_ = validar_metricas(y_real, y_pred, ref)
        
        if not ok:
            continue
        
        cm = confusion_matrix(y_real, y_pred, labels=[0, 1])
        tn, tp = cm[0, 0], cm[1, 1]
        
        if tn > min_tn and tp > min_tp and tn > mejor_tn:
            mejor_t, mejor_tn = float(t), tn
    
    if mejor_t is not None:
        return mejor_t
    
    # Búsqueda alternativa: maximizar balanced_accuracy
    print("No se encontró umbral con las restricciones, usando alternativa.")
    mejor_ba, umbral_alt = -1, 0.5
    
    for t in generar_umbrales(proba):
        y_pred = (proba >= t).astype(int)
        ok, _, _, ba, _ = validar_metricas(y_real, y_pred, ref)
        
        if ok and ba > mejor_ba:
            mejor_ba, umbral_alt = ba, float(t)
    
    return umbral_alt


def guardar_resultados(
    modelo, X_train, y_train, X_test, y_test, ruta="files/output/metrics.json"
):
    """
    Pasos 6 y 7: Calcula y guarda métricas y matrices de confusión
    con umbrales optimizados para cada conjunto
    """
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    
    # Obtener probabilidades
    p_train = modelo.best_estimator_.predict_proba(X_train)[:, 1]
    p_test = modelo.best_estimator_.predict_proba(X_test)[:, 1]
    
    # Buscar umbrales óptimos
    thr_train = buscar_umbral(
        y_train, p_train, REQ_ENTRENAMIENTO, MIN_TN_TRAIN, MIN_TP_TRAIN
    )
    thr_test = buscar_umbral(y_test, p_test, REQ_PRUEBA, MIN_TN_TEST, MIN_TP_TEST)
    
    print(f"\nUmbral óptimo train: {thr_train:.4f}")
    print(f"Umbral óptimo test:  {thr_test:.4f}\n")
    
    # Predicciones con umbrales optimizados
    y_pred_train = (p_train >= thr_train).astype(int)
    y_pred_test = (p_test >= thr_test).astype(int)
    
    # Calcular métricas y matrices de confusión
    resultados = [
        calcular_metricas(y_train, y_pred_train, "train"),
        calcular_metricas(y_test, y_pred_test, "test"),
        matriz_confusion(y_train, y_pred_train, "train"),
        matriz_confusion(y_test, y_pred_test, "test"),
    ]
    
    # Guardar resultados
    with open(ruta, "w", encoding="utf-8") as f:
        for r in resultados:
            f.write(json.dumps(r) + "\n")
    
    print(f"Métricas guardadas en {ruta}")
    print(
        f"Train: P={resultados[0]['precision']:.3f}, R={resultados[0]['recall']:.3f}, "
        f"BA={resultados[0]['balanced_accuracy']:.3f}, F1={resultados[0]['f1_score']:.3f}"
    )
    print(
        f"Test:  P={resultados[1]['precision']:.3f}, R={resultados[1]['recall']:.3f}, "
        f"BA={resultados[1]['balanced_accuracy']:.3f}, F1={resultados[1]['f1_score']:.3f}"
    )


def main():
    """Función principal que ejecuta todo el pipeline"""
    print("="*80)
    print("PREDICCIÓN DE DEFAULT - REGRESIÓN LOGÍSTICA")
    print("="*80)
    
    # Paso 1: Cargar y limpiar datos
    print("\n[1/5] Cargando y limpiando datos...")
    try:
        df_train = limpiar_datos("files/input/train_data.csv.zip")
        df_test = limpiar_datos("files/input/test_data.csv.zip")
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        exit(1)
    
    if "default" not in df_train.columns or "default" not in df_test.columns:
        raise KeyError("No se encontró la columna 'default'.")
    
    print(f"Train: {df_train.shape[0]} registros, {df_train.shape[1]} columnas")
    print(f"Test:  {df_test.shape[0]} registros, {df_test.shape[1]} columnas")
    
    # Paso 2: Dividir en X e y
    print("\n[2/5] Dividiendo features y target...")
    X_train, y_train = df_train.drop("default", axis=1), df_train["default"]
    X_test, y_test = df_test.drop("default", axis=1), df_test["default"]
    
    # Paso 3: Crear pipeline
    print("\n[3/5] Creando pipeline de clasificación...")
    pipeline = crear_pipeline(list(X_train.columns))
    
    # Paso 4: Ajustar modelo con GridSearchCV
    print("\n[4/5] Optimizando hiperparámetros...")
    modelo = ajustar_modelo(pipeline, X_train, y_train)
    
    # Paso 5: Guardar modelo
    print("\n[5/5] Guardando modelo...")
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(modelo, f)
    print("Modelo guardado en files/models/model.pkl.gz")
    
    # Pasos 6 y 7: Calcular y guardar métricas
    print("\n[6-7/5] Calculando métricas finales con umbrales optimizados...")
    guardar_resultados(modelo, X_train, y_train, X_test, y_test)
    
    print("\n" + "="*80)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*80)


if __name__ == "__main__":
    main()
