"""
Aplicación de modelado del crecimiento de una planta de fríjol con Streamlit
Versión extendida: incluye interpolación automática según el modelo, métricas adicionales,
estadísticas descriptivas, correlaciones y cálculo de intervalos de confianza.
Todas las métricas se calculan "manualmente" (usando numpy para operaciones) cuando es posible
y se muestran tanto en la interfaz como en el reporte PDF.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from fpdf import FPDF
from scipy.optimize import curve_fit
from scipy import stats
from scipy.interpolate import interp1d
from typing import Tuple

# ==========================
# Constantes globales
# ==========================
CSV_SEPARATOR = ";"
TEMP_GRAPH_FILE = "grafico.png"
PDF_REPORT_FILE = "reporte_frijol_extendido.pdf"

# ==========================
# Funciones auxiliares (estadísticas manuales)
# ==========================

def mean_manual(arr: np.ndarray) -> float:
    return float(np.sum(arr) / arr.size)


def var_manual(arr: np.ndarray, ddof=1) -> float:
    m = mean_manual(arr)
    return float(np.sum((arr - m) ** 2) / (arr.size - ddof))


def std_manual(arr: np.ndarray, ddof=1) -> float:
    return float(np.sqrt(var_manual(arr, ddof=ddof)))


def median_quartiles(arr: np.ndarray):
    a = np.sort(arr)
    med = np.median(a)
    q1 = np.percentile(a, 25)
    q3 = np.percentile(a, 75)
    return med, q1, q3


def coef_variacion(arr: np.ndarray):
    m = mean_manual(arr)
    s = std_manual(arr)
    return float(s / m) if m != 0 else np.nan


def covariance_manual(x: np.ndarray, y: np.ndarray):
    # Covarianza muestral con n-1
    xm = mean_manual(x)
    ym = mean_manual(y)
    return float(np.sum((x - xm) * (y - ym)) / (x.size - 1))


def pearson_manual(x: np.ndarray, y: np.ndarray):
    cov = covariance_manual(x, y)
    sx = std_manual(x)
    sy = std_manual(y)
    return float(cov / (sx * sy))


def rmse_manual(y_true: np.ndarray, y_pred: np.ndarray):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_manual(y_true: np.ndarray, y_pred: np.ndarray):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - mean_manual(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan


def mae_manual(y_true: np.ndarray, y_pred: np.ndarray):
    return float(np.mean(np.abs(y_true - y_pred)))


def mape_manual(y_true: np.ndarray, y_pred: np.ndarray):
    # Evitar división por cero
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def confidence_interval_mean(arr: np.ndarray, alpha=0.05):
    # Intervalo de confianza para la media (t-student)
    n = arr.size
    m = mean_manual(arr)
    s = std_manual(arr, ddof=1)
    se = s / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    lo = m - tcrit * se
    hi = m + tcrit * se
    return float(lo), float(hi), float(m), float(se), int(n)

# ==========================
# Carga de datos
# ==========================

def load_data(file) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    data = pd.read_csv(file, sep=CSV_SEPARATOR)
    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.replace("\ufeff", "", regex=True)

    if "Dia" in data.columns and "Altura_cm" in data.columns:
        X = data["Dia"].values.reshape(-1, 1).astype(float)
        y = data["Altura_cm"].astype(str).str.replace(",", ".").astype(float).values
    else:
        X = data.iloc[:, 0].values.reshape(-1, 1).astype(float)
        y = data.iloc[:, 1].astype(str).str.replace(",", ".").astype(float).values

    return X, y, data

# ==========================
# Modelos existentes (mejorados)
# ==========================

def train_lineal(X, y):
    lin = LinearRegression()
    lin.fit(X, y)
    y_pred = lin.predict(X)
    texto = f"Modelo Lineal:\nAltura = {lin.coef_[0]:.6f}*Dia + {lin.intercept_:.6f}"
    return y_pred, texto, lin


def train_polinomico(X, y, grado: int):
    poly = PolynomialFeatures(degree=grado)
    X_poly = poly.fit_transform(X)
    lin = LinearRegression()
    lin.fit(X_poly, y)
    y_pred = lin.predict(X_poly)
    coefs = lin.coef_
    intercept = lin.intercept_
    eq = " + ".join([f"{coefs[i]:.6f}*x^{i}" for i in range(len(coefs))])
    texto = f"Modelo Polinómico grado {grado}:\nAltura = {eq} + {intercept:.6f}"
    return y_pred, texto, (lin, poly)


def train_exponencial(X, y):
    Xf = X.flatten()
    # Ajuste ln(y) = ln(a) + b*x
    Y = np.log(y + 1e-9)
    lin = LinearRegression()
    lin.fit(Xf.reshape(-1, 1), Y)
    a = np.exp(lin.intercept_)
    b = lin.coef_[0]
    y_pred = a * np.exp(b * Xf)
    texto = f"Modelo Exponencial:\nAltura = {a:.6f} * exp({b:.6f}*Dia)"
    return y_pred, texto, (a, b)


def train_logistico(X, y):
    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    Xf = X.flatten()
    try:
        popt, _ = curve_fit(logistic, Xf, y, maxfev=10000)
        L, k, x0 = popt
        y_pred = logistic(Xf, L, k, x0)
        texto = f"Modelo Logístico:\nAltura = {L:.6f} / (1+exp(-{k:.6f}*(Dia-{x0:.6f})))"
        return y_pred, texto, popt
    except Exception as e:
        # Devolver ceros si falla
        y_pred = np.zeros_like(y)
        texto = "No se pudo ajustar modelo logístico (datos insuficientes o mala inicialización)."
        return y_pred, texto, None

# ==========================
# Interpolación según modelo
# ==========================

def interpolate_by_model(X, y, model_name: str, grado: int = 1):
    # X: Nx1 array, y: N array
    xf = X.flatten()
    # Ordenar por X para interpolación estable
    order = np.argsort(xf)
    xf_s = xf[order]
    y_s = y[order]

    if model_name in ["Lineal"]:
        # Ajuste polinómico grado 1 para interpolación
        coefs = np.polyfit(xf_s, y_s, 1)
        poly = np.poly1d(coefs)
        y_interp = poly(xf_s)
        kind = f"Polinomial grado 1 (lineal)"
    elif model_name == "Polinómico":
        # usar grado indicado
        coefs = np.polyfit(xf_s, y_s, grado)
        poly = np.poly1d(coefs)
        y_interp = poly(xf_s)
        kind = f"Polinomial grado {grado}"
    elif model_name == "Exponencial":
        # interpolación en espacio log y luego exponencial
        # Ajustar log(y) con polinomio lineal
        Ylog = np.log(y_s + 1e-9)
        coefs = np.polyfit(xf_s, Ylog, 1)
        poly = np.poly1d(coefs)
        y_interp = np.exp(poly(xf_s))
        kind = "Interpolación exponencial (log-espacio)"
    elif model_name == "Logístico":
        # usar interpolación spline cúbica para suavizar
        try:
            f = interp1d(xf_s, y_s, kind='cubic', fill_value='extrapolate')
            y_interp = f(xf_s)
            kind = "Spline cúbica (logístico)"
        except Exception:
            f = interp1d(xf_s, y_s, kind='linear', fill_value='extrapolate')
            y_interp = f(xf_s)
            kind = "Interpolación lineal (fallback)"
    else:
        # fallback: lineal
        f = interp1d(xf_s, y_s, kind='linear', fill_value='extrapolate')
        y_interp = f(xf_s)
        kind = "Interpolación lineal (default)"

    # devolver en el orden original
    y_interp_full = np.empty_like(y)
    y_interp_full[order] = y_interp
    return y_interp_full, kind

# ==========================
# Cálculo de métricas y estadísticas (agrupadas)
# ==========================

def calculate_all_metrics(y, y_pred):
    metrics = {}
    metrics['MSE'] = float(np.mean((y - y_pred) ** 2))
    metrics['RMSE'] = rmse_manual(y, y_pred)
    metrics['MAE'] = mae_manual(y, y_pred)
    metrics['MAPE'] = mape_manual(y, y_pred)
    metrics['R2'] = r2_manual(y, y_pred)
    metrics['Bias'] = float(np.mean(y_pred - y))
    metrics['Var_y'] = float(var_manual(y))
    metrics['Var_pred'] = float(var_manual(y_pred))
    metrics['Covariance'] = float(covariance_manual(y, y_pred))
    metrics['Pearson'] = float(pearson_manual(y, y_pred))
    # Spearman y Pearson usando scipy para consistencia en p-valor
    try:
        metrics['Pearson_scipy'] = stats.pearsonr(y, y_pred)[0]
    except Exception:
        metrics['Pearson_scipy'] = np.nan
    try:
        metrics['Spearman'] = stats.spearmanr(y, y_pred).correlation
    except Exception:
        metrics['Spearman'] = np.nan
    return metrics


def calculate_descriptives(arr: np.ndarray):
    desc = {}
    desc['mean'] = mean_manual(arr)
    desc['var'] = var_manual(arr)
    desc['std'] = std_manual(arr)
    desc['cv'] = coef_variacion(arr)
    med, q1, q3 = median_quartiles(arr)
    desc['median'] = med
    desc['q1'] = q1
    desc['q3'] = q3
    desc['ci_mean'] = confidence_interval_mean(arr)
    return desc

# ==========================
# Plot y reporte PDF extendido
# ==========================

def plot_results(X, y, y_pred, modelo: str, interpolacion=None):
    plt.figure()
    plt.scatter(X, y, label="Datos Reales")
    plt.plot(X, y_pred, label=f"{modelo} Predicción")
    if interpolacion is not None:
        plt.plot(X, interpolacion, '--', label="Interpolación")
    plt.xlabel("Día")
    plt.ylabel("Altura (cm)")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)


def export_pdf(enunciado: str, resultados: str, X, y, y_pred, modelo: str, data: pd.DataFrame,
               metrics: dict, descriptives_y: dict, descriptives_pred: dict, interp_kind: str):
    # Guardar gráfica
    plt.figure()
    plt.scatter(X, y, label="Datos Reales")
    plt.plot(X, y_pred, color="red", label=modelo)
    plt.xlabel("Día")
    plt.ylabel("Altura (cm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(TEMP_GRAPH_FILE)
    plt.close()

    # Crear PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Título
    pdf.multi_cell(0, 10, "Reporte de Modelado del Crecimiento del Fríjol (Extendido)", align="C")
    pdf.ln(6)

    # Enunciado
    pdf.multi_cell(0, 8, "Enunciado del ejercicio:\n" + enunciado)
    pdf.ln(4)

    # Datos cargados
    pdf.multi_cell(0, 8, "Datos cargados (primeras filas):")
    head_data = data.head(10)
    for _, row in head_data.iterrows():
        pdf.cell(0, 8, f"Dia: {row[0]}   Altura_cm: {row[1]}", ln=True)
    pdf.ln(6)

    # Pasos
    pdf.multi_cell(
        0, 8,
        f"Pasos de la solución:\n"
        f"1. Se seleccionó el modelo: {modelo}.\n"
        f"2. Se entrenó el modelo con los datos cargados.\n"
        f"3. Se calcularon los parámetros y múltiples métricas de error/descriptivas.\n"
        f"4. Se generó la gráfica comparando datos reales con predicciones y la interpolación ({interp_kind}).\n"
    )
    pdf.ln(6)

    # Resultados (texto original del modelo)
    pdf.multi_cell(0, 8, "Resultados (modelo):\n" + resultados)
    pdf.ln(4)

    # Métricas
    pdf.multi_cell(0, 8, "Métricas calculadas:")
    for k, v in metrics.items():
        pdf.cell(0, 7, f"{k}: {v}", ln=True)
    pdf.ln(4)

    # Descriptivas
    pdf.multi_cell(0, 8, "Estadísticas descriptivas (Y real):")
    pdf.cell(0, 7, f"Media: {descriptives_y['mean']}", ln=True)
    pdf.cell(0, 7, f"Varianza: {descriptives_y['var']}", ln=True)
    pdf.cell(0, 7, f"Desviación estándar: {descriptives_y['std']}", ln=True)
    pdf.cell(0, 7, f"Coef. variación: {descriptives_y['cv']}", ln=True)
    pdf.cell(0, 7, f"Mediana: {descriptives_y['median']}", ln=True)
    pdf.cell(0, 7, f"Q1: {descriptives_y['q1']}  Q3: {descriptives_y['q3']}", ln=True)
    lo, hi, m, se, n = descriptives_y['ci_mean']
    pdf.cell(0, 7, f"IC media (95%): [{lo:.6f}, {hi:.6f}] (n={n}, se={se:.6f})", ln=True)
    pdf.ln(4)

    pdf.multi_cell(0, 8, "Estadísticas descriptivas (Predicción):")
    pdf.cell(0, 7, f"Media: {descriptives_pred['mean']}", ln=True)
    pdf.cell(0, 7, f"Varianza: {descriptives_pred['var']}", ln=True)
    pdf.cell(0, 7, f"Desviación estándar: {descriptives_pred['std']}", ln=True)
    pdf.cell(0, 7, f"Coef. variación: {descriptives_pred['cv']}", ln=True)
    pdf.cell(0, 7, f"Mediana: {descriptives_pred['median']}", ln=True)
    pdf.cell(0, 7, f"Q1: {descriptives_pred['q1']}  Q3: {descriptives_pred['q3']}", ln=True)
    lo2, hi2, m2, se2, n2 = descriptives_pred['ci_mean']
    pdf.cell(0, 7, f"IC media (95%): [{lo2:.6f}, {hi2:.6f}] (n={n2}, se={se2:.6f})", ln=True)
    pdf.ln(6)

    # Gráfica
    pdf.image(TEMP_GRAPH_FILE, x=20, w=170)

    # Guardar
    pdf.output(PDF_REPORT_FILE)
    return PDF_REPORT_FILE

# ==========================
# Interfaz Streamlit
# ==========================

def main():
    st.title("Modelado del Crecimiento del Fríjol - Versión Extendida")
    st.write("Cargue datos en CSV y seleccione un modelo para analizar el crecimiento. Todos los cálculos (métricas, descriptivas, correlaciones e intervalos) se muestran y se exportan al PDF.")

    # Cargar archivo
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
    if uploaded_file is not None:
        X, y, data = load_data(uploaded_file)
        st.success("Datos cargados correctamente")
        st.dataframe(data.head())

        # Enunciado del ejercicio
        enunciado = st.text_area("Enunciado del ejercicio:", 
                                 "Analizar el crecimiento de la planta de fríjol usando los datos experimentales.")

        # Selección de modelo
        modelo = st.selectbox("Seleccione modelo:", ["Lineal", "Polinómico", "Exponencial", "Logístico"])
        grado = 1
        if modelo == "Polinómico":
            grado = st.slider("Seleccione grado del polinomio:", 1, 10, 2)

        # Entrenar modelo
        if st.button("Entrenar Modelo"):
            if modelo == "Lineal":
                y_pred, texto, params = train_lineal(X, y)
            elif modelo == "Polinómico":
                y_pred, texto, params = train_polinomico(X, y, grado)
            elif modelo == "Exponencial":
                y_pred, texto, params = train_exponencial(X, y)
            elif modelo == "Logístico":
                y_pred, texto, params = train_logistico(X, y)

            # Interpolación adaptativa según modelo
            y_interp, interp_kind = interpolate_by_model(X, y, modelo, grado)

            # Métricas y descriptivas
            metrics = calculate_all_metrics(y, y_pred)
            descriptives_y = calculate_descriptives(y)
            descriptives_pred = calculate_descriptives(y_pred)

            # Mostrar resultados en la interfaz
            st.text_area("Resultados (modelo)", texto, height=140)

            st.subheader("Métricas de error")
            for k, v in metrics.items():
                st.write(f"{k}: {v}")

            st.subheader("Estadísticas descriptivas (Y real)")
            for k, v in descriptives_y.items():
                st.write(f"{k}: {v}")

            st.subheader("Estadísticas descriptivas (Predicción)")
            for k, v in descriptives_pred.items():
                st.write(f"{k}: {v}")

            st.subheader("Interpolación aplicada")
            st.write(f"Tipo de interpolación aplicada: {interp_kind}")

            # Plot
            plot_results(X, y, y_pred, modelo, interpolacion=y_interp)

            # Exportar PDF
            pdf_path = export_pdf(enunciado, texto, X, y, y_pred, modelo, data, metrics, descriptives_y, descriptives_pred, interp_kind)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("Descargar PDF", data=pdf_file, file_name=PDF_REPORT_FILE, mime="application/pdf")


if __name__ == "__main__":
    main()
