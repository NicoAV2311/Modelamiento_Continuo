"""
Aplicación de modelado del crecimiento de una planta de fríjol con Streamlit.
Permite cargar datos desde CSV, aplicar diferentes modelos de regresión,
visualizar resultados y exportar un informe en PDF con la gráfica.
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
from typing import Tuple

# ==========================
# Constantes globales
# ==========================
CSV_SEPARATOR = ";"
TEMP_GRAPH_FILE = "grafico.png"
PDF_REPORT_FILE = "reporte_frijol.pdf"

# ==========================
# Funciones auxiliares
# ==========================
def load_data(file) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Carga los datos desde un archivo CSV y devuelve X, y y DataFrame.
    """
    data = pd.read_csv(file, sep=CSV_SEPARATOR)
    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.replace("\ufeff", "", regex=True)

    if "Dia" in data.columns and "Altura_cm" in data.columns:
        X = data["Dia"].values.reshape(-1, 1)
        y = data["Altura_cm"].astype(str).str.replace(",", ".").astype(float).values
    else:
        X = data.iloc[:, 0].values.reshape(-1, 1)
        y = data.iloc[:, 1].astype(str).str.replace(",", ".").astype(float).values

    return X, y, data


def train_lineal(X, y):
    lin = LinearRegression()
    lin.fit(X, y)
    y_pred = lin.predict(X)
    error = mean_squared_error(y, y_pred)
    texto = f"Modelo Lineal:\nAltura = {lin.coef_[0]:.3f}*Dia + {lin.intercept_:.3f}\nError MSE: {error:.3f}"
    return y_pred, texto


def train_polinomico(X, y, grado: int):
    poly = PolynomialFeatures(degree=grado)
    X_poly = poly.fit_transform(X)
    lin = LinearRegression()
    lin.fit(X_poly, y)
    y_pred = lin.predict(X_poly)
    coefs = lin.coef_
    intercept = lin.intercept_
    error = mean_squared_error(y, y_pred)
    eq = " + ".join([f"{coefs[i]:.3f}*x^{i}" for i in range(len(coefs))])
    texto = f"Modelo Polinómico grado {grado}:\nAltura = {eq} + {intercept:.3f}\nError MSE: {error:.3f}"
    return y_pred, texto


def train_exponencial(X, y):
    Xf = X.flatten()
    Y = np.log(y + 1e-5)  # evitar log(0)
    lin = LinearRegression()
    lin.fit(Xf.reshape(-1, 1), Y)
    a = np.exp(lin.intercept_)
    b = lin.coef_[0]
    y_pred = a * np.exp(b * Xf)
    error = mean_squared_error(y, y_pred)
    texto = f"Modelo Exponencial:\nAltura = {a:.3f} * exp({b:.3f}*Dia)\nError MSE: {error:.3f}"
    return y_pred, texto


def train_logistico(X, y):
    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    Xf = X.flatten()
    try:
        popt, _ = curve_fit(logistic, Xf, y, maxfev=5000)
        L, k, x0 = popt
        y_pred = logistic(Xf, L, k, x0)
        error = mean_squared_error(y, y_pred)
        texto = f"Modelo Logístico:\nAltura = {L:.3f} / (1+exp(-{k:.3f}*(Dia-{x0:.3f})))\nError MSE: {error:.3f}"
    except Exception:
        y_pred = np.zeros_like(y)
        texto = "No se pudo ajustar modelo logístico (datos insuficientes)."
    return y_pred, texto


def plot_results(X, y, y_pred, modelo: str):
    plt.figure()
    plt.scatter(X, y, label="Datos Reales")
    plt.plot(X, y_pred, color="red", label=modelo)
    plt.xlabel("Día")
    plt.ylabel("Altura (cm)")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)


def export_pdf(enunciado: str, resultados: str, X, y, y_pred, modelo: str, data: pd.DataFrame):
    """
    Genera un reporte PDF con enunciado, pasos de solución, resultados y gráfica.
    """

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
    pdf.multi_cell(0, 10, "Reporte de Modelado del Crecimiento del Fríjol", align="C")
    pdf.ln(10)

    # Enunciado
    pdf.multi_cell(0, 8, "Enunciado del ejercicio:\n" + enunciado)
    pdf.ln(5)

    # Datos cargados
    pdf.multi_cell(0, 8, "Datos cargados (primeras filas):")
    head_data = data.head(10)
    for _, row in head_data.iterrows():
        pdf.cell(0, 8, f"Dia: {row[0]}   Altura_cm: {row[1]}", ln=True)
    pdf.ln(5)

    # Pasos
    pdf.multi_cell(
        0, 8,
        f"Pasos de la solución:\n"
        f"1. Se seleccionó el modelo: {modelo}.\n"
        f"2. Se entrenó el modelo con los datos cargados.\n"
        f"3. Se calcularon los parámetros y el error MSE.\n"
        f"4. Se generó la gráfica comparando datos reales con predicciones.\n"
    )
    pdf.ln(5)

    # Resultados
    pdf.multi_cell(0, 8, "Resultados:\n" + resultados)
    pdf.ln(10)

    # Gráfica
    pdf.image(TEMP_GRAPH_FILE, x=30, w=150)

    # Guardar
    pdf.output(PDF_REPORT_FILE)
    return PDF_REPORT_FILE


# ==========================
# Interfaz Streamlit
# ==========================
def main():
    st.title("Modelado del Crecimiento del Fríjol")
    st.write("Cargue datos en CSV y seleccione un modelo para analizar el crecimiento.")

    # Cargar archivo
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
    if uploaded_file is not None:
        X, y, data = load_data(uploaded_file)
        st.success("Datos cargados correctamente")
        st.dataframe(data.head())

        # Enunciado del ejercicio (puede ser dinámico o fijo)
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
                y_pred, texto = train_lineal(X, y)
            elif modelo == "Polinómico":
                y_pred, texto = train_polinomico(X, y, grado)
            elif modelo == "Exponencial":
                y_pred, texto = train_exponencial(X, y)
            elif modelo == "Logístico":
                y_pred, texto = train_logistico(X, y)

            st.text_area("Resultados", texto, height=150)
            plot_results(X, y, y_pred, modelo)

            # Exportar PDF
            pdf_path = export_pdf(enunciado, texto, X, y, y_pred, modelo, data)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("Descargar PDF", data=pdf_file, file_name=PDF_REPORT_FILE, mime="application/pdf")


if __name__ == "__main__":
    main()
