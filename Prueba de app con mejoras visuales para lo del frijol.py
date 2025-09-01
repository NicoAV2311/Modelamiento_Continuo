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
PDF_REPORT_FILE = "Reporte Reto 1.pdf"

# ==========================
# Funciones auxiliares (estadísticas manuales)
# ==========================

def mean_manual(arr: np.ndarray) -> float: # Media manual
    return float(np.sum(arr) / arr.size)


def median_quartiles(arr: np.ndarray): # Mediana y cuartiles
    a = np.sort(arr)
    med = np.median(a)
    q1 = np.percentile(a, 25)
    q3 = np.percentile(a, 75)
    return med, q1, q3


def var_manual(arr: np.ndarray, ddof=1) -> float: # Varianza manual
    m = mean_manual(arr)
    return float(np.sum((arr - m) ** 2) / (arr.size - ddof))


def std_manual(arr: np.ndarray, ddof=1) -> float: # Desviación estándar manual
    return float(np.sqrt(var_manual(arr, ddof=ddof)))


def coef_variacion(arr: np.ndarray): # Coeficiente de variación
    m = mean_manual(arr)
    s = std_manual(arr)
    return float(s / m) if m != 0 else np.nan


def covariance_manual(x: np.ndarray, y: np.ndarray): # Covarianza manual
    # Covarianza muestral con n-1
    xm = mean_manual(x)
    ym = mean_manual(y)
    return float(np.sum((x - xm) * (y - ym)) / (x.size - 1))


def rmse_manual(y_true: np.ndarray, y_pred: np.ndarray): # Error cuadrático medio
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_manual(y_true: np.ndarray, y_pred: np.ndarray): # R^2 manual (coeficiente de determinación)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - mean_manual(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan


def confidence_interval_mean(arr: np.ndarray, alpha=0.05): # Intervalo de confianza para la media
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

def load_data(file) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]: # Cargar datos desde CSV
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

def train_lineal(X, y):  # Entrenamiento modelo lineal
    lin = LinearRegression()
    lin.fit(X, y)
    y_pred = lin.predict(X)
    texto = f"Modelo Lineal:\nAltura = {lin.coef_[0]:.6f}*Dia + {lin.intercept_:.6f}"
    return y_pred, texto, lin


def train_polinomico(X, y, grado: int): # Entrenamiento modelo polinómico
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


def train_exponencial(X, y): # Entrenamiento modelo exponencial
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


def train_logistico(X, y): # Entrenamiento modelo logístico
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

def interpolate_by_model(X, y, model_name: str, grado: int = 1): # Interpolación según modelo
    # X: Nx1 array, y: N array
    xf = X.flatten()
    # Ordenar por X para interpolación estable
    order = np.argsort(xf)
    xf_s = xf[order]
    y_s = y[order]

    if model_name in ["Lineal"]: # Interpolación lineal
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

def calculate_all_metrics(y, y_pred): # Cálculo de todas las métricas
    metrics = {}
    metrics['MSE'] = float(np.mean((y - y_pred) ** 2))          # Error cuadrático medio
    metrics['RMSE'] = rmse_manual(y, y_pred)                    # Raíz del error cuadrático medio
    metrics['R2'] = r2_manual(y, y_pred)                        # Coeficiente de determinación
    metrics['Var_y'] = float(var_manual(y))                     # Varianza de y
    metrics['Var_pred'] = float(var_manual(y_pred))             # Varianza de y_pred
    metrics['Covariance'] = float(covariance_manual(y, y_pred)) # Covarianza
    return metrics


def calculate_descriptives(arr: np.ndarray):
    desc = {}
    desc['mean'] = mean_manual(arr)                             # Media
    desc['var'] = var_manual(arr)                               # Varianza
    desc['std'] = std_manual(arr)                               # Desviación estándar
    desc['cv'] = coef_variacion(arr)                            # Coeficiente de variación
    med, q1, q3 = median_quartiles(arr)                         # Mediana y cuartiles
    desc['median'] = med                                        # Mediana
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

    # ==========================
# Gráficas adicionales
# ==========================

def plot_comparison(X, y, y_pred, modelo, interpolacion=None):
    plt.figure()
    plt.scatter(X, y, label="Datos Reales")
    plt.plot(X, y_pred, label=f"{modelo} Predicción", color="red")
    if interpolacion is not None:
        plt.plot(X, interpolacion, '--', label="Interpolación", color="green")
    plt.xlabel("Día")
    plt.ylabel("Altura (cm)")
    plt.title("Comparación: Datos vs Modelo")
    plt.legend()
    st.pyplot(plt)


def plot_residuals(X, y, y_pred):
    residuals = y - y_pred
    plt.figure()
    plt.scatter(X, residuals, color="purple")
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Día")
    plt.ylabel("Error (y - y_pred)")
    plt.title("Residuos del modelo")
    st.pyplot(plt)


def plot_distribution(arr, titulo="Distribución de Alturas"):
    plt.figure()
    plt.hist(arr, bins=8, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(np.mean(arr), color="red", linestyle="--", label=f"Media={np.mean(arr):.2f}")
    plt.axvline(np.median(arr), color="green", linestyle="--", label=f"Mediana={np.median(arr):.2f}")
    plt.xlabel("Altura (cm)")
    plt.ylabel("Frecuencia")
    plt.title(titulo)
    plt.legend()
    st.pyplot(plt)


def plot_boxplot(arr, titulo="Boxplot de Alturas"):
    plt.figure()
    plt.boxplot(arr, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.ylabel("Altura (cm)")
    plt.title(titulo)
    st.pyplot(plt)


def plot_correlation(y, y_pred):
    plt.figure()
    plt.scatter(y, y_pred, alpha=0.7, color="orange")
    plt.xlabel("Altura Real")
    plt.ylabel("Altura Predicha")
    plt.title("Correlación Y real vs Y predicho")
    # línea de tendencia
    m, b = np.polyfit(y, y_pred, 1)
    plt.plot(y, m*y + b, color="red")
    st.pyplot(plt)


def export_pdf(enunciado: str, resultados: str, X, y, y_pred, modelo: str, data: pd.DataFrame,
               metrics: dict, descriptives_y: dict, descriptives_pred: dict, interp_kind: str):
    # Guardar todas las gráficas relevantes
    # 1. Comparación Datos vs Modelo
    plt.figure()
    plt.scatter(X, y, label="Datos Reales")
    plt.plot(X, y_pred, color="red", label=modelo)
    plt.xlabel("Día")
    plt.ylabel("Altura (cm)")
    plt.title("Comparación: Datos vs Modelo")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafico_comparacion.png")
    plt.close()

    # 2. Residuos del Modelo
    residuals = y - y_pred
    plt.figure()
    plt.scatter(X, residuals, color="purple")
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Día")
    plt.ylabel("Error (y - y_pred)")
    plt.title("Residuos del modelo")
    plt.tight_layout()
    plt.savefig("grafico_residuos.png")
    plt.close()

    # 3. Distribución de Alturas Reales
    plt.figure()
    plt.hist(y, bins=8, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(np.mean(y), color="red", linestyle="--", label=f"Media={np.mean(y):.2f}")
    plt.axvline(np.median(y), color="green", linestyle="--", label=f"Mediana={np.median(y):.2f}")
    plt.xlabel("Altura (cm)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de Y real")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafico_hist_y.png")
    plt.close()

    # 4. Distribución de Alturas Predichas
    plt.figure()
    plt.hist(y_pred, bins=8, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(np.mean(y_pred), color="red", linestyle="--", label=f"Media={np.mean(y_pred):.2f}")
    plt.axvline(np.median(y_pred), color="green", linestyle="--", label=f"Mediana={np.median(y_pred):.2f}")
    plt.xlabel("Altura (cm)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de Y predicho")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafico_hist_ypred.png")
    plt.close()

    # 5. Boxplot de Datos Reales
    plt.figure()
    plt.boxplot(y, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.ylabel("Altura (cm)")
    plt.title("Boxplot de Y real")
    plt.tight_layout()
    plt.savefig("grafico_box_y.png")
    plt.close()

    # 6. Boxplot de Datos Predichos
    plt.figure()
    plt.boxplot(y_pred, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.ylabel("Altura (cm)")
    plt.title("Boxplot de Y predicho")
    plt.tight_layout()
    plt.savefig("grafico_box_ypred.png")
    plt.close()

    # 7. Correlación Y vs Y_pred
    plt.figure()
    plt.scatter(y, y_pred, alpha=0.7, color="orange")
    plt.xlabel("Altura Real")
    plt.ylabel("Altura Predicha")
    plt.title("Correlación Y real vs Y predicho")
    m, b = np.polyfit(y, y_pred, 1)
    plt.plot(y, m*y + b, color="red")
    plt.tight_layout()
    plt.savefig("grafico_corr.png")
    plt.close()

    # Crear PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Título
    pdf.multi_cell(0, 10, "Reporte de Modelado del Crecimiento del Fríjol", align="C")
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
    # Q1 and Q3 removed from descriptives_y
    # ci_mean removed from descriptives_y
    pdf.ln(4)

    pdf.multi_cell(0, 8, "Estadísticas descriptivas (Predicción):")
    pdf.cell(0, 7, f"Media: {descriptives_pred['mean']}", ln=True)
    pdf.cell(0, 7, f"Varianza: {descriptives_pred['var']}", ln=True)
    pdf.cell(0, 7, f"Desviación estándar: {descriptives_pred['std']}", ln=True)
    pdf.cell(0, 7, f"Coef. variación: {descriptives_pred['cv']}", ln=True)
    pdf.cell(0, 7, f"Mediana: {descriptives_pred['median']}", ln=True)
    # Q1 and Q3 removed from descriptives_pred
    # ci_mean removed from descriptives_pred
    pdf.ln(6)


    # Insertar todas las gráficas al PDF con descripciones
    pdf.multi_cell(0, 8, "Gráficas generadas:")
    pdf.ln(2)
    # 1. Comparación Datos vs Modelo
    pdf.multi_cell(0, 8, "Comparación visual entre los datos experimentales reales y las predicciones del modelo seleccionado. Permite observar el ajuste del modelo a los datos.")
    pdf.image("grafico_comparacion.png", x=20, w=170)
    pdf.ln(2)
    # 2. Residuos del Modelo
    pdf.multi_cell(0, 8, "Gráfica de residuos: muestra la diferencia entre los valores reales y los predichos por el modelo para cada día. Permite identificar patrones o sesgos en el ajuste.")
    pdf.image("grafico_residuos.png", x=20, w=170)
    pdf.ln(2)
    # 3. Distribución de Y real
    pdf.multi_cell(0, 8, "Histograma de las alturas reales observadas. Permite visualizar la frecuencia de los valores de altura medidos experimentalmente.")
    pdf.image("grafico_hist_y.png", x=20, w=170)
    pdf.ln(2)
    # 4. Distribución de Y predicho
    pdf.multi_cell(0, 8, "Histograma de las alturas predichas por el modelo. Permite comparar la dispersión y tendencia de las predicciones frente a los datos reales.")
    pdf.image("grafico_hist_ypred.png", x=20, w=170)
    pdf.ln(2)
    # 5. Boxplot de Y real
    pdf.multi_cell(0, 8, "Boxplot de las alturas reales: muestra la mediana, los cuartiles y posibles valores atípicos de las alturas observadas.")
    pdf.image("grafico_box_y.png", x=20, w=170)
    pdf.ln(2)
    # 6. Boxplot de Y predicho
    pdf.multi_cell(0, 8, "Boxplot de las alturas predichas: muestra la mediana, los cuartiles y posibles valores atípicos de las alturas estimadas por el modelo.")
    pdf.image("grafico_box_ypred.png", x=20, w=170)
    pdf.ln(2)
    # 7. Correlación Y vs Y_pred
    pdf.multi_cell(0, 8, "Gráfica de dispersión entre las alturas reales y las predichas. Una relación cercana a la línea roja indica buen ajuste del modelo.")
    pdf.image("grafico_corr.png", x=20, w=170)

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
                if k not in ["q1", "q3", "ci_mean"]:
                    st.write(f"{k}: {v}")

            st.subheader("Estadísticas descriptivas (Predicción)")
            for k, v in descriptives_pred.items():
                if k not in ["q1", "q3", "ci_mean"]:
                    st.write(f"{k}: {v}")

            st.subheader("Interpolación aplicada")
            st.write(f"Tipo de interpolación aplicada: {interp_kind}")

            # Plot
            plot_results(X, y, y_pred, modelo, interpolacion=y_interp)

            # Gráficas separadas
            st.subheader("Gráficas de Resultados")

            st.markdown("**1. Comparación Datos vs Modelo**")
            plot_comparison(X, y, y_pred, modelo, interpolacion=y_interp)

            st.markdown("**2. Residuos del Modelo**")
            plot_residuals(X, y, y_pred)

            st.markdown("**3. Distribución de Alturas Reales**")
            plot_distribution(y, "Distribución de Y real")

            st.markdown("**4. Distribución de Alturas Predichas**")
            plot_distribution(y_pred, "Distribución de Y predicho")

            st.markdown("**5. Boxplot de Datos Reales**")
            plot_boxplot(y, "Boxplot de Y real")

            st.markdown("**6. Boxplot de Datos Predichos**")
            plot_boxplot(y_pred, "Boxplot de Y predicho")

            st.markdown("**7. Correlación Y vs Y_pred**")
            plot_correlation(y, y_pred)


            # Exportar PDF
            pdf_path = export_pdf(enunciado, texto, X, y, y_pred, modelo, data, metrics, descriptives_y, descriptives_pred, interp_kind)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("Descargar PDF", data=pdf_file, file_name=PDF_REPORT_FILE, mime="application/pdf")


if __name__ == "__main__":
    main()
