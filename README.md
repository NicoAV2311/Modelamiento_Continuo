# Aplicación de Modelado del Crecimiento de una Planta de Fríjol

## Descripción General
Esta aplicación, desarrollada en **Python con Streamlit**, permite **analizar, modelar y visualizar** el crecimiento de una planta de fríjol a partir de datos experimentales en formato CSV.  

Incluye distintos modelos matemáticos de ajuste, cálculos de métricas de error, estadísticas descriptivas, correlaciones y la exportación de reportes en PDF con gráficos y resultados.  

La versión presentada corresponde a una **extensión mejorada** respecto a versiones previas, incorporando nuevas funcionalidades de análisis, interpolación y exportación.

---

## Funcionalidades Principales
1. **Carga de datos en CSV** (con columnas `Dia` y `Altura_cm`).
2. **Selección de modelos matemáticos**:
   - Lineal
   - Polinómico (grado variable)
   - Exponencial
   - Logístico
3. **Entrenamiento automático** de cada modelo y cálculo de la ecuación ajustada.
4. **Interpolación adaptativa** según el modelo seleccionado.
5. **Cálculo de métricas de error**:
   - MSE, RMSE, MAE, MAPE
   - R²
   - Bias
   - Varianzas y covarianza
   - Correlaciones de Pearson y Spearman
6. **Estadísticas descriptivas** (manuales y automáticas):
   - Media, Varianza, Desviación Estándar
   - Coeficiente de Variación
   - Mediana, Q1, Q3
   - Intervalos de confianza (IC 95%)
7. **Visualización gráfica** interactiva de los datos, predicciones e interpolaciones.
8. **Generación de reportes en PDF** con:
   - Enunciado del ejercicio
   - Datos cargados
   - Resultados del modelo
   - Métricas de error
   - Estadísticas descriptivas
   - Gráficas
9. **Descarga directa del reporte en PDF** desde la interfaz de Streamlit.

---

## Estructura del Proyecto
📦 proyecto_frijol
┣ app.py # Código principal de la aplicación (Streamlit)
┣ README.md # Documento explicativo del proyecto
┣ requirements.txt # Dependencias necesarias
┣ grafico.png # Imagen temporal generada para el PDF
┣ reporte_frijol_extendido.pdf # Ejemplo de reporte generado
┗ data
┗ ejemplo.csv # Archivo de datos de prueba


---

## Módulos y Funcionalidades

### 1. **Carga y Preprocesamiento de Datos**
- `load_data`: Carga archivos CSV y convierte los datos a `numpy` y `pandas`.

### 2. **Modelos de Crecimiento**
- `train_lineal`: Ajuste de regresión lineal.
- `train_polinomico`: Ajuste polinómico de grado variable.
- `train_exponencial`: Ajuste exponencial (log-transformación).
- `train_logistico`: Ajuste logístico mediante `scipy.optimize.curve_fit`.

### 3. **Interpolación**
- `interpolate_by_model`: Selección automática de método de interpolación según el modelo:
  - Lineal/polinomial
  - Exponencial (en log-espacio)
  - Spline cúbico para logístico

### 4. **Cálculo de Métricas**
- `calculate_all_metrics`: Métricas de error, correlaciones y varianzas.

### 5. **Estadísticas Descriptivas**
- `calculate_descriptives`: Cálculo manual de media, varianza, desviación estándar, coeficiente de variación, cuartiles e intervalos de confianza.

### 6. **Visualización y Reportes**
- `plot_results`: Genera gráficos en Streamlit.
- `export_pdf`: Crea un reporte PDF con resultados, métricas y gráfica.

### 7. **Aplicación Streamlit**
- `main`: Interfaz gráfica que permite cargar datos, seleccionar modelo, entrenar, visualizar métricas y exportar reporte.

---

## Evidencias de Cambios Hechos
- Se añadió **modelo logístico** con ajuste mediante `curve_fit`.
- Se implementó **interpolación automática** según el modelo.
- Se incluyeron **estadísticas descriptivas avanzadas** (cuartiles, coeficiente de variación, intervalos de confianza).
- Se incorporaron **métricas adicionales**: Bias, Pearson, Spearman.
- Se desarrolló un **reporte PDF extendido** con resultados completos y gráficas.
- Se centralizaron los cálculos de métricas de error y descriptivas en funciones auxiliares.
- Se mejoró la interfaz en **Streamlit** para mayor interactividad.

---

## Entregables
1. **Código fuente completo** (`app.py`).
2. **Archivo README.md** documentado.
3. **Archivo requirements.txt** con dependencias:
numpy
pandas
matplotlib
streamlit
scikit-learn
fpdf
scipy

4. **Archivo CSV de ejemplo** (`data/ejemplo.csv`) para pruebas.
5. **Reporte PDF de ejemplo** (`reporte_frijol_extendido.pdf`).

---

## Ejecución del Proyecto

### 1. Clonar el repositorio

git clone https://github.com/NicoAV2311/Modelamiento_Continuo.git
cd Modelamiento Continuo

###

### 2. Instalar Dependencias
pip install -r requirements.txt

### 3. Ejecutar Aplicación
python -m streamlit run Prueba de app con mejoras visuales para lo del frijol.py

### 4. Integración
1. Cargar un archivo CSV con datos de crecimiento
2. Seleccionar modelo de ajuste
3. Entregar y visualizar metricas y graficas
4. Descargar reporte PDF

Ejemplo de datos CSV
0	0,59
1	1,18
2	2,1
3	3,42
4	5,13
5	7,19
6	9,54
7	12,08
8	14,72
9	17,35
10	19,91
11	22,33
12	24,58
13	26,64
14	28,48
15	30,12
16	31,56
17	32,82
18	33,91
19	34,84
20	35,64
21	36,33
22	36,91
23	37,4
24	37,82
25	38,17
26	38,46
27	38,71
28	38,92
29	39,1
30	39,24

Autores
Nicolas Arango Vergara
Sebastian Gomez Sepulveda
Santiago Hoyos Araque
Juan Pablo Zapata Arenas
