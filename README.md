
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
  - MSE, RMSE
  - R²
  - Varianzas y covarianza
6. **Estadísticas descriptivas** (manuales y automáticas):
  - Media, Varianza, Desviación Estándar
  - Coeficiente de Variación
  - Mediana
7. **Visualización gráfica** interactiva de los datos, predicciones e interpolaciones.
8. **Generación de reportes en PDF** con:
  - Enunciado del ejercicio
  - Datos cargados
  - Resultados del modelo
  - Métricas de error
  - Estadísticas descriptivas
  - Gráficas explicadas
9. **Descarga directa del reporte en PDF** desde la interfaz de Streamlit.

---

## Estructura del Proyecto
Reto 1 Actualizado
┣ Nuevas Pruebas codigo.py  # Código principal de la aplicación (Streamlit)
┣ README.md                 # Documento explicativo del proyecto
┣ requirements.txt          # Dependencias necesarias
┣ grafico.png               # Imagen temporal generada para el PDF
┣ Reporte Reto 1.pdf        # Ejemplo de reporte generado
┗ data/
   ┗ ejemplo.csv           # Archivo de datos de prueba

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
- `export_pdf`: Crea un reporte PDF con resultados, métricas y gráficas explicadas.

### 7. **Aplicación Streamlit**
- `main`: Interfaz gráfica que permite cargar datos, seleccionar modelo, entrenar, visualizar métricas y exportar reporte.

---

## Evidencias de Cambios Hechos
- Se añadió **modelo logístico** con ajuste mediante `curve_fit`.
- Se implementó **interpolación automática** según el modelo.
- Se incluyeron **estadísticas descriptivas avanzadas** (cuartiles, coeficiente de variación, intervalos de confianza).
- Se desarrolló un **reporte PDF extendido** con resultados completos y gráficas explicadas.
- Se centralizaron los cálculos de métricas de error y descriptivas en funciones auxiliares.
- Se mejoró la interfaz en **Streamlit** para mayor interactividad.

---

## Entregables
1. **Código fuente completo** (`Nuevas Pruebas codigo.py`).
2. **Archivo README.md** documentado.
3. **Archivo requirements.txt** con dependencias:
  - numpy
  - pandas
  - matplotlib
  - streamlit
  - scikit-learn
  - fpdf
  - scipy
4. **Archivo CSV de ejemplo** (`data/ejemplo.csv`) para pruebas.
5. **Reporte PDF de ejemplo** (`Reporte Reto 1.pdf`).

---

## Ejecución del Proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/NicoAV2311/Modelamiento_Continuo.git
cd "Modelación y Simulación de sistemas/Reto 1 Actualizado"
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar Aplicación

```bash
streamlit run "Nuevas Pruebas codigo.py"
```

### 4. Integración y Uso
1. Cargar un archivo CSV con datos de crecimiento
2. Seleccionar modelo de ajuste
3. Visualizar métricas y gráficas
4. Descargar reporte PDF

#### Ejemplo de datos CSV

```
Dia;Altura_cm
Dia	Altura_cm
1	0.00
2	0.00
3	0.00
4	0.64
5	0.88
6	1.19
7	1.62
8	2.19
9	2.93
10	3.89
11	5.12
12	6.63
13	8.43
14	10.52
15	12.81
16	15.23
17	17.65
18	19.94
19	22.03
20	23.83
21	25.34
22	26.57
23	27.53
24	28.28
25	28.84
26	29.27
27	29.59
28	29.82
29	29.99
30	30.12
```

---

## Autores
- Nicolas Arango Vergara
- Sebastian Gomez Sepulveda
- Santiago Hoyos Araque
- Juan Pablo Zapata Arenas
