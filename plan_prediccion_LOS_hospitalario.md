# Plan de Solución: Predicción del Tiempo de Estadía Hospitalaria (LOS)
> Documento de consultoría técnica — Preparado como guía completa de implementación

---

## Tabla de contenidos

1. [Descripción del problema](#1-descripción-del-problema)
2. [Descripción de los datos disponibles](#2-descripción-de-los-datos-disponibles)
3. [Arquitectura general del algoritmo](#3-arquitectura-general-del-algoritmo)
4. [Etapa A — Carga de datos](#etapa-a--carga-de-datos)
5. [Etapa B — Limpieza de datos](#etapa-b--limpieza-de-datos)
6. [Etapa C — Análisis exploratorio (EDA)](#etapa-c--análisis-exploratorio-eda)
7. [Etapa D — Ingeniería de features](#etapa-d--ingeniería-de-features)
8. [Etapa E — División train / test](#etapa-e--división-train--test)
9. [Etapa F — Entrenamiento y comparación de modelos](#etapa-f--entrenamiento-y-comparación-de-modelos)
10. [Etapa G — Evaluación final](#etapa-g--evaluación-final)
11. [Etapa H — Interpretación y entrega](#etapa-h--interpretación-y-entrega)
12. [Decisiones técnicas justificadas](#12-decisiones-técnicas-justificadas)
13. [Stack tecnológico completo](#13-stack-tecnológico-completo)
14. [Preguntas clave al cliente](#14-preguntas-clave-al-cliente)

---

## 1. Descripción del problema

El objetivo es construir un **modelo predictivo** que, dado el perfil clínico de un paciente al momento de su ingreso al hospital (sus diagnósticos y procedimientos médicos en nomenclatura ICD-10), estime cuántos días permanecerá hospitalizado. A este tiempo se le llama **LOS** (Length of Stay, o tiempo de estadía).

El modelo no es un sistema de optimización: no decide cómo asignar camas ni minimiza costos directamente. Su función es producir una estimación numérica del LOS para cada nuevo paciente, lo que a su vez permite que el personal hospitalario tome mejores decisiones de planificación.

La variable que el modelo debe predecir —llamada variable objetivo o target— es el LOS calculado en días como la diferencia entre la fecha de egreso y la fecha de ingreso de cada paciente.

---

## 2. Descripción de los datos disponibles

Los datos provienen de un archivo Excel con dos hojas que deben trabajarse de forma conjunta.

### Hoja 1: Procedimientos por paciente

Esta hoja contiene cuatro columnas. La columna `Case` es el identificador único del paciente. La columna `Procedure` contiene el código ICD-10 del procedimiento médico realizado. Las columnas `Date` y `Release` contienen la fecha de ingreso y la fecha de egreso respectivamente.

Un mismo paciente puede aparecer en múltiples filas porque puede haber tenido varios procedimientos durante la misma hospitalización. Las fechas de ingreso y egreso son las mismas en todas las filas de un mismo paciente.

### Hoja 2: Datos diagnósticos

Esta hoja contiene tres columnas. La columna `CASE` es el mismo identificador de paciente de la hoja 1. La columna `PrincSec` indica si el diagnóstico es primario (`P`) o secundario (`S`). La columna `Diagnosis` contiene el código ICD-10 del diagnóstico.

Un mismo paciente puede tener múltiples diagnósticos. Cuando aparece el valor `DDDDDD` en la columna de diagnóstico, se trata de un **separador administrativo** que debe eliminarse antes de cualquier análisis.

> **Advertencia importante:** El profesor indicó que los IDs de pacientes entre ambas hojas "se supone que son los mismos", lo cual implica que pueden existir discrepancias. Antes de avanzar, es obligatorio verificar que todos los IDs de la hoja 2 tienen correspondencia en la hoja 1 y viceversa.

---

## 3. Arquitectura general del algoritmo

El pipeline completo se estructura en ocho etapas secuenciales. La salida de cada etapa alimenta la siguiente.

```
[Excel hoja 1 + hoja 2]
        │
        ▼
[A] Carga de datos brutos
        │
        ▼
[B] Limpieza: calcular LOS, eliminar DDDDDD, verificar IDs
        │
        ▼
[C] Análisis exploratorio: distribución LOS, correlaciones
        │
        ▼
[D] Ingeniería de features: capítulos ICD, CCI, TF-IDF
        │
        ▼
[E] División 80% train / 20% test
        │
        ▼
[F] Entrenamiento de 4 modelos con validación cruzada k-fold
        │
        ▼
[G] Evaluación final del modelo ganador sobre test set
        │
        ▼
[H] Interpretación SHAP + función de predicción para nuevos casos
```

---

## Etapa A — Carga de datos

El primer paso es leer ambas hojas del archivo Excel y verificar que su estructura es la esperada. Se usa `pandas` para esto.

```python
import pandas as pd

# Leer ambas hojas del Excel
df_proc = pd.read_excel("datos.xlsx", sheet_name="Procedimientos por paciente")
df_diag = pd.read_excel("datos.xlsx", sheet_name="Datos diagnosticos")

# Convertir fechas al tipo datetime de pandas para poder operar sobre ellas
df_proc['Date']    = pd.to_datetime(df_proc['Date'],    dayfirst=True)
df_proc['Release'] = pd.to_datetime(df_proc['Release'], dayfirst=True)

# Vista rápida de ambas hojas
print(df_proc.head())
print(df_diag.head())
print(df_proc.dtypes)
print(df_diag.dtypes)
```

**Qué verificar en este paso:** que las fechas se leyeron correctamente (tipo `datetime64`), que los IDs de paciente son del mismo tipo en ambas hojas (ambos enteros o ambos strings), y que las columnas tienen los nombres esperados.

---

## Etapa B — Limpieza de datos

Esta es la etapa más crítica de todo el pipeline. Un error aquí contamina todas las etapas siguientes.

### B1. Calcular el LOS como variable objetivo

Como un mismo paciente aparece en múltiples filas de la hoja 1, hay que agrupar por `Case` y tomar la fecha de entrada mínima y la fecha de salida máxima antes de calcular la diferencia.

```python
# Agrupar por paciente y obtener fechas extremas
los = df_proc.groupby('Case').agg(
    fecha_entrada=('Date',    'min'),
    fecha_salida =('Release', 'max')
).reset_index()

# LOS en días como entero
los['LOS'] = (los['fecha_salida'] - los['fecha_entrada']).dt.days

# Revisar casos problemáticos
print("LOS negativos:", (los['LOS'] < 0).sum())   # Error de datos si > 0
print("LOS = 0 días:", (los['LOS'] == 0).sum())   # Alta y egreso el mismo día
print(los['LOS'].describe())
```

Si existen valores de LOS negativos, indica un error en los datos fuente (fecha de salida anterior a la de entrada) y debe investigarse caso a caso. Los LOS de 0 días corresponden a pacientes ambulatorios que ingresaron y egresaron el mismo día; dependiendo del contexto clínico, pueden incluirse o filtrarse.

### B2. Limpiar la tabla de diagnósticos

```python
# Eliminar separadores DDDDDD
df_diag = df_diag[df_diag['Diagnosis'] != 'DDDDDD'].copy()

# Estandarizar: mayúsculas y sin espacios
df_diag['Diagnosis'] = df_diag['Diagnosis'].str.strip().str.upper()
df_diag['PrincSec']  = df_diag['PrincSec'].str.strip().str.upper()

print("Valores únicos en PrincSec:", df_diag['PrincSec'].unique())
# Esperado: solo ['P', 'S']
```

### B3. Verificar consistencia de IDs entre hojas

```python
# Renombrar para igualar nombres
df_diag_ids = df_diag[['CASE']].rename(columns={'CASE': 'Case'})

merge_check = pd.merge(
    los[['Case']],
    df_diag_ids.drop_duplicates(),
    on='Case',
    how='outer',
    indicator=True
)
print(merge_check['_merge'].value_counts())
# 'both'       → pacientes presentes en ambas hojas (lo esperado)
# 'left_only'  → pacientes con procedimientos pero sin diagnósticos
# 'right_only' → diagnósticos sin procedimientos asociados
```

Cualquier valor diferente de `'both'` requiere una decisión explícita: excluir esos pacientes o imputar la información faltante.

### B4. Consolidar procedimientos y diagnósticos por paciente

```python
# Lista de procedimientos por paciente
proc_agg = df_proc.groupby('Case')['Procedure'].apply(list).reset_index()
proc_agg.columns = ['Case', 'procedures_list']

# Diagnósticos primarios (P) por paciente
diag_P = (df_diag[df_diag['PrincSec'] == 'P']
          .groupby('CASE')['Diagnosis']
          .apply(list)
          .reset_index()
          .rename(columns={'CASE': 'Case', 'Diagnosis': 'diag_primarios'}))

# Diagnósticos secundarios (S) por paciente
diag_S = (df_diag[df_diag['PrincSec'] == 'S']
          .groupby('CASE')['Diagnosis']
          .apply(list)
          .reset_index()
          .rename(columns={'CASE': 'Case', 'Diagnosis': 'diag_secundarios'}))

# Unir todo en un DataFrame maestro: una fila por paciente
df = (los
      .merge(proc_agg, on='Case', how='left')
      .merge(diag_P,   on='Case', how='left')
      .merge(diag_S,   on='Case', how='left'))

print(f"Dataset maestro: {df.shape[0]} pacientes, {df.shape[1]} columnas")
```

---

## Etapa C — Análisis exploratorio (EDA)

Antes de entrenar cualquier modelo es obligatorio entender la forma de los datos. Esta etapa produce visualizaciones que guiarán decisiones técnicas posteriores, especialmente si se debe aplicar la transformación logarítmica al LOS.

### C1. Distribución del LOS

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# LOS original
sns.histplot(df['LOS'], bins=40, ax=axes[0], color='steelblue')
axes[0].set_title('Distribución LOS original (días)')
axes[0].set_xlabel('LOS (días)')

# LOS transformado con log
sns.histplot(np.log1p(df['LOS']), bins=40, ax=axes[1], color='coral')
axes[1].set_title('Distribución log(LOS + 1)')
axes[1].set_xlabel('log(LOS + 1)')

plt.tight_layout()
plt.savefig('distribucion_LOS.png', dpi=150)
plt.show()

# Métricas de forma de la distribución
print(f"Asimetría (skewness): {df['LOS'].skew():.2f}")
print(f"Curtosis:             {df['LOS'].kurtosis():.2f}")
# Si skewness > 2: usar transformación log en el target del modelo
```

### C2. LOS promedio por diagnóstico primario

```python
df_diag_P_los = (df_diag[df_diag['PrincSec'] == 'P']
                 .merge(los[['Case', 'LOS']], left_on='CASE', right_on='Case'))

top_diag = (df_diag_P_los
            .groupby('Diagnosis')['LOS']
            .agg(['mean', 'count'])
            .reset_index()
            .query('count >= 5')           # Solo diagnósticos con al menos 5 casos
            .sort_values('mean', ascending=False)
            .head(20))

plt.figure(figsize=(10, 6))
sns.barplot(data=top_diag, x='mean', y='Diagnosis', palette='Blues_r')
plt.xlabel('LOS promedio (días)')
plt.title('Top 20 diagnósticos primarios por LOS promedio')
plt.tight_layout()
plt.savefig('top_diagnosticos_LOS.png', dpi=150)
plt.show()
```

### C3. Correlaciones iniciales

```python
from scipy import stats

df['n_procedimientos']   = df['procedures_list'].apply(
    lambda x: len(x) if isinstance(x, list) else 0)
df['n_diag_secundarios'] = df['diag_secundarios'].apply(
    lambda x: len(x) if isinstance(x, list) else 0)

for col in ['n_procedimientos', 'n_diag_secundarios']:
    r, p = stats.pearsonr(df[col], df['LOS'])
    print(f"Correlación {col} vs LOS: r={r:.3f}  p={p:.4f}")
```

---

## Etapa D — Ingeniería de features

Esta es la etapa que transforma los datos clínicos en el formato numérico que los modelos de machine learning pueden procesar. Se trabaja en tres capas de complejidad creciente.

### Por qué los códigos ICD-10 no pueden entrar directamente al modelo

Los modelos de machine learning trabajan exclusivamente con números. Un código como `E6601` es texto y el modelo no puede interpretarlo directamente. La solución no es buscar manualmente qué significa cada código, sino aprovechar que la estructura jerárquica del sistema ICD-10 ya codifica información clínica, y complementarla con índices de comorbilidad validados médicamente.

### D1. Capa 1 — Estructura jerárquica del propio código (sin librerías externas)

El sistema ICD-10 fue diseñado con una jerarquía interna: la primera letra identifica el capítulo clínico, los tres primeros caracteres identifican el bloque de enfermedades, y el código completo especifica la condición exacta.

```python
# Extraer capítulo (letra inicial) y bloque (3 primeros chars)
df_diag['capitulo'] = df_diag['Diagnosis'].str[0]        # 'E' para E6601
df_diag['bloque']   = df_diag['Diagnosis'].str[:3]       # 'E66' para E6601

# Los capítulos del sistema ICD-10 son:
# A-B: enfermedades infecciosas y parasitarias
# C-D: neoplasias (tumores)
# E:   enfermedades endocrinas, nutricionales y metabólicas
# F:   trastornos mentales y del comportamiento
# G:   enfermedades del sistema nervioso
# H:   enfermedades del ojo, oído y apéndices
# I:   enfermedades del sistema circulatorio
# J:   enfermedades del sistema respiratorio
# K:   enfermedades del sistema digestivo
# M:   enfermedades del sistema musculoesquelético
# N:   enfermedades del sistema genitourinario
# S-T: traumatismos, envenenamientos y lesiones

# Para procedimientos ICD-10-PCS, los 3 primeros caracteres también tienen significado:
df['tipo_proc'] = df['procedures_list'].apply(
    lambda lst: [c[:3] for c in lst] if isinstance(lst, list) else [])

# One-hot encoding de capítulos de diagnóstico por paciente
from sklearn.preprocessing import MultiLabelBinarizer

df['capitulos_list'] = df['diag_primarios'].apply(
    lambda lst: [c[0] for c in lst] if isinstance(lst, list) else [])

mlb = MultiLabelBinarizer()
caps_encoded = mlb.fit_transform(df['capitulos_list'])
df_caps = pd.DataFrame(caps_encoded,
                       columns=['cap_' + c for c in mlb.classes_],
                       index=df.index)
```

### D2. Capa 2 — Índice de Comorbilidad de Charlson (CCI)

El Índice de Charlson es el estándar clínico para cuantificar la severidad global de un paciente. Fue creado en 1987 y asigna pesos numéricos a cada tipo de diagnóstico según su impacto en el riesgo de mortalidad y complicaciones. La suma de los pesos de todos los diagnósticos de un paciente produce su CCI.

Ejemplos de pesos: infarto de miocardio = 1, diabetes sin complicaciones = 1, diabetes con complicaciones = 2, insuficiencia renal moderada = 2, tumor metastásico = 6, SIDA = 6. Un paciente sin enfermedades de fondo tiene CCI = 0; con múltiples condiciones graves puede superar CCI = 10.

```python
# Instalación: pip install comorbidipy
from comorbidipy import charlson, elixhauser

# Calcular CCI para cada paciente
def calcular_cci(lista_diagnosticos):
    if not isinstance(lista_diagnosticos, list) or len(lista_diagnosticos) == 0:
        return 0
    todos = lista_diagnosticos  # incluye primarios y secundarios
    try:
        return charlson(todos, icd_version=10)
    except:
        return 0

# Combinar diagnósticos primarios y secundarios para el CCI
df['todos_diag'] = df.apply(
    lambda row: (row['diag_primarios']  if isinstance(row['diag_primarios'],  list) else []) +
                (row['diag_secundarios'] if isinstance(row['diag_secundarios'], list) else []),
    axis=1)

df['charlson_score'] = df['todos_diag'].apply(calcular_cci)

print("Distribución del Índice de Charlson:")
print(df['charlson_score'].value_counts().sort_index())
```

El CCI es la feature más valiosa del modelo porque codifica en un único número toda la complejidad clínica del paciente, y tiene una correlación demostrada con el LOS en la literatura médica.

El índice de Elixhauser es una alternativa más moderna con 31 categorías binarias (0/1) en lugar de un solo número. Puede usarse en complemento con el CCI.

### D3. Capa 3 — TF-IDF sobre códigos ICD-10 completos

TF-IDF (Term Frequency – Inverse Document Frequency) es una técnica del procesamiento de lenguaje natural que aquí se reutiliza para representar los diagnósticos. Cada paciente se trata como un "documento" y sus códigos ICD-10 como las "palabras". TF cuenta cuántas veces aparece cada código en ese paciente; IDF penaliza los códigos que aparecen en casi todos los pacientes (porque si todos tienen el mismo código, ese código no ayuda a distinguir quién estará más días).

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Convertir listas de diagnósticos a strings de texto
df['diag_texto'] = df['todos_diag'].apply(
    lambda lst: ' '.join(lst) if isinstance(lst, list) else '')
df['proc_texto'] = df['procedures_list'].apply(
    lambda lst: ' '.join(lst) if isinstance(lst, list) else '')

# TF-IDF: máximo 100 features para diagnósticos, 50 para procedimientos
# min_df=2 descarta códigos que solo aparecen en 1 paciente (probablemente errores)
tfidf_diag = TfidfVectorizer(max_features=100, min_df=2)
tfidf_proc = TfidfVectorizer(max_features=50,  min_df=2)

X_tfidf_diag = tfidf_diag.fit_transform(df['diag_texto']).toarray()
X_tfidf_proc = tfidf_proc.fit_transform(df['proc_texto']).toarray()
```

### D4. Ensamblaje final de la matriz de features

```python
import numpy as np

# Features de conteo (siempre incluir)
X_conteo = df[['n_procedimientos', 'n_diag_secundarios', 'charlson_score']].fillna(0).values

# Features temporales
df['mes_entrada']  = df['fecha_entrada'].dt.month
df['dia_semana']   = df['fecha_entrada'].dt.dayofweek
X_temporal = df[['mes_entrada', 'dia_semana']].values

# Matriz X final: concatenar todas las capas
X = np.hstack([
    X_conteo,               # 3 columnas
    df_caps.values,         # ~21 columnas (una por capítulo ICD)
    X_temporal,             # 2 columnas
    X_tfidf_diag,           # 100 columnas
    X_tfidf_proc            # 50 columnas
])

# Variable objetivo (con transformación logarítmica — ver sección 12)
y = np.log1p(df['LOS'].values)

print(f"Matriz X: {X.shape}")   # (n_pacientes, n_features)
print(f"Vector y: {y.shape}")   # (n_pacientes,)
```

---

## Etapa E — División train / test

Se reserva el 20% de los pacientes como conjunto de test. Este conjunto no se toca hasta la evaluación final: no se usa para entrenar, no se usa para elegir el mejor modelo, y no se usa para ajustar hiperparámetros. Su único propósito es medir la precisión real del modelo sobre datos que nunca vio.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42    # fija la semilla aleatoria para reproducibilidad
)

print(f"Train: {X_train.shape[0]} pacientes")
print(f"Test:  {X_test.shape[0]} pacientes")

# Verificar que ambos conjuntos tienen distribuciones similares de LOS
print("LOS promedio train:", np.expm1(y_train).mean().round(2), "días")
print("LOS promedio test: ", np.expm1(y_test).mean().round(2), "días")
```

---

## Etapa F — Entrenamiento y comparación de modelos

Se entrenan cuatro modelos sobre el conjunto de train, evaluados con validación cruzada de 5 pliegues (k-fold cross-validation). Esto significa que el train set se divide en 5 partes iguales, se entrena con 4 y se evalúa con la quinta, rotando 5 veces. El MAE promedio de las 5 rondas es la métrica de comparación.

### Los cuatro modelos y por qué se incluye cada uno

**Regresión Ridge** actúa como baseline. Asume que el LOS es una combinación lineal de las features. Sus coeficientes son fáciles de interpretar. Si los modelos más complejos no lo superan significativamente, Ridge es la elección correcta por su simplicidad.

**Regresión Lasso** es similar a Ridge pero con una penalización que fuerza algunos coeficientes a ser exactamente cero, produciendo un modelo esparso que selecciona automáticamente las features más relevantes.

**Random Forest** construye cientos de árboles de decisión en paralelo, cada uno entrenado con una muestra aleatoria de los datos. La predicción final es el promedio de todos los árboles. Captura interacciones entre diagnósticos (por ejemplo, que diabetes + insuficiencia renal juntas predicen un LOS mayor que la suma de sus efectos individuales). Es el modelo recomendado como punto de partida no lineal.

**Gradient Boosting (XGBoost)** construye los árboles en secuencia: cada árbol nuevo aprende a corregir los errores del árbol anterior, siguiendo la dirección del gradiente de la función de error. Es el método más preciso en datos tabulares clínicos, pero requiere calibrar más hiperparámetros.

```python
from sklearn.linear_model  import Ridge, Lasso
from sklearn.ensemble       import RandomForestRegressor
from xgboost                import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics         import mean_absolute_error

kf = KFold(n_splits=5, shuffle=True, random_state=42)

modelos = {
    'Ridge':          Ridge(alpha=1.0),
    'Lasso':          Lasso(alpha=0.01),
    'Random Forest':  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost':        XGBRegressor(n_estimators=300, learning_rate=0.05,
                                   random_state=42, n_jobs=-1, verbosity=0)
}

resultados = {}
for nombre, modelo in modelos.items():
    scores = cross_val_score(modelo, X_train, y_train,
                             cv=kf,
                             scoring='neg_mean_absolute_error',
                             n_jobs=-1)
    mae_log  = -scores.mean()
    std_log  = scores.std()
    resultados[nombre] = mae_log
    print(f"{nombre:18s}  MAE (log-scale): {mae_log:.4f} ± {std_log:.4f}")

mejor_nombre = min(resultados, key=resultados.get)
print(f"\nModelo ganador: {mejor_nombre}")
```

---

## Etapa G — Evaluación final

El modelo ganador se entrena sobre todo el conjunto de train y se evalúa sobre el test set. Las métricas se reportan en días reales (revirtiendo la transformación logarítmica).

```python
mejor_modelo = modelos[mejor_nombre]
mejor_modelo.fit(X_train, y_train)

y_pred_log  = mejor_modelo.predict(X_test)

# Revertir transformación log: expm1(x) = e^x - 1
y_pred_dias = np.expm1(y_pred_log)
y_true_dias = np.expm1(y_test)

mae  = mean_absolute_error(y_true_dias, y_pred_dias)
rmse = np.sqrt(mean_squared_error(y_true_dias, y_pred_dias))
r2   = r2_score(y_true_dias, y_pred_dias)

print(f"MAE:  {mae:.2f} días   (el modelo se equivoca en promedio {mae:.2f} días)")
print(f"RMSE: {rmse:.2f} días")
print(f"R²:   {r2:.4f}         (explica el {r2*100:.1f}% de la variabilidad del LOS)")

# Gráfico predicho vs real
plt.figure(figsize=(7, 7))
plt.scatter(y_true_dias, y_pred_dias, alpha=0.4, s=20, color='steelblue')
plt.plot([0, y_true_dias.max()], [0, y_true_dias.max()], 'r--', lw=1)
plt.xlabel('LOS real (días)')
plt.ylabel('LOS predicho (días)')
plt.title(f'Predicho vs Real — {mejor_nombre}')
plt.tight_layout()
plt.savefig('predicho_vs_real.png', dpi=150)
plt.show()
```

### Cómo interpretar el MAE

Si el LOS promedio de los pacientes es, por ejemplo, 7 días y el MAE es 1.8 días, el error relativo es del 26%, que es un resultado razonable para datos clínicos reales con la cantidad de variables disponibles. Un MAE inferior a 2 días en un hospital de complejidad media es considerado un buen resultado en la literatura.

---

## Etapa H — Interpretación y entrega

### H1. Importancia de variables (feature importance)

```python
import pandas as pd

# Construir nombres de columnas en el mismo orden que se ensamblaron en X
feature_names = (
    ['n_procedimientos', 'n_diag_secundarios', 'charlson_score'] +
    ['cap_' + c for c in mlb.classes_] +
    ['mes_entrada', 'dia_semana'] +
    ['diag_tfidf_' + str(i) for i in range(100)] +
    ['proc_tfidf_' + str(i) for i in range(50)]
)

importances = mejor_modelo.feature_importances_
feat_imp = (pd.DataFrame({'feature': feature_names, 'importance': importances})
            .sort_values('importance', ascending=False)
            .head(20))

plt.figure(figsize=(10, 6))
plt.barh(feat_imp['feature'], feat_imp['importance'], color='steelblue')
plt.xlabel('Importancia relativa')
plt.title('Top 20 variables más influyentes en el LOS')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('importancia_variables.png', dpi=150)
plt.show()
```

### H2. Explicabilidad con SHAP

SHAP (SHapley Additive exPlanations) responde a la pregunta: dado que el modelo predijo que este paciente estará 12 días, ¿cuánto de eso se debe a su diagnóstico de diabetes, cuánto a tener 5 procedimientos, cuánto a su índice de Charlson? Es la herramienta estándar para explicar predicciones a personal médico no técnico.

```python
import shap

explainer   = shap.TreeExplainer(mejor_modelo)
shap_values = explainer.shap_values(X_test)

# Gráfico resumen: importancia con dirección del efecto (positivo/negativo)
shap.summary_plot(shap_values, X_test,
                  feature_names=feature_names,
                  max_display=15)

# Explicación de un paciente individual (ej. el primero del test set)
shap.force_plot(
    explainer.expected_value,
    shap_values[0, :],
    X_test[0, :],
    feature_names=feature_names
)
```

### H3. Función de predicción para nuevos pacientes

```python
import joblib

# Guardar el modelo entrenado para reutilizarlo
joblib.dump(mejor_modelo, 'modelo_LOS.pkl')
joblib.dump(tfidf_diag,   'tfidf_diag.pkl')
joblib.dump(tfidf_proc,   'tfidf_proc.pkl')
joblib.dump(mlb,          'mlb_capitulos.pkl')

def predecir_LOS(diagnosticos_P: list,
                 diagnosticos_S: list,
                 procedimientos: list,
                 modelo, tfidf_d, tfidf_p, mlb_caps, charlson_fn) -> float:
    """
    Predice el LOS en días para un nuevo paciente.

    Parámetros:
        diagnosticos_P  : lista de códigos ICD-10 primarios, ej. ['E6601']
        diagnosticos_S  : lista de códigos ICD-10 secundarios, ej. ['Z6841', 'I10']
        procedimientos  : lista de códigos de procedimiento, ej. ['0DB64Z3']
    Retorna:
        LOS estimado en días (float)
    """
    todos_diag  = diagnosticos_P + diagnosticos_S
    diag_texto  = ' '.join(todos_diag)
    proc_texto  = ' '.join(procedimientos)

    X_diag = tfidf_d.transform([diag_texto]).toarray()
    X_proc = tfidf_p.transform([proc_texto]).toarray()

    cci    = charlson_fn(todos_diag, icd_version=10)
    caps   = mlb_caps.transform([[c[0] for c in todos_diag]])

    X_conteo  = np.array([[len(procedimientos), len(diagnosticos_S), cci]])
    X_temporal = np.array([[0, 0]])  # sin fecha, usar cero o promedio histórico

    X_nuevo = np.hstack([X_conteo, caps, X_temporal, X_diag, X_proc])

    los_log = modelo.predict(X_nuevo)[0]
    return round(np.expm1(los_log), 1)

# Ejemplo de uso
pred = predecir_LOS(
    diagnosticos_P=['E6601'],
    diagnosticos_S=['Z6841', 'I10'],
    procedimientos=['0DB64Z3'],
    modelo=mejor_modelo,
    tfidf_d=tfidf_diag,
    tfidf_p=tfidf_proc,
    mlb_caps=mlb,
    charlson_fn=charlson
)
print(f"LOS estimado: {pred} días")
```

---

## 12. Decisiones técnicas justificadas

### ¿Por qué Random Forest / Gradient Boosting y no regresión lineal?

La regresión lineal asume que el efecto de cada diagnóstico sobre el LOS es constante e independiente del resto. Matemáticamente, el modelo dice que el LOS es $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots$, donde cada $\beta_j$ es fijo. Esto ignora las **interacciones entre variables**: un paciente con diabetes e insuficiencia renal simultáneas no tiene simplemente LOS(diabetes) + LOS(renal); la combinación puede ser mucho más grave que la suma.

Los árboles de decisión capturan estas interacciones naturalmente mediante reglas del tipo "si el paciente tiene diabetes AND insuficiencia renal AND más de 3 procedimientos, entonces LOS ≈ 14 días". Random Forest construye cientos de árboles en paralelo con subconjuntos aleatorios de los datos, y promedia sus predicciones. Este promedio cancela los errores individuales de cada árbol (efecto estadístico llamado **variance reduction por ensemble**), produciendo predicciones más estables y precisas.

La estrategia recomendada es siempre entrar primero con regresión Ridge como referencia de mínimo rendimiento, y usar Random Forest o XGBoost solo si la mejora en MAE es al menos un 10-15%. Si no lo supera, la regresión es preferible por su mayor interpretabilidad.

### ¿Por qué Python y no R?

R es un lenguaje estadístico poderoso, pero para este problema Python tiene ventajas concretas. Primero, Python ofrece un ecosistema unificado: pandas, scikit-learn, XGBoost, SHAP y comorbidipy comparten la misma interfaz de objetos (`.fit()`, `.predict()`, `.transform()`), lo que reduce la fricción entre pasos del pipeline. En R, cada librería tiene su propia sintaxis. Segundo, para despliegue en producción (que el modelo corra automáticamente al ingresar un nuevo paciente), Python es el estándar de la industria; un modelo guardado con `joblib` puede integrarse con cualquier sistema de información hospitalaria. Tercero, todas las librerías requeridas son completamente gratuitas, de código abierto, y con comunidades activas de soporte.

### ¿Por qué transformar el LOS con log(LOS + 1)?

El LOS hospitalario tiene distribución fuertemente asimétrica: la mayoría de los pacientes está entre 1 y 7 días, pero unos pocos tienen estadías de 60 o 90 días. Estos valores extremos se llaman **outliers**. Al entrenar el modelo con el LOS original, la función de error (que el modelo minimiza) penaliza desproporcionadamente los errores en esos casos extremos, obligando al modelo a destinar mucho de su capacidad a predecirlos bien a costa de volverse impreciso para el grueso de los pacientes.

La transformación $\log(LOS + 1)$ comprime la escala: un LOS de 90 días se convierte en $\log(91) \approx 4.5$ y un LOS de 7 días en $\log(8) \approx 2.1$. La diferencia que antes era de 83 días ahora es de 2.4 unidades, y el modelo asigna importancias más equilibradas. El $+1$ evita el error matemático de $\log(0)$ cuando LOS = 0. Al reportar resultados, se revierte con $e^{\hat{y}} - 1$ (función `numpy.expm1`).

### ¿Qué es Gradient Boosting y cómo ayuda?

Gradient Boosting construye árboles de decisión en secuencia: el primer árbol predice el LOS lo mejor que puede, el segundo árbol aprende exclusivamente de los errores del primero, el tercero de los errores del segundo, y así sucesivamente. La palabra "gradient" (gradiente) viene del cálculo diferencial: cada árbol nuevo se construye en la dirección que más reduce el error, que es exactamente lo que el gradiente de la función de pérdida indica. Es optimización iterativa aplicada al aprendizaje automático. XGBoost y LightGBM son implementaciones industriales de este método, altamente optimizadas y gratuitas.

### ¿Para qué sirve SHAP?

SHAP resuelve el problema de la "caja negra": un Random Forest o XGBoost puede ser muy preciso, pero es difícil entender por qué predijo un LOS específico para un paciente. SHAP calcula la contribución exacta de cada variable a cada predicción individual, usando la teoría de valores de Shapley de la teoría de juegos (que respondió a la pregunta: si varios jugadores colaboran para ganar un premio, ¿cuánto le corresponde a cada uno?). Con SHAP se puede decirle al médico: "el modelo predice 12 días para este paciente porque su Índice de Charlson de 5 añade 3 días sobre el promedio, y tener 6 procedimientos añade 2 días más".

### ¿Para qué sirve sklearn.feature_extraction?

Esta librería convierte texto en matrices numéricas. Específicamente, `TfidfVectorizer` toma las listas de códigos ICD-10 de cada paciente (que son texto) y produce un vector numérico que representa la importancia relativa de cada código en ese paciente comparado con toda la población. Es la herramienta que permite que los modelos procesen los diagnósticos sin necesidad de ninguna API ni diccionario externo.

### ¿Cómo se determina la severidad sin traducir los códigos manualmente?

Existen dos mecanismos. El primero y más importante es el **Índice de Comorbilidad de Charlson (CCI)**, una herramienta médica validada desde 1987 que asigna pesos numéricos a cada diagnóstico ICD-10 según su impacto en complicaciones y mortalidad. La librería `comorbidipy` lo calcula automáticamente en Python. El segundo mecanismo es la estructura jerárquica propia del código ICD-10: la primera letra ya identifica el sistema orgánico afectado, y los primeros tres caracteres identifican el bloque de enfermedades. Extraer esa información con operaciones de texto en pandas ya produce features con significado clínico sin ninguna API.

---

## 13. Stack tecnológico completo

Todas las librerías son gratuitas y se instalan con `pip install`.

| Librería | Versión mínima | Propósito |
|---|---|---|
| `pandas` | 1.5+ | Manipulación de DataFrames, lectura de Excel, operaciones de texto |
| `numpy` | 1.23+ | Operaciones numéricas, matrices, transformación logarítmica |
| `matplotlib` | 3.5+ | Visualizaciones estáticas (histogramas, scatter plots) |
| `seaborn` | 0.12+ | Visualizaciones estadísticas de mayor nivel (barplots, heatmaps) |
| `scikit-learn` | 1.1+ | Modelos Ridge/Lasso/Random Forest, preprocesamiento, TF-IDF, métricas |
| `xgboost` | 1.7+ | Gradient Boosting de alta performance |
| `lightgbm` | 3.3+ | Alternativa más rápida a XGBoost para datasets grandes |
| `shap` | 0.41+ | Explicabilidad de predicciones individuales |
| `comorbidipy` | última | Índice de Charlson e Índice de Elixhauser desde ICD-10 |
| `scipy` | 1.9+ | Tests estadísticos (correlaciones de Pearson, tests de normalidad) |
| `joblib` | 1.2+ | Serialización y guardado de modelos entrenados |

Instalación completa en un solo comando:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap comorbidipy scipy joblib openpyxl
```

> El paquete `openpyxl` es necesario para que `pandas` pueda leer archivos `.xlsx`.

---

## 14. Preguntas clave al cliente

Estas preguntas deben resolverse antes de comenzar la implementación, ya que algunas pueden cambiar decisiones de diseño fundamentales.

### Sobre los datos

**¿Cuántos pacientes tiene la base de datos?** El tamaño total del dataset determina qué tan complejos pueden ser los modelos y si se necesitan técnicas adicionales para datasets pequeños (como leave-one-out cross-validation en lugar de k-fold).

**¿Los diagnósticos son de admisión o de egreso?** Si son diagnósticos de egreso, el modelo tiene fuga de información (se estarían usando datos que solo se conocen después de que ocurrió el LOS que se pretende predecir). Para un sistema de predicción en tiempo real solo son válidos los diagnósticos disponibles al momento de la admisión.

**¿Hay pacientes censados?** Es decir, ¿hubo pacientes que aún estaban hospitalizados cuando se cerró la base de datos y su LOS aparece truncado? Si existe censura, el modelo lineal está sesgado y se necesitarían modelos de supervivencia.

**¿Hay IDs duplicados o reutilizados?** Un ID que aparece en dos hospitalizaciones distintas puede indicar reingreso del mismo paciente o reutilización del identificador.

### Sobre el objetivo de uso

**¿El modelo predice al momento de la admisión o días después?** Esto define qué variables son legítimas como inputs.

**¿Se necesita una predicción puntual o un intervalo?** Si el cliente necesita saber "entre 5 y 9 días con 80% de probabilidad", se requieren modelos de cuantiles o intervalos de predicción.

**¿Qué error máximo es clínicamente aceptable?** Un MAE de 1.5 días puede ser excelente para planificación de camas pero insuficiente para gestión de UCI de alta rotación.

**¿Quién usará el modelo?** Si lo usarán médicos directamente, la interpretabilidad (SHAP) es crítica. Si lo consumirá un sistema automático, puede priorizarse precisión pura.

---

*Documento preparado como guía de implementación del proyecto de predicción de LOS hospitalario. Todas las decisiones técnicas están fundamentadas en la sección 12. El pipeline completo está diseñado para ejecutarse de forma secuencial desde la etapa A hasta la H.*
