# Plan de Implementación: Modelos de Regresión Lineal Regularizada con Interacciones y Aplicación Web
**Estrategia de Desarrollo e Integración del Modelo Base y la Solución de Software**

Este plan de implementación detalla la estrategia metodológica, las especificaciones del modelo base y el desarrollo del servidor de inferencia clínica, incorporando regularizaciones L2 y términos de interacción validados empíricamente para responder a las observaciones sobre comparabilidad lineal.

---

## Contexto y Justificación Metodológica de las Interacciones

> [!IMPORTANT]
> **Sobre la comparabilidad lineal:**
> Un análisis estadístico profundo (detallado en el documento [analisis_critico_interacciones.md](file:///c:/Users/tomas/Downloads/E3_Capstone/Capstone-Grupo-16--main/Modelo_Base_Ultima%20entrega/analisis_critico_interacciones.md)) demuestra que incorporar todas las interacciones de segundo orden de forma cruda en una regresión lineal es numéricamente inviable, ya que generaría más de $1.36$ millones de variables para solo $11,951$ observaciones, provocando el colapso del estimador OLS.
> 
> Para responder con la máxima rigurosidad técnica, se adoptaron dos soluciones integrales en el diseño del modelo base:
> 1.  **Interacciones Estructurales vía Separación de Datos:** La segmentación del dataset en modelos de **Urgencia** y **No Urgencia** equivale a incluir interacciones completas de todas las $1,650$ variables con la variable `es_urgencia` ($X_j \times \text{es\_urgencia}$).
> 2.  **Interacciones Clínicas Explícitas:** Se testearon y evaluaron individualmente 3 términos de interacción continuos sobre los datos reales. La especificación final es:
>     *   **Modelo de Urgencias:** Se integran las 3 interacciones por ser altamente significativas ($p < 0.001$):
>         *   `charlson_index` $\times$ `n_diag_total`
>         *   `n_procedimientos` $\times$ `n_diag_total`
>         *   `charlson_index` $\times$ `n_procedimientos`
>     *   **Modelo de No Urgencias:** Se integran solo 2 interacciones (descartando `n_procedimientos * n_diag_total` por no ser significativa, $p = 0.55$):
>         *   `charlson_index` $\times$ `n_diag_total`
>         *   `charlson_index` $\times$ `n_procedimientos`
> 
> Esto proporciona un benchmark lineal estable, matemáticamente comparable y robusto.

---

## Cambios Propuestos

### Componente: Modelos de Regresión Lineal Regularizada (Ridge) con Interacciones

#### [NEW] [entrenar_lr_urgencias_no_urgencias.py](file:///c:/Users/tomas/Downloads/E3_Capstone/Capstone-Grupo-16--main/Modelo_Base_Ultima%20entrega/entrenar_lr_urgencias_no_urgencias.py)

Script en Python para el entrenamiento y validación de los modelos lineales base con regularización Ridge:
*   **Carga de datos:** Lee el dataset maestro del Escenario B (`model_data_v3_escenario_B_charlson.csv`).
*   **Cálculo de Interacciones:** Genera las variables de interacción de forma dinámica.
*   **Separación:** Divide el DataFrame según el tipo de ingreso: `df_urg` (`es_urgencia == 1`) y `df_no_urg` (`es_urgencia == 0`).
*   **Entrenamiento Independiente:**
    *   **Urgencias:** Entrena con las 3 interacciones utilizando regularización Ridge ($\alpha = 100.0$).
    *   **No Urgencias:** Entrena con 2 interacciones utilizando regularización Ridge ($\alpha = 50.0$).
    *   Para cada subgrupo, realiza partición 80/20 train/test estratificada por tramos de LOS.
    *   Corre validación cruzada de 5 pliegues (StratifiedKFold) en el train set usando la variable transformada `log1p(los_dias)`.
    *   Entrena el modelo final de `Ridge` de scikit-learn.
*   **Evaluación y Exportación:**
    *   Evalúa en test set revirtiendo con la exponencial inversa (`expm1`) y calcula MAE, RMSE, MedAE y métricas de clasificación PLOS $\ge 27$.
    *   Exporta los archivos pickle: `lr_base_Urgencias.pkl` y `lr_base_No_Urgencias.pkl` a la carpeta de entrega.
    *   Exporta las métricas de validación y predicciones en formato CSV.

---

### Componente: Aplicación Web de Inferencia Clínica

#### [NEW] [app.py](file:///c:/Users/tomas/Downloads/E3_Capstone/Capstone-Grupo-16--main/Web/app.py)
Servidor backend en Flask responsable de gestionar los modelos en producción:
*   Carga dinámicamente en memoria los modelos entrenados (`lr_base_Urgencias.pkl` y `lr_base_No_Urgencias.pkl`) y los modelos avanzados.
*   Construye las variables de entrada a partir de la información del paciente.
*   Calcula dinámicamente el Índice de Charlson usando la librería `comorbidipy`.
*   Aplica la predicción correspondiente según el tipo de admisión y modelo seleccionado.

#### [NEW] [templates/](file:///c:/Users/tomas/Downloads/E3_Capstone/Capstone-Grupo-16--main/Web/templates/)
Interfaz de usuario de la plataforma web:
*   **prediccion:** Formulario interactivo que permite añadir diagnósticos (códigos ICD-10-CM) y procedimientos (códigos ICD-10-PCS).
*   **explicabilidad:** Muestra en tiempo real los "Factores Clave de Estancia" calculados dinámicamente a partir de los coeficientes de regresión del modelo Ridge.
*   **comparación:** Permite alternar la estimación entre la Regresión Ridge y los modelos avanzados (XGBoost/Random Forest).

---

## Plan de Verificación

### Pruebas Automatizadas
1.  **Ejecutar entrenamiento de baselines:**
    ```powershell
    python "Modelo_Base_Ultima entrega/entrenar_lr_urgencias_no_urgencias.py"
    ```
    *   Verificar que se crean los pickles correspondientes en `Modelo_Base_Ultima entrega/`.
    *   Confirmar la correcta escritura de los reportes de validación cruzada y test en CSV.
2.  **Iniciar Servidor de Aplicación Web:**
    ```powershell
    python Web/app.py
    ```
    *   Acceder localmente a `http://127.0.0.1:5000/`.
    *   Validar la funcionalidad de predicción individual, de carga masiva (bulk) y el cálculo dinámico de Charlson.
