Integrantes:
Sebastián Heredia       Alejandra Bolívar 
Michael Gualteros       Herny Herrera
Juan Rodriguez

# Análisis de Sentimientos en Tweets con Deep Learning

Este proyecto implementa y compara diferentes arquitecturas de redes neuronales para análisis de sentimientos en tweets, haciendo especial énfasis en el tratamiento del desbalanceo de clases y en la capacidad de cada modelo para clasificar correctamente sentimientos positivos y negativos.

---

El repositorio de este proyecto es: https://github.com/mgualteros/Taller_2_Deep_Learning

## Contexto del problema

El conjunto de datos presentaba un **importante desbalanceo**, donde apenas el **7%** de los ejemplos pertenecía a la clase **positiva (1)**. Además, se encontraron:

- Palabras irrelevantes o sin sentido en el contexto de su categoría. (amor y felicidad lo consideraba palabras negativas y racismo como positivo)
- Etiquetado incorrecto en algunos casos.

Para abordar esto:

- Se usó `RandomOverSampler` para aplicar **oversampling** a la clase minoritaria.
- Se separaron datos en entrenamiento y prueba asegurando balance.

---

## Requisitos y dependencias

pip install -r requirements.txt

==============================
DESCRIPCIÓN DE ARCHIVOS .py
==============================

1. data_loader.py
------------------
  Prepara los datos para entrenamiento y evaluación.
  - Carga los datos.
  - Tokeniza los textos.
  - Aplica padding a las secuencias.
  - Hace oversampling para balancear las clases.

  Devuelve Tuplas con datos listos para usar: (X_train, y_train), (X_test, y_test), tokenizer.

2. model_rnn.py
------------------
  Construye un modelo de análisis de sentimientos usando RNN básico.
- Define una red neuronal simple con capa `SimpleRNN`.
- Devuelve un modelo Keras compilado listo para entrenar.

3. model_lstm.py
------------------
- Construye un modelo LSTM clásico para análisis de sentimientos.
- Usa capas `Embedding`, `LSTM` y `Dense` para crear un modelo profundo.
- Devuelve un modelo Keras compilado.

4. model_bilstm_attention.py
------------------
- Construye un modelo BiLSTM con atención, más robusto.
- Usa capas bidireccionales LSTM.
- Agrega un mecanismo de atención para enfocar la red en las palabras relevantes.
- Devuelve un modelo Keras compilado.

5. train.py
------------------
-  Entrenar cualquiera de los modelos definidos.
-  Usa los datos y modelo para entrenar durante varias épocas.
-  Devuelve un objeto `history` del entrenamiento (historial de pérdida y precisión).

6. evaluate.py
------------------
- Evalua un modelo previamente entrenado.
- Carga el modelo `.h5`.
- Evalúa sobre el conjunto de prueba.
- Devuelve el loss y accuracy del modelo evaluado

7. utils.py
------------------
- Contine funciones utilitarias para visualizar y analizar resultados.
- mostrar_curvas_perdida(history): muestra gráficamente la pérdida de entrenamiento y validación.
  - mconfusion(modelo, x_test, y_test, umbral=0.5): imprime el reporte de clasificación y una matriz de confusión con visualización.
  - ver_resumen_modelo(modelo): imprime el resumen (summary) de la arquitectura del modelo.
  - Devuelve gráficas y salidas útiles para interpretación.