Integrantes:
Sebastián Heredia       Alejandra Bolívar 
Michael Gualteros       Herny Herrera
Juan Rodriguez

# Análisis de Sentimientos en Tweets con Deep Learning

Este proyecto implementa y compara modelos de redes neuronales recurrentes para análisis de sentimiento en textos cortos (tweets). Utiliza un conjunto de datos público con más de 30,000 tweets etiquetados como positivos o negativos. El objetivo es identificar cuál arquitectura logra mejor desempeño en la predicción del sentimiento.
---

El repositorio de este proyecto es: https://github.com/mgualteros/Taller_2_Deep_Learning

## Contexto del problema

El conjunto de datos presentaba un **importante desbalanceo**, donde apenas el **7%** de los ejemplos pertenecía a la clase **positiva (1)**. Además, se encontraron:

- Palabras irrelevantes o sin sentido en el contexto de su categoría. ("love" y "hhapy" lo consideraba palabras negativas y "racism" como positivo)
- Etiquetado incorrecto en algunos casos.

Para abordar esto:

- Se usó `RandomOverSampler` para aplicar **oversampling** a la clase minoritaria.
- Se separaron datos en entrenamiento y prueba asegurando balance.

===================================
RESULTADOS OBTENIDOS Y CONCLSUIONES
===================================

1. Resultados obtenidos con el Modelo RNN
   ------------------
El modelo RNN fue entrenado con los datos de entrenamiento y probado en los datos de test. Durante la evaluación, observamos lo siguiente:

Precisión (Accuracy): El modelo RNN alcanzó una precisión superior al 95%, lo que indica que pudo clasificar correctamente la mayoría de los tweets, aunque las redes recurrentes simples no son tan efectivas a largo plazo.
Pérdida (Loss): La pérdida fue relativamente baja, lo que muestra que el modelo minimizó el error durante el entrenamiento.
Análisis: Aunque el rendimiento del modelo fue bueno, los resultados de la RNN generalmente no son tan sólidos cuando se trata de capturar dependencias a largo plazo o patrones complejos en los datos, especialmente en tareas de clasificación de texto.
Tendencia de clasificación: El modelo mostró una ligera tendencia a clasificar los tweets negativos con mayor precisión, lo que puede ser un punto a favor dependiendo del enfoque de clasificación que se busque.


2. Resultados obtenidos con el Modelo LSTM
   ------------------
El modelo LSTM, al ser más complejo y especializado en manejar dependencias de largo plazo, mostró inicialmente un alto desempeño en términos de precisión, pero tras un análisis más profundo, se identificaron problemas relacionados con el desbalanceo de los datos:

Precisión (Accuracy): El modelo LSTM alcanzó una precisión de aproximadamente 93%. Aunque esta cifra parece alta, es importante destacar que el modelo no clasificó ningún tweet positivo correctamente. Esto sugiere que el modelo se sesgó hacia la clase negativa debido al desbalance de clases en el dataset, donde los tweets negativos eran más frecuentes que los positivos.
Pérdida (Loss): La pérdida fue relativamente baja, pero dado el sesgo hacia la clasificación negativa, la baja pérdida no refleja un buen desempeño en la clasificación de ambas clases.
Análisis: El modelo LSTM, aunque tiene la capacidad de capturar dependencias a largo plazo en las secuencias de texto, falló en la clasificación adecuada de los tweets positivos. La precisión de 93% no es una indicación de buen rendimiento en este caso, ya que una precisión alta puede estar siendo arrastrada por la alta tasa de aciertos en la clasificación de tweets negativos.
Tendencia de clasificación: El modelo LSTM mostró una tendencia a clasificar todos los tweets como negativos, lo que refuerza la hipótesis de que el desbalance de clases afectó su rendimiento. No hubo clasificación correcta de tweets positivos.

3. Resultados obtenidos con el Modelo BiLSTM con Atención
   ------------------

El modelo BiLSTM con Atención es el más avanzado entre los tres, tomo más tiempo de procesamiento, al incorporar tanto una LSTM bidireccional como un mecanismo de atención para identificar las partes relevantes de los textos:

Precisión (Accuracy): Este modelo alcanzó una precisión superior al 95%, similar a los otros modelos, pero con la ventaja de que su arquitectura de atención permitió que el modelo se enfocara en las partes más relevantes de cada tweet, mejorando su capacidad para clasificar correctamente los tweets.
Pérdida (Loss): El modelo mostró una pérdida muy baja, reflejando una excelente capacidad para generalizar y hacer predicciones precisas.
Análisis: La principal ventaja del BiLSTM con Atención sobre los otros modelos es su capacidad para capturar información contextual desde ambas direcciones de la secuencia (izquierda y derecha). Además, la capa de atención permite que el modelo se enfoque en las palabras clave de los tweets, lo que mejora la precisión en casos más complejos donde los patrones sentimentales no están presentes de manera lineal.
Tendencia de clasificación: El BiLSTM con Atención logró un buen desempeño en la clasificación tanto de tweets positivos como negativos, sin presentar una inclinación significativa hacia uno u otro. Su capacidad para centrarse en las palabras clave le permitió lograr una clasificación más precisa.

------------------------------------
Análisis Comparativo de los Modelos:
------------------------------------

Modelo	Precisión (Accuracy)	Pérdida (Loss)	Observaciones
RNN	> 95%	Baja	Mejor desempeño con tweets negativos. Menos efectivo a largo plazo.
LSTM	93%	Baja	Modelo sesgado hacia la clase negativa debido al desbalance de clases. No clasifica bien los tweets positivos.
BiLSTM con Atención	> 95%	Muy baja	Mejor precisión, enfocándose en partes clave de los tweets. Mejor para manejar contexto y relaciones complejas.

------------
Conclusiones
------------

RNN y LSTM: Aunque ambos modelos alcanzaron una precisión alta, el LSTM sufrió de un sesgo hacia la clase negativa debido al desbalance en los datos. Esto hizo que el modelo no fuera eficaz para clasificar correctamente los tweets positivos, lo que resalta la importancia de considerar el desbalance de clases en el entrenamiento.

BiLSTM con Atención: Este modelo no solo superó el desbalance de clases de manera más efectiva gracias a su arquitectura de atención, sino que también mostró un rendimiento equilibrado, con una buena clasificación tanto de los tweets negativos como positivos.

=======================
DESCRIPCIÓN DE MODELOS
=======================

1. Modelo RNN
------------------
El modelo RNN está diseñado para procesar secuencias de texto y clasificarlas en dos categorías: positivas o negativas. Para este modelo, utilizamos una red neuronal recurrente (RNN) que captura dependencias temporales en las secuencias de texto.

Arquitectura:
Capa de Embedding: La capa de embedding tiene una dimensión de 128, lo que significa que cada palabra del vocabulario se representa como un vector de 128 dimensiones. El número total de palabras en el vocabulario es de max_features (valor que se define al cargar los datos).
Capa SimpleRNN: Esta capa tiene 64 unidades, lo que permite que la RNN procese la secuencia de texto y capture dependencias temporales.
Capa de salida (Dense): La capa final es una capa densa con 1 unidad y una activación sigmoid, lo que nos da la predicción binaria (positivo o negativo) para cada tweet.

Parámetros clave:
Tamaño de las secuencias: input_len (longitud máxima de las secuencias de entrada).
Tamaño del vocabulario: vocab_size (determinado por la tokenización).
Función de activación: sigmoid para clasificación binaria.

2. Modelo LSTM
------------------
El modelo LSTM se utiliza para capturar dependencias a largo plazo en los datos de texto, gracias a la capacidad de las LSTM de manejar el desvanecimiento del gradiente. Este modelo tiene una arquitectura con dos capas LSTM que procesan las secuencias en dos niveles.

Arquitectura:
Capa de Embedding: Similar al modelo RNN, la capa de embedding tiene una dimensión de 128, lo que significa que cada palabra se representa con un vector de 128 dimensiones.
Primera capa LSTM: Tiene 64 unidades y devuelve secuencias para pasar la información a la siguiente capa LSTM.
Segunda capa LSTM: Esta capa tiene 32 unidades, y sus salidas se usan para realizar la predicción.
Capas Dropout: Se aplican para evitar el sobreajuste. La tasa de Dropout es del 50% en la primera capa, 40% en la segunda capa y 30% en la capa densa.
Capa de salida (Dense): Esta capa tiene 1 unidad con una activación sigmoid para hacer la clasificación binaria.

Parámetros clave:
Tamaño de las secuencias: input_len (longitud máxima de las secuencias de entrada).
Tamaño del vocabulario: vocab_size.
Función de activación: sigmoid para la clasificación binaria.

3. Modelo BiLSTM con Atención
------------------
El modelo BiLSTM con atención utiliza LSTM bidireccionales junto con un mecanismo de atención para mejorar la capacidad del modelo de centrarse en las partes más relevantes del texto. Esto ayuda a mejorar la precisión al procesar secuencias de texto de manera más eficiente.

Arquitectura:
Capa de Embedding: Similar a los otros modelos, la capa de embedding tiene una dimensión de 128, representando cada palabra como un vector de 128 dimensiones.
Capa BiLSTM: Esta capa tiene 64 unidades y es bidireccional, lo que significa que el modelo procesa las secuencias de texto en ambas direcciones (de izquierda a derecha y de derecha a izquierda).
Capa de atención: Calcula los pesos de atención para cada paso temporal de la secuencia. Esto permite que el modelo se enfoque en las partes más importantes del tweet.
Lambda: Realiza una reducción a lo largo de la secuencia para generar un vector de contexto a partir de las salidas ponderadas por la atención.
Capa de salida (Dense): Tiene 1 unidad con activación sigmoid para realizar la clasificación binaria.

Parámetros clave:
Tamaño de las secuencias: ventana (longitud máxima de las secuencias de entrada, igual al input_len en los otros modelos).
Tamaño del vocabulario: vocab_size.
Función de activación: sigmoid para la clasificación binaria.

============================
DESCRIPCIÓN DE ARCHIVOS .py
============================

1. data_loader.py
------------------
  - Prepara los datos para entrenamiento y evaluación.
  - Carga los datos.
  - Tokeniza los textos.
  - Aplica padding a las secuencias.
  - Hace oversampling para balancear las clases.
  - Devuelve Tuplas con datos listos para usar: (X_train, y_train), (X_test, y_test), tokenizer.
------------------
2. model_rnn.py
------------------
  - Construye un modelo de análisis de sentimientos usando RNN básico.
  - Define una red neuronal simple con capa `SimpleRNN`.
  - Devuelve un modelo Keras compilado listo para entrenar.
------------------
3. model_lstm.py
------------------
  - Construye un modelo LSTM clásico para análisis de sentimientos.
  - Usa capas `Embedding`, `LSTM` y `Dense` para crear un modelo profundo.
  - Devuelve un modelo Keras compilado.
------------------
4. model_bilstm_attention.py
------------------
  - Construye un modelo BiLSTM con atención, más robusto.
  - Usa capas bidireccionales LSTM.
  - Agrega un mecanismo de atención para enfocar la red en las palabras relevantes.
  - Devuelve un modelo Keras compilado.
------------------
5. train.py
------------------
  -  Entrenar cualquiera de los modelos definidos.
  -  Usa los datos y modelo para entrenar durante varias épocas.
  -  Devuelve un objeto `history` del entrenamiento (historial de pérdida y precisión).
------------------
6. evaluate.py
------------------
  - Evalua un modelo previamente entrenado.
  - Carga el modelo `.h5`.
  - Evalúa sobre el conjunto de prueba.
  - Devuelve el loss y accuracy del modelo evaluado
------------------
7. utils.py
------------------
  - Contine funciones utilitarias para visualizar y analizar resultados.
  - mostrar_curvas_perdida(history): muestra gráficamente la pérdida de entrenamiento y validación.
  - mconfusion(modelo, x_test, y_test, umbral=0.5): imprime el reporte de clasificación y una matriz de confusión con visualización.
  - ver_resumen_modelo(modelo): imprime el resumen (summary) de la arquitectura del modelo.
  - Devuelve gráficas y salidas útiles para interpretación.

===================================================================
Instrucciones para ejecutar el proyecto de análisis de sentimientos
===================================================================

A continuación se describen los pasos para ejecutar el código correctamente.

1. Clonar el repositorio
   ------------------

Primero, clona el repositorio o descarga el proyecto desde el origen para tener acceso a todos los archivos necesarios.

git clone <url_del_repositorio>


2. Requisitos previos
   ------------------

Antes de ejecutar el proyecto, asegúrate de tener las siguientes dependencias instaladas. Puedes instalar los requerimientos usando el archivo requirements.txt:

pip install -r requirements.txt

Esto instalará las librerías necesarias para el procesamiento de datos, entrenamiento de los modelos y evaluación de resultados.

3. Estructura del proyecto
   ------------------

La estructura del proyecto es la siguiente:

mi_proyecto_sentimiento/
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_entrenamiento_RNN.ipynb
│   ├── 03_entrenamiento_LSTM.ipynb
│   └── 04_BiLSTM_atencion.ipynb
├── README.md
├── src/
│   ├── data_loader.py
│   ├── model_rnn.py
│   ├── model_lstm.py
│   ├── model_bilstm_attention.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/
│   ├── rnn_model.h5
│   ├── lstm_model.h5
│   ├── bilstm_attention_model.h5
├── requirements.txt

Los scripts .py contienen el código fuente de los modelos y el preprocesamiento de datos. Los notebooks se utilizan para la exploración y el entrenamiento de los modelos.

4. Cargar y preparar los datos
   ------------------

Para cargar y preparar los datos, puedes usar la función load_and_prepare_data desde el archivo data_loader.py. Esta función realiza los siguientes pasos:

- Carga el dataset de tweets.
- Realiza la tokenización y el padding de las secuencias.
- Realiza un split en conjuntos de entrenamiento y prueba.
- Aplica oversampling a los datos para corregir el desbalanceo de clases.

Ejemplo de uso:

from src.data_loader import load_and_prepare_data

url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
x_resampled, y_resampled, x_test, y_test, tokenizer = load_and_prepare_data(url)

5. Construcción del modelo
   ------------------

Cada modelo (RNN, LSTM, BiLSTM con atención) tiene su propia función de construcción en el archivo correspondiente:

- model_rnn.py: Define el modelo RNN para clasificación de sentimientos.
- model_lstm.py: Define el modelo LSTM.
- model_bilstm_attention.py: Define el modelo BiLSTM con atención.

Ejemplo de cómo construir un modelo BiLSTM con atención:

from src.model_bilstm_attention import build_bilstm_attention_model

model = build_bilstm_attention_model(ventana=300, vocab_size=10000)

6. Entrenamiento del modelo
   ------------------

Una vez construido el modelo, puedes entrenarlo usando la función train_model, que recibe el modelo, los datos de entrenamiento y los parámetros como batch_size, epochs y val_split.

Ejemplo de entrenamiento:

from src.train import train_model

history = train_model(model, x_resampled, y_resampled, batch_size=64, epochs=10, val_split=0.2)

7. Evaluación del modelo
   ------------------

Después de entrenar el modelo, puedes evaluarlo usando la función evaluate_model, que devuelve las métricas de pérdida y precisión.

Ejemplo de evaluación:

from src.evaluate import evaluate_model

loss, accuracy = evaluate_model(model, x_test, y_test)

8. Reporte y matriz de confusión
   ------------------

Para generar un reporte de clasificación y la matriz de confusión, puedes utilizar la función mconfusion desde el archivo utils.py. Esta función genera el reporte y visualiza la matriz de confusión.

Ejemplo de uso:

from src.utils import mconfusion

mconfusion(model, x_test, y_test)

9. Visualización de curvas de pérdida
   ------------------

Para visualizar las curvas de pérdida durante el entrenamiento, puedes usar la función grafica_perdida también en el archivo utils.py. Esta función toma el historial del entrenamiento y genera el gráfico.

Ejemplo de uso:

from src.utils import grafica_perdida

grafica_perdida(history)

10. Guardar y cargar el modelo
    ------------------
    
El modelo entrenado se guarda en la carpeta models en formato .h5 y se puede cargar posteriormente para realizar predicciones o reentrenamiento.

Guardar el modelo:

model.save('models/bilstm_attention_model.h5')

Cargar el modelo:

from tensorflow.keras.models import load_model

model = load_model('models/bilstm_attention_model.h5')

11. Resultados y análisis comparativo
    ------------------

En cuanto a los resultados obtenidos:

- RNN: mostró un buen desempeño en la clasificación de tweets negativos.
- LSTM: clasificó casi todo como negativo, lo que indica que las métricas no son completamente adecuadas para este modelo.
- BiLSTM con atención: mostró un excelente desempeño, especialmente clasificando correctamente los tweets positivos con una precisión superior al 95%.
