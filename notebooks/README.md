# Análisis de Sentimientos en Twitter con Redes Neuronales Recurrentes

Este repositorio contiene una serie de notebooks organizados en la carpeta `notebooks/` que abordan un flujo completo de trabajo para el análisis de sentimientos en tweets utilizando diferentes arquitecturas de redes neuronales. El dataset usado proviene del repositorio abierto de análisis de sentimientos de Twitter.

## Estructura del Proyecto

### notebooks/
Contiene los notebooks principales del proyecto:

#### `01_exploracion.ipynb`
Este notebook realiza una exploración inicial del dataset, incluyendo:
- Carga de datos.
- Limpieza básica de texto.
- Visualización de la distribución de clases.
- Análisis preliminar de la longitud de los tweets y palabras más comunes.

#### `02_entrenamiento_RNN.ipynb`
Se implementa y entrena una red neuronal recurrente (RNN) básica para clasificación binaria de sentimientos. Incluye:
- Preprocesamiento de texto.
- Tokenización y padding.
- Construcción y entrenamiento de una RNN simple.
- Evaluación del rendimiento del modelo.

#### `03_entrenamiento_LSTM.ipynb`
Este notebook entrena una red LSTM (Long Short-Term Memory), que mejora la capacidad de la red para capturar dependencias a largo plazo en el texto. Contiene:
- Arquitectura LSTM con Embedding.
- Mejora de métricas respecto a la RNN básica.
- Comparación de resultados con la versión anterior.

#### `04_BiLSTM_atencion.ipynb`
Aquí se implementa un modelo más avanzado utilizando:
- Una arquitectura Bidireccional LSTM (BiLSTM).
- Un mecanismo de atención personalizado que permite al modelo enfocarse en las partes más relevantes del tweet.
- Evaluación del desempeño y comparación con modelos anteriores.

