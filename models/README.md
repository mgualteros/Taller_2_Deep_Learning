# Modelos entrenados

Este directorio contiene los modelos entrenados para análisis de sentimientos sobre tweets, cada uno guardado en formato `.h5` después de su respectivo proceso de entrenamiento.

## Modelos disponibles

### 🧠 `rnn_model.h5`
Modelo secuencial basado en una Red Neuronal Recurrente (RNN) simple. 
- Estructura: Embedding + SimpleRNN + Dense
- Es ideal para entender cómo se comporta un modelo básico en tareas de texto secuencial.

### 🔁 `lstm_model.h5`
Modelo secuencial profundo basado en LSTM (Long Short-Term Memory).
- Estructura: Embedding + LSTM (doble capa) + Dense
- Mejora la capacidad de capturar dependencias a largo plazo en el texto respecto a la RNN simple.

### 🔁🧠 `bilstm_attention_model.h5`
Modelo avanzado basado en una arquitectura BiLSTM con mecanismo de atención.
- Estructura: Embedding + BiLSTM + Atención + Dense
- Permite capturar el contexto en ambas direcciones del texto y enfocar la atención en las partes más relevantes de los tweets.

---

Cada modelo fue entrenado con el mismo conjunto de datos, tokenizado y balanceado, permitiendo comparaciones justas de rendimiento.

