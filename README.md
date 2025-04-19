Integrantes:
Sebastián Heredia       Alejandra Bolívar 
Michael Gualteros       Herny Herrera
Juan Rodriguez

# 🧠 Análisis de Sentimientos en Tweets con Deep Learning

Este proyecto implementa y compara diferentes arquitecturas de redes neuronales para análisis de sentimientos en tweets, haciendo especial énfasis en el tratamiento del desbalanceo de clases y en la capacidad de cada modelo para clasificar correctamente sentimientos positivos y negativos.

---

## 📉 Contexto del problema

El conjunto de datos presentaba un **importante desbalanceo**, donde apenas el **7%** de los ejemplos pertenecía a la clase **positiva (1)**. Además, se encontraron:

- Palabras irrelevantes o sin sentido en el contexto de su categoría. (amor y felicidad lo consideraba palabras negativas y racismo como positivo)
- Etiquetado incorrecto en algunos casos.

Para abordar esto:

- Se aplicó **limpieza y preprocesamiento** del texto.
- Se usó `RandomOverSampler` para aplicar **oversampling** a la clase minoritaria.
- Se separaron datos en entrenamiento y prueba asegurando balance.

---

## ⚙️ Requisitos y dependencias

Es recomendable usar un entorno virtual. Instala las dependencias con:

pip install -r requirements.txt

--------------------------------------------

PASOS PARA EJECUTAR EL PROYECTO DE ANÁLISIS DE SENTIMIENTOS EN TWITTER

 INSTALAR DEPENDENCIAS
-------------------------
pip install -r requirements.txt
