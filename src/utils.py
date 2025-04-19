import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def plot_loss_curves(history):
    """
    Grafica las curvas de pérdida durante el entrenamiento y validación.

    Parámetros:
    - history: objeto History devuelto por model.fit()
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Pérdida entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Historial de Pérdida durante el Entrenamiento')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def mconfusion(model, x_test, y_test, threshold=0.5):
    """
    Genera un reporte de clasificación y visualiza la matriz de confusión para un modelo de clasificación binaria.

    Esta función permite evaluar el rendimiento del modelo utilizando un umbral definido para la clase positiva.
    Muestra métricas como precisión, recall, F1-score y una matriz de confusión en formato gráfico.

    Parámetros:
    - model: modelo Keras entrenado.
    - x_test: datos de prueba (features).
    - y_test: etiquetas reales de los datos de prueba.
    - threshold: umbral de probabilidad para predecir la clase positiva (default: 0.5).

    Retorna:
    - Nada. Muestra el reporte por consola y la matriz de confusión como gráfico.
    """
    y_pred_probs = model.predict(x_test)
    y_pred = (y_pred_probs > threshold).astype("int32")

    print("Reporte de clasificación:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

def ver_resumen_modelo(modelo):
    """
    Muestra el resumen de arquitectura del modelo Keras.

    Parámetro:
    - modelo: modelo Keras (RNN, LSTM, BiLSTM, etc.)

    Muestra:
    - Arquitectura del modelo
    """
    modelo.summary()
