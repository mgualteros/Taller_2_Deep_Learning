def train_model(model, x_train, y_train, batch_size=64, epochs=10, val_split=0.2):
    """
    Entrena un modelo de Keras.

    Parámetros:
    - model (keras.Model): El modelo de Keras que se va a entrenar.
    - x_train (array): Datos de entrada para el entrenamiento.
    - y_train (array): Etiquetas de los datos de entrenamiento.
    - batch_size (int, opcional): Tamaño del batch para el entrenamiento. El valor por defecto es 64.
    - epochs (int, opcional): Número de épocas para entrenar el modelo. El valor por defecto es 10.
    - val_split (float, opcional): Proporción de los datos de entrenamiento que se utilizarán para la validación. El valor por defecto es 0.2.

    Retorna:
    - history (History): Objeto que contiene el historial de entrenamiento, con métricas como pérdida y precisión.
    """
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=val_split
    )
    return history
