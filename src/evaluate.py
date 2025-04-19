def evaluate_model(model, x_test, y_test):
    """
Evalúa un modelo de Keras.

Parámetros:
- model (keras.Model): El modelo de Keras que se va a evaluar.
- x_test (array): Datos de entrada para la evaluación.
- y_test (array): Etiquetas de los datos de evaluación.

Retorna:
- loss (float): La pérdida calculada en los datos de prueba.
- accuracy (float): La precisión calculada en los datos de prueba.
"""
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
    return loss, accuracy
