import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Embedding,
    Multiply, Softmax, Lambda
)
from tensorflow.keras.optimizers import Adam

def build_bilstm_attention_model(ventana, vocab_size, embedding_dim=128):
    """
    Construye un modelo BiLSTM con atención para análisis de sentimientos.

    Este modelo utiliza una arquitectura de LSTM bidireccional junto con una capa de atención para mejorar la capacidad del 
    modelo de enfocarse en las partes relevantes de la secuencia de texto.

    Parámetros:
    - ventana: longitud máxima de las secuencias de entrada (después de aplicar padding)
    - vocab_size: número de palabras en el vocabulario (tamaño del vocabulario)
    - embedding_dim: dimensión de la representación densa para las palabras (embedding)

    Retorna:
    - model: modelo Keras compilado listo para entrenamiento. 
    """
    inputs = Input(shape=(ventana,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=ventana)(inputs)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(embedding)
    attention_weights = Dense(1, activation='tanh')(lstm_out)
    attention_weights = Softmax(axis=1)(attention_weights)
    attended = Multiply()([lstm_out, attention_weights])
    context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)
    output = Dense(1, activation='sigmoid')(context_vector)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
