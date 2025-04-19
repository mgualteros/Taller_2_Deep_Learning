from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def build_lstm_model(input_len, vocab_size, embedding_dim=128):
    """
    Construye un modelo LSTM profundo para análisis de sentimientos.

    Parámetros:
    - input_len: longitud máxima de las secuencias (padding)
    - vocab_size: número de palabras en el vocabulario
    - embedding_dim: dimensión de la capa de embedding

    Retorna:
    - model: modelo Keras compilado
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_len),
        LSTM(64, return_sequences=True),
        Dropout(0.5),
        LSTM(32),
        Dropout(0.4),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model