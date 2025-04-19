from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam

def build_rnn_model(input_len, vocab_size, embedding_dim=128, rnn_units=64):
    """
    Construye un modelo RNN para análisis de sentimiento.

    Parámetros:
    - input_len: longitud máxima de las secuencias (padding)
    - vocab_size: número de palabras en el vocabulario
    - embedding_dim: dimensión de la capa de embedding
    - rnn_units: número de unidades en la capa RNN

    Retorna:
    - model: modelo Keras compilado
    """
    inputs = Input(shape=(input_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    x = SimpleRNN(rnn_units)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
