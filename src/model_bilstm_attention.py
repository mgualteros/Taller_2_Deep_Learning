import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Embedding,
    Multiply, Softmax, Lambda
)
from tensorflow.keras.optimizers import Adam
def build_bilstm_attention_model(ventana, vocab_size, embedding_dim=128):
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