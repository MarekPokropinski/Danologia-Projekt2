import tensorflow as tf
from tensorflow.keras import regularizers


def build_model(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size, 32, embeddings_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            16,  return_sequences=True, kernel_regularizer=regularizers.l2(1e-5), recurrent_regularizer=regularizers.l2(1e-6))),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            16, kernel_regularizer=regularizers.l2(1e-5), recurrent_regularizer=regularizers.l2(1e-6))),
        # tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(128,  return_sequences=True, kernel_regularizer=None, recurrent_regularizer=None)),
        # tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(128, kernel_regularizer=None, recurrent_regularizer=None)),
        tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=regularizers.l2(1e-5)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model
