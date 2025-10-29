import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K


def build_model(input_shape, vocab_size):
    """Pravi CNN + BiGRU model za prepoznavanje govora."""
    inputs = layers.Input(shape=input_shape, name='input')

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization(epsilon=1e-3)(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(epsilon=1e-3)(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((x.shape[1], -1))(x)

    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(vocab_size + 1, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='output')(x)

    model = Model(inputs, outputs, name='SpeechToTextModel')
    return model


def ctc_loss(y_true, y_pred, input_len, label_len):
    """Računa CTC gubitak za sekvencijalno treniranje modela."""
    input_len = tf.cast(input_len, dtype='int32')
    label_len = tf.cast(label_len, dtype='int32')
    return K.ctc_batch_cost(y_true, y_pred, input_len, label_len)
