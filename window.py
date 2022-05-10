import tensorflow as tf

class Layer(tf.keras.layers.Layer):
    def __init__(self, k_gaussians):
        self.k_gaussians = k_gaussians
        self.dense = tf.keras.layers.Dense(units=3*self.k)  # every gaussian needs three parameters a, b, k

    def call(self, inputs):
        # inputs[0].shape: [batch_size, (num_timesteps), num_lstm_units] (out of first lstm)
        # inputs[1].shape: [batch_size, (num_chars), len_alphabet] (one hot encoded characters)

        n_timesteps = inputs[0].shape[1]  # works only with dense tensors...

        parameters = self.dense(inputs[0])
        a, b, k = tf.split(parameters, num_or_size_splits=3, axis=2)  # each size: [batch_size, (num_timesteps), k_gaussians]

        gaussians = tf.multiply(a, tf.exp(tf.multiply(-b, tf.square(k-u))))  # [batch_size, (num_timesteps), (num_chars), 1]

        char_weights = tf.reduce_sum(gaussians, axis=1)  # [batch_size, (num_chars), 1]

        tiled_chars = tf.tile(tf.expand_dims(inputs[1], axis=1), multiples=tf.constant([1, n_timesteps, 1, 1], dtype=tf.int32))

        weighted_chars = tf.multiply(char_weights, tiled_chars)  # shape: [batch_size, (num_timesteps), (num_chars), len_alphabet]

        return tf.reduce_sum(weighted_chars, axis=2)  # shape: [batch_size, (num_timesteps), len_alphabet]