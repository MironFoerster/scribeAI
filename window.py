import tensorflow as tf


class GravesStyleNoOffset(tf.keras.layers.Layer):
    def __init__(self, k_gaussians):
        super().__init__()
        self.k_gaussians = k_gaussians
        self.dense = tf.keras.layers.Dense(units=3*self.k_gaussians)  # every gaussian needs three parameters a, b, k

    def call(self, inputs):
        lstm_out = inputs[0]
        # lstm_out.shape: [batch_size, num_timesteps, num_lstm_units] (out of first lstm)
        char_seq = inputs[1]
        # char_seq.shape: [batch_size, num_chars, len_alphabet] (one hot encoded characters)

        parameters = self.dense(lstm_out)
        exp_params = tf.exp(parameters)
        a, b, k = tf.split(exp_params, num_or_size_splits=3, axis=2)
        # a/b/k.shape: [batch_size, num_timesteps, k_gaussians]

        a = tf.expand_dims(a, axis=2)
        b = tf.expand_dims(b, axis=2)
        k = tf.expand_dims(k, axis=2)
        # a/b/k.shape: [batch_size, num_timesteps, 1, k_gaussians]
        # (added axis will be broadcast to num_chars)

        char_indices = tf.reshape(tf.range(char_seq.shape[1], dtype=tf.float32), shape=[1, 1, char_seq.shape[1], 1])  # range(num_chars)
        # char_indices.shape: [1 (bc to batch_size), 1 (bc to num_timesteps), num_chars, 1 (bc to k_gaussians)]
        # (bc to ~ will be broadcast to)
        # computing k gaussians per character per timestep according to A. Graves paper
        # prev. mentioned broadcasting takes place

        gaussians = tf.multiply(a, tf.exp(-tf.multiply(b, tf.square(k-char_indices))))
        # gaussians.shape: [batch_size, num_timesteps, num_chars, k_gaussians]
        # [batch_size, (num_timesteps), (num_chars), 1]
        # mixing the gaussians --> one gaussian per character per timestep
        char_weights = tf.reduce_sum(gaussians, axis=3, keepdims=True)
        # char_weights.shape: [batch_size, num_timesteps, num_chars, 1 (will be broadcast to len_alphabet)]

        # insert a timestep dimension into char_seq
        char_seq = tf.expand_dims(char_seq, axis=1)
        # expanded_char_seq.shape: [batch_size, 1 (will be broadcast to num_timesteps), num_chars, len_alphabet]

        # weight every char in char_seq at every timestep
        # prev. mentioned broadcasting takes place
        weighted_chars = tf.multiply(char_weights, char_seq)
        # weighted_chars.shape: [batch_size, num_timesteps, num_chars, len_alphabet]

        # sum all weighted_chars encodings at each timestep to make a single encoding
        alphabet_window = tf.reduce_sum(weighted_chars, axis=2)
        # alphabet_window.shape: [batch_size, (num_timesteps), len_alphabet]

        return alphabet_window


class GravesStyleWithOffset(tf.keras.layers.Layer):
    def __init__(self, k_gaussians):
        super().__init__()
        self.k_gaussians = k_gaussians
        self.dense = tf.keras.layers.Dense(units=3*self.k_gaussians)  # every gaussian needs three parameters a, b, k

    def call(self, inputs):
        lstm_out = inputs[0]
        # lstm_out.shape: [batch_size, num_timesteps, num_lstm_units] (out of first lstm)
        char_seq = inputs[1]
        # char_seq.shape: [batch_size, num_chars, len_alphabet] (one hot encoded characters)

        parameters = self.dense(lstm_out)
        exp_params = tf.exp(parameters)
        a, b, k = tf.split(exp_params, num_or_size_splits=3, axis=2)
        # a/b/k.shape: [batch_size, num_timesteps, k_gaussians]

        a = tf.expand_dims(a, axis=2)
        b = tf.expand_dims(b, axis=2)
        k = tf.expand_dims(k, axis=2)
        # a/b/k.shape: [batch_size, num_timesteps, 1, k_gaussians]
        # (added axis will be broadcast to num_chars)

        char_indices = tf.reshape(tf.range(char_seq.shape[1], dtype=tf.float32), shape=[1, 1, char_seq.shape[1], 1])  # range(num_chars)
        # char_indices.shape: [1 (bc to batch_size), 1 (bc to num_timesteps), num_chars, 1 (bc to k_gaussians)]
        # (bc to ~ will be broadcast to)
        # computing k gaussians per character per timestep according to A. Graves paper
        # prev. mentioned broadcasting takes place

        gaussians = tf.multiply(a, tf.exp(-tf.multiply(b, tf.square(k-char_indices))))
        # gaussians.shape: [batch_size, num_timesteps, num_chars, k_gaussians]
        # [batch_size, (num_timesteps), (num_chars), 1]
        # mixing the gaussians --> one gaussian per character per timestep
        char_weights = tf.reduce_sum(gaussians, axis=3, keepdims=True)
        # char_weights.shape: [batch_size, num_timesteps, num_chars, 1 (will be broadcast to len_alphabet)]

        # insert a timestep dimension into char_seq
        char_seq = tf.expand_dims(char_seq, axis=1)
        # expanded_char_seq.shape: [batch_size, 1 (will be broadcast to num_timesteps), num_chars, len_alphabet]

        # weight every char in char_seq at every timestep
        # prev. mentioned broadcasting takes place
        weighted_chars = tf.multiply(char_weights, char_seq)
        # weighted_chars.shape: [batch_size, num_timesteps, num_chars, len_alphabet]

        # sum all weighted_chars encodings at each timestep to make a single encoding
        alphabet_window = tf.reduce_sum(weighted_chars, axis=2)
        # alphabet_window.shape: [batch_size, (num_timesteps), len_alphabet]

        return alphabet_window


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.char_seq_embedding_layer = tf.keras.layers.Bidirectional(
            layer=tf.keras.layers.GRU(64, return_sequences=True)
        )
        self.char_weight_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        lstm_out = inputs[0]
        # lstm_out.shape: [batch_size, num_timesteps, num_lstm_units] (out of first lstm)
        lstm_out = tf.expand_dims(lstm_out, axis=2)
        # lstm_out.shape: [batch_size, num_timesteps, num_chars(1), num_lstm_units] (out of first lstm)

        char_seq = inputs[1]
        # char_seq.shape: [batch_size, num_chars, len_alphabet] (one hot encoded characters)
        char_seq_repres = self.char_seq_embedding_layer(char_seq)
        # char_repres.shape: [batch_size, num_chars, num_embedding_units]
        char_seq_repres = tf.expand_dims(char_seq_repres, axis=1)
        # char_repres.shape: [batch_size, num_timesteps(1), num_chars, num_embedding_units]

        lstm_out = tf.tile(lstm_out, [1, 1, char_seq_repres.shape[2], 1])
        char_seq_repres = tf.tile(char_seq_repres, [1, lstm_out.shape[1], 1, 1])

        char_weights = self.char_weight_layer(tf.concat([lstm_out, char_seq_repres], axis=-1))
        # char_weights.shape: [batch_size, num_timesteps, num_chars, 1(bc to len_alphabet)]
        print(char_weights[0, :, 0, 0]-char_weights[0, :, 1, 0])

        # insert a timestep dimension into char_seq
        char_seq = tf.expand_dims(char_seq, axis=1)
        # expanded_char_seq.shape: [batch_size, 1 (will be broadcast to num_timesteps), num_chars, len_alphabet]

        # weight every char in char_seq at every timestep
        # prev. mentioned broadcasting takes place
        weighted_chars = tf.multiply(char_weights, char_seq)
        # weighted_chars.shape: [batch_size, num_timesteps, num_chars, len_alphabet]

        # sum all weighted_chars encodings at each timestep to make a single alphabet
        alphabet_window = tf.reduce_sum(weighted_chars, axis=2)
        # alphabet_window.shape: [batch_size, (num_timesteps), len_alphabet]

        return alphabet_window, char_weights
