import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from window import Layer as WindowLayer

class Model(tf.keras.Model):
    def __init__(self, num_lstms=3, hidden_size=256):
        super().__init__()
        self.num_lstms = num_lstms
        self.k_components = 20
        self.bias = 1
        self.hidden_size = hidden_size
        self.window_layer = WindowLayer(k_gaussians=15)
        self.lstms = []
        self.dense_outs = []
        for i in range(num_lstms):
            self.lstms.append(tf.keras.layers.LSTM(units=self.hidden_size,
                                                   return_sequences=True,
                                                   return_state=True,
                                                   stateful=True))
            self.dense_outs.append(tf.keras.layers.Dense(units=6*self.k_components+1,
                                                         use_bias=False))

        self.out_bias = tf.Variable(tf.zeros_initializer()(shape=[6*self.k_components+1], dtype=tf.float32))


    def call(self, inputs):
        # inputs.shape: [batch_size, (timesteps)/None, ]
        y = tf.zeros_like(self.out_bias)
        hidden_state = inputs[0]
        char_seq = inputs[1]


        for layer in range(self.num_lstms):
            if layer == 1:  # second layer
                # compute window
                self.window_layer((hidden_state, char_seq))
            hidden_state = self.lstms[layer](hidden_state)
            y += self.dense_outs[layer](hidden_state)

        y += self.out_bias

        # apply processing to bring certain parts of
        # outputs to desired numerical range
        y_pred = self.process_network_output(y)
        return y_pred

    def process_network_output(self, network_y):
        # input/return shape: [batch_size, (num_timesteps), 6*k +1]
        # apply processing to bring certain parts of
        # outputs to desired numerical range
        # for later use as parameters for mixture density layer

        k = self.k_components
        eos_probs = 1/(1+tf.exp(network_y[:, :, -1]))  # modified sigmoid
        component_weights = tf.nn.softmax(network_y[:, :, 0:k])  # softmax
        correlations = tf.tanh(network_y[:, :, k:2*k])  # tanh
        means = network_y[:, :, 2*k:4*k]  # no processing
        std_devs = tf.exp(network_y[:, :, 4*k:6*k])  # exp

        # concat to recreate shape
        processed_output = tf.concat([eos_probs, component_weights, correlations, means, std_devs], axis=2)
        return processed_output




def covar_mat_from_corr_and_stddev(corrs, devs):
    flat_corrs = tf.reshape(corrs, [-1])
    flat_corr_mats = tf.map_fn(lambda x: tf.constant([[1, x.numpy()], [x.numpy(), 1]]), flat_corrs)
    corr_mats = tf.reshape(flat_corr_mats, [corrs.shape[0], corrs.shape[1], corrs.shape[2], 2, 2])

    devs_mats = tf.reshape(devs, [devs.shape[0], devs.shape[1], int(devs.shape[2]/2), 1, 2])

    trans_devs_mats = tf.transpose(devs_mats, perm=[0, 1, 2, 4, 3])

    covar_matrices = corr_mats * devs_mats * trans_devs_mats
    return covar_matrices


class Loss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        # y_pred.shape: [batch_size, (num_timesteps), 1+6*k_components]
        # last dimension is made up of following values in order:
        # component_weights (k) + correlations (k) + means ((2)*k) + std_devs ((2)*k) + eos_prob (1)
        # y_true.shape: [batch_size, (num_timesteps), 3]

        k = int((y_pred.shape[2] - 1) / 6)
        y_true = y_true.to_tensor()
        y_pred = y_pred.to_tensor()
        shape = y_pred.shape

        mixture = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                # [batch_size, max_timesteps, k_components]
                probs=y_pred[:, :, 0:k]
            ),
            components_distribution=tfd.MultivariateNormalTriL(
                # [batch_size, max_timesteps, k_components, n_variables]
                loc=tf.reshape(y_pred[:, :, 2*k:4*k], [shape[0], shape[1], k, 2]),
                # [batch_size, max_timesteps, k_components, n_variables, n_variables]
                scale_tril=tf.linalg.cholesky(covar_mat_from_corr_and_stddev(y_pred[:, :, k:2*k], y_pred[:, :, 4*k:6*k]))
            )
        )

        bernoulli = tfd.Bernoulli(
            probs=y_pred[:, :, -1]  # eos_probs [batch_size, max_timesteps, 1]
        )

        mixture_prob = mixture.log_prob(y_true[:, :, :2])
        bernoulli_prob = bernoulli.log_prob(y_true[:, :, 2])
        loss = - mixture_prob - bernoulli_prob  # [batch_size, num_timesteps]

        nan_free_loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        batch_losses = tf.math.reduce_sum(nan_free_loss, axis=1, keepdims=True)  # [batch_size]
        total_loss = tf.math.reduce_mean(batch_losses, axis=0, keepdims=True)  # []

        return total_loss
