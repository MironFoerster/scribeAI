import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
tfd = tfp.distributions
import window


class Model(tf.keras.Model):
    def __init__(self, num_lstms=3, hidden_size=256):
        super().__init__()
        self.num_lstms = num_lstms
        self.k_components = 20
        self.bias = 0  # unbiased predictions, higher bias means cleaner predictions
        self.hidden_size = hidden_size
        self.window_layer = window.AttentionLayer()
        self.alphabet = "!'(),-./0123456789:?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.lstms = []
        self.dense_outs = []
        for i in range(num_lstms):
            units = self.hidden_size
            if i == 1:  # second layer
                units += len(self.alphabet)
            self.lstms.append(tf.keras.layers.LSTM(units=units,
                                                   return_sequences=True,
                                                   return_state=False,
                                                   stateful=True))
            self.dense_outs.append(tf.keras.layers.Dense(units=6*self.k_components+1,
                                                         use_bias=False))

        self.out_bias = tf.Variable(tf.zeros_initializer()(shape=[6*self.k_components+1], dtype=tf.float32))

    def call(self, inputs, training=None):
        # wrong: inputs.shape: ([batch_size, num_timesteps, 3], [batch_size, num_chars, len_alphabet])
        y = tf.zeros_like(self.out_bias)
        hidden_state = inputs[0]
        char_seq = inputs[1]
        for layer in range(self.num_lstms):
            if layer == 1:  # second layer
                # compute window
                win = self.window_layer((hidden_state, char_seq))
                hidden_state = tf.concat([hidden_state, win], axis=-1)
            hidden_state = self.lstms[layer](hidden_state)
            y = y + self.dense_outs[layer](hidden_state)

        y = y + self.out_bias

        # bring certain parts of outputs to desired numerical range
        pred_params = self.process_network_output(y, bias=self.bias)

        if training:
            return pred_params
        else:
            return pred_params, win

    def process_network_output(self, network_y, bias=0):
        # input/return shape: [batch_size, (num_timesteps), 6*k +1]
        # apply processing to bring certain parts of
        # outputs to desired numerical range
        # for later use as parameters for mixture density layer

        k = self.k_components
        eos_probs = tf.expand_dims(1/(1+tf.exp(network_y[:, :, -1])), axis=-1)  # modified sigmoid
        component_weights = tf.nn.softmax(network_y[:, :, 0:k] * (1+bias))  # softmax
        correlations = tf.tanh(network_y[:, :, k:2*k])  # tanh
        means = network_y[:, :, 2*k:4*k]  # no processing
        std_devs = tf.exp(network_y[:, :, 4*k:6*k]-bias)  # exp

        # concat to recreate shape
        processed_output = tf.concat([eos_probs, component_weights, correlations, means, std_devs], axis=2)
        return processed_output

    def is_predict_finished(self, window):
        pred_finished = False

        return pred_finished

    def plot_predictions(self, pred_points, dist_img, eos_probs):
        fig, ax = plt.subplot()
        ax.imshow(dist_img)

        stroke_x = []
        stroke_y = []
        for point in pred_points[0, :, :].numpy():
            stroke_x.append(point[0])
            stroke_y.append(point[1])
            if point[2] == 1:  # if eos == 1
                ax.plot(stroke_x, stroke_y, 'b-', linewidth=2.0)
                stroke_x = []
                stroke_y = []

        ax.scatter(x=pred_points[0, :, 0].numpy(),
                   y=pred_points[0, :, 1].numpy(),
                   sizes=eos_probs)

    def plot_windows(self, char_seq, windows):
        return

    def predict(self, char_seq, primer=None, bias=1):
        self.reset_states()
        self.bias = bias
        if type(char_seq) == str:
            # convert chars to one hot
            pass

        # define starting point
        pred_points = [tf.constant([[[0, 0, 0]]])]
        pred_params = []
        pred_wins = []

        pred_finished = False
        while not pred_finished:
            pred_param, pred_win = self.__call__((pred_points[-1], char_seq), training=False)
            # pred_param.shape: [batch_size(1), num_timesteps(1), 6*k+1]
            # pred_win.shape: [batch_size(1), num_timesteps(1), len_alphabet]

            # create dist and sample a single point
            mixture, bernoulli = create_dists(pred_param)
            pred_point_coords = mixture.sample()
            pred_point_eos = bernoulli.sample()
            pred_point = tf.concat([pred_point_coords, pred_point_eos], axis=-1)
            # pred_point.shape: [batch_size(1), num_timesteps(1), 3]

            # append param, point and win to associated lists
            pred_params.append(pred_param)
            pred_points.append(pred_point)
            pred_wins.append(pred_win)

            # check if prediction should be terminated
            pred_finished = self.is_predict_finished(window)

        pred_params = tf.concat(pred_params, axis=1)
        # pred_params.shape: [batch_size(1), num_timesteps, 6*k+1]
        pred_points = tf.concat(pred_points, axis=1)
        # pred_params.shape: [batch_size(1), num_timesteps, 3]

        # create full sequence distributions from network output
        mixture, bernoulli = create_dists(pred_params)

        dist_img = img_from_mixture_dist(mixture, pred_points)

        self.plot_predictions(pred_points, dist_img, eos_probs=pred_params[:, :, -1])
        self.plot_windows(char_seq, pred_wins)  # todo pred wins is still a list

        # reset bias
        self.bias = 0
        return pred_points


def covar_mat_from_corr_and_stddev(corrs, devs):
    flat_corrs = tf.reshape(corrs, [-1])
    flat_corr_mats = tf.map_fn(lambda x: tf.constant([[1, x.numpy()], [x.numpy(), 1]]), flat_corrs)
    corr_mats = tf.reshape(flat_corr_mats, [corrs.shape[0], corrs.shape[1], corrs.shape[2], 2, 2])

    devs_mats = tf.reshape(devs, [devs.shape[0], devs.shape[1], int(devs.shape[2]/2), 1, 2])

    trans_devs_mats = tf.transpose(devs_mats, perm=[0, 1, 2, 4, 3])

    covar_matrices = corr_mats * devs_mats * trans_devs_mats
    return covar_matrices


def img_from_mixture_dist(mixture, pred_points):
    # pred_points.shape: [batchsize(1), timesteps, 3]
    max_x, nx = 5000, 5000
    max_y, ny = 1000, 1000
    x = np.linspace(0, max_x, nx)
    y = np.linspace(0, max_y, ny)
    x_grid, y_grid = np.meshgrid(x, y)
    mesh_grid = np.concatenate((np.expand_dims(x_grid, axis=-1), np.expand_dims(y_grid, axis=-1)), axis=-1)
    # mesh_grid.shape: [max_y, max_x, 2]

    # add batch and timestep dim to mesh_grid
    mesh_grids = np.expand_dims(np.expand_dims(mesh_grid, axis=0), axis=0)
    # mesh_grids.shape: [batch_size(1), num_timesteps(1), max_y, max_x, 2]

    # repeat timesteps dim
    mesh_grids = np.repeat(mesh_grids, pred_points.shape[1], axis=1)
    # mesh_grids.shape: [batch_size(1), num_timesteps, max_y, max_x, 2]

    # create offsets for moving the origin at each timestep
    offsets = np.expand_dims(np.expand_dims(pred_points[:, :, :2], axis=2), axis=2)
    # offsets.shape: [batch_size(1), num_timesteps, y(1), x(1), 2]
    # x, y will be broadcast

    # move mesh_grids at each timestep according to pred_points
    mesh_grids = mesh_grids + offsets
    # mesh_grids.shape: [batch_size(1), num_timesteps, max_y, max_x, 2]

    # add components dim to mesh_grids
    mesh_grids = np.expand_dims(mesh_grids, axis=0)
    # mesh_grids.shape: [batch_size(1), num_timesteps, k_components(1), max_y, max_x, 2]

    # permute mesh grids to desired shape for .prob() call
    mesh_grids = np.transpose(mesh_grids, axes=[3, 4, 0, 1, 2, 5])
    # mesh_grids.shape: [max_y, max_x, batch_size(1), num_timesteps, k_components, event_size(2)]
    # schematically: [M1, ..., Mm, B1, ..., Bb, event_size]
    # or: [sample_shape, batch_shape, event_shape]

    dist_imgs = mixture.prob(mesh_grid)
    # dist_imgs.shape: [max_y, max_x, batch_size(1), num_timesteps, k_components]
    # schematically: [M1, ..., Mm, B1, ..., Bb]
    # or: [sample_shape, batch_shape]

    # sum components
    dist_imgs = np.reduce_sum(dist_imgs.numpy(), axis=-1)
    # dist_imgs.shape: [max_y, max_x, batch_size(1), num_timesteps]

    # squeeze batch dimension
    dist_imgs = np.squeeze(dist_imgs, axis=2)
    # dist_imgs.shape: [max_y, max_x, num_timesteps]

    # sum timesteps
    dist_img = np.reduce_sum(dist_imgs, axis=-1)
    # dist_img.shape: [max_y, max_x]
    return dist_img


def create_dists(pred_params):
    # pred_params.shape: [batch_size, num_timesteps, k*6+1]
    k = int((pred_params.shape[2] - 1) / 6)
    if isinstance(pred_params, tf.RaggedTensor):
        pred_params = pred_params.to_tensor()

    shape = pred_params.shape

    mixture = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            # [batch_size, max_timesteps, k_components]
            probs=pred_params[:, :, 0:k]
        ),
        components_distribution=tfd.MultivariateNormalTriL(
            # [batch_size, max_timesteps, k_components, n_variables]
            loc=tf.reshape(pred_params[:, :, 2 * k:4 * k], [shape[0], shape[1], k, 2]),
            # [batch_size, max_timesteps, k_components, n_variables, n_variables]
            scale_tril=tf.linalg.cholesky(
                covar_mat_from_corr_and_stddev(pred_params[:, :, k:2 * k], pred_params[:, :, 4 * k:6 * k]))
        )
    )

    bernoulli = tfd.Bernoulli(
        probs=pred_params[:, :, -1]  # eos_probs.shape: [batch_size, max_timesteps, 1]
    )
    return mixture, bernoulli


class Loss(tf.keras.losses.Loss):
    def call(self, true_points, pred_params):
        # pred_params.shape: [batch_size, (num_timesteps), 1+6*k_components]
        # last dimension is made up of following values in order:
        # component_weights (k) + correlations (k) + means ((2)*k) + std_devs ((2)*k) + eos_prob (1)
        # true_points.shape: [batch_size, (num_timesteps), 3]

        if isinstance(true_points, tf.RaggedTensor):
            true_points = true_points.to_tensor()
        if isinstance(pred_params, tf.RaggedTensor):
            pred_params = pred_params.to_tensor()

        mixture, bernoulli = create_dists(pred_params)

        mixture_prob = mixture.log_prob(true_points[:, :, :2])
        bernoulli_prob = bernoulli.log_prob(true_points[:, :, 2])
        loss = - mixture_prob - bernoulli_prob  # [batch_size, num_timesteps]

        nan_free_loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        batch_losses = tf.math.reduce_sum(nan_free_loss, axis=1, keepdims=True)  # [batch_size]
        total_loss = tf.math.reduce_mean(batch_losses, axis=0, keepdims=True)  # []

        return total_loss
