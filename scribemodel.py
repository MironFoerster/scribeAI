import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
tfd = tfp.distributions
import window
import os

def apply_mask(tensor, mask):
    return tf.where(tf.expand_dims(mask, axis=-1), tensor, tf.zeros_like(tensor))

def coords_from_offsets(offsets):
    if tf.is_tensor(offsets):
        offsets = tf.split(offsets, offsets.shape[1], axis=1)

    # create absolute coordinates from offsets list
    coords = []
    coord = tf.constant([[[0, 0]]], dtype=tf.float32)
    # coord.shape: [1, 1, 2]
    for offset in offsets:
        coord = coord + offset[:, :, :2]
        coords.append(tf.concat([coord, tf.expand_dims(offset[:, :, -1], axis=-1)], axis=-1))
    coords = tf.concat(coords, axis=1)
    return coords

class Model(tf.keras.Model):
    def __init__(self, num_lstms=3, hidden_size=256):
        super().__init__()
        self.counter = None
        self.figures_dir = "figures"
        self.num_lstms = num_lstms
        self.k_components = 20
        self.bias = 0  # unbiased predictions, higher bias means cleaner predictions
        self.hidden_size = hidden_size
        self.masking = tf.keras.layers.Masking(mask_value=0.)
        self.alphabet = "!'(),-./0123456789:?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.window_layer = window.AttentionLayer(len(self.alphabet))
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

    def call(self, inputs):
        # wrong: inputs.shape: ([batch_size, num_timesteps, 3], [batch_size, num_chars, len_alphabet])
        y = tf.zeros_like(self.out_bias)
        hidden_state = inputs[0]
        char_seq = inputs[1]
        mask = self.masking.compute_mask(hidden_state)
        # mask.shape: [batch_size, num_timesteps]

        for layer in range(self.num_lstms):
            if layer == 1:  # second layer
                # compute window
                win = self.window_layer((apply_mask(hidden_state, mask), char_seq))
                alphabet_window = win[0]
                hidden_state = tf.concat([hidden_state, alphabet_window], axis=-1)
            hidden_state = self.lstms[layer](hidden_state)
            # hidden_state.shape: [batch_size, num_timesteps, lstm_size(256/328)]
            peephole = self.dense_outs[layer](hidden_state)
            y = y + peephole

        y = y + self.out_bias

        # bring certain parts of outputs to desired numerical range
        pred_params = self.process_network_output(y, bias=self.bias)

        # masking
        #pred_params = apply_mask(pred_params, mask)

        return pred_params, win

    def process_network_output(self, network_y, bias=0):
        # input/return shape: [batch_size, (num_timesteps), 6*k +1]
        # apply processing to bring certain parts of
        # outputs to desired numerical range
        # for later use as parameters for mixture density layer#
        p= False
        if tf.reduce_sum(network_y).numpy() == 0:
            p = True

        k = self.k_components
        eos_probs = tf.expand_dims(1/(1+tf.exp(network_y[:, :, -1])), axis=-1)  # modified sigmoid
        component_weights = tf.nn.softmax(network_y[:, :, 0:k] * (1+bias))  # softmax
        correlations = tf.tanh(network_y[:, :, k:2*k])  # tanh
        means = network_y[:, :, 2*k:4*k]  # no processing
        std_devs = tf.exp(network_y[:, :, 4*k:6*k]-bias)  # exp

        # concat to recreate shape
        processed_output = tf.concat([component_weights, correlations, means, std_devs, eos_probs], axis=2)
        if p:
            print(network_y)
            print(processed_output)
        return processed_output

    def is_predict_finished(self, win):
        self.counter += 1
        if self.counter > 100:
            return True
        else:
            return False

    def plot_predictions(self, dist_img=None, pred_points=None, eos_probs=None, img_size=None, save=None):
        fig, ax = plt.subplots()
        if dist_img is not None:
            ax.imshow(dist_img, extent=(0, img_size[0], img_size[1], 0))#(-0.5, img_size[0]-0.5, img_size[1]-0.5, -0.5))
        if pred_points is not None:
            stroke_x = []
            stroke_y = []
            for point in pred_points[0, :, :].numpy():
                stroke_x.append(point[0])
                stroke_y.append(point[1])

                if point[2] == 1:  # if eos == 1
                    last_x = point[0]
                    ax.plot(stroke_x, stroke_y, 'b-', linewidth=2.0)
                    stroke_x = []
                    stroke_y = []
            ax.plot(stroke_x, stroke_y, 'b-', linewidth=2.0)

        if eos_probs is not None:
            ax.scatter(x=pred_points[0, :, 0].numpy(),
                       y=pred_points[0, :, 1].numpy(),
                       sizes=eos_probs[0, :]*1000)
        plt.gca().invert_yaxis()
        if type(save) == str:  # save_filename
            plt.savefig(os.path.join(self.figures_dir, "preds", save))
        else:
            plt.show()
        plt.close("all")

    def plot_windows(self, string_chars, alphabet_windows, char_weights, save=None):
        fig, axs = plt.subplots(2, 1, layout='constrained')
        axs[0].imshow(char_weights, aspect="auto", interpolation="none")
        axs[0].set_xticks(np.arange(len(string_chars)), list(string_chars))
        axs[0].set_yticks(np.arange(char_weights.shape[0]))
        axs[0].set_title("window at char sequence")

        axs[1].imshow(alphabet_windows, aspect="auto", interpolation='none')
        axs[1].set_xticks(np.arange(len(self.alphabet)), list(self.alphabet))
        axs[1].set_yticks(np.arange(alphabet_windows.shape[0]))
        axs[1].set_title("window at alphabet")

        if type(save) == str:  # save_filename
            plt.savefig(os.path.join(self.figures_dir, "wins", save))
        else:
            plt.show()
        plt.close("all")

    def predict(self, char_seq, primer=None, bias=1, save=None):  # todo implement priming
        self.counter = 1  # only temporary for check if predict is finished
        self.reset_states()
        self.bias = bias

        if primer is not None:
            pass
            # pred_param, win = self.__call__((pred_offsets[-1], one_hot_chars))

        if type(char_seq) == str:
            # convert string to inices
            string_chars = char_seq
            index_chars = tf.expand_dims(tf.constant([self.alphabet.index(c)+1 for c in char_seq]), axis=0)
        else:
            # convert indices to string
            string_chars = "".join([self.alphabet[idx-1] for idx in char_seq])
            index_chars = char_seq
        # one_hot_chars.shape: [batch_size(1), num_chars, len_alphabet]

        # define starting offset
        pred_offsets = [tf.constant([[[0, 0, 0]]], dtype=tf.float32)]
        pred_params = []
        alphabet_windows = []
        char_weights = []

        pred_finished = False
        while not pred_finished:
            pred_param, win = self.__call__((pred_offsets[-1], index_chars))
            # pred_param.shape: [batch_size(1), num_timesteps(1), 6*k+1]
            # char_weight.shape: [batch_size(1), num_timesteps(1), num_chars, 1]

            # create dist and sample a single offset
            mixture, bernoulli = create_dists(pred_param)
            pred_offset_coords = mixture.sample()
            pred_offset_eos = bernoulli.sample()
            pred_offset = tf.concat([pred_offset_coords, tf.expand_dims(pred_offset_eos, axis=-1)], axis=-1)
            # pred_offset.shape: [batch_size(1), num_timesteps(1), 3]

            # squeeze alphabet_window vector
            alphabet_window = tf.squeeze(win[0], axis=0)
            # alphabet_window.shape: [num_timesteps(1), len_alphabet]
            # squeeze char_weight vector
            char_weight = tf.squeeze(tf.squeeze(win[1], axis=-1), axis=0)
            # char_weight.shape: [num_timesteps(1), num_chars]

            # append param, offset and char_weight to associated lists
            pred_params.append(pred_param)
            pred_offsets.append(pred_offset)
            alphabet_windows.append(alphabet_window)
            char_weights.append(char_weight)

            # check if prediction should be stopped
            pred_finished = self.is_predict_finished(win)  # todo how to see when predict is finished?

        pred_params = tf.concat(pred_params, axis=1)
        # pred_params.shape: [batch_size(1), num_timesteps, 6*k+1]
        alphabet_windows = tf.concat(alphabet_windows, axis=0)
        # alphabet_windows.shape: [num_timesteps, len_alphabet]
        char_weights = tf.concat(char_weights, axis=0)
        # char_weights.shape: [num_timesteps, num_chars]

        pred_points = coords_from_offsets(pred_offsets)

        # create full sequence distributions from network output
        mixture, bernoulli = create_dists(pred_params)

        img_size = (100, 10)
        dpu = 10  # dots per unit

        dist_img = self.img_from_mixture_dist(mixture, pred_points[:, 1:, :], img_size, dpu)
        self.plot_predictions(dist_img, pred_points, eos_probs=pred_params[:, :, -1], img_size=img_size, save=save)
        self.plot_windows(string_chars, alphabet_windows, char_weights, save=save)

        # reset bias
        self.bias = 0
        return pred_points

    def img_from_mixture_dist(self, mixture, pred_points, img_size, dpu):  # fixme: something is not working properly i puppose the offset stuff
        # pred_points.shape: [batchsize(1), timesteps, 3]
        x = np.linspace(0, img_size[0], img_size[0]*dpu, endpoint=False)
        y = np.linspace(0, img_size[1], img_size[1]*dpu, endpoint=False)
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
        offsets = np.expand_dims(np.expand_dims(pred_points[:, :, :2], axis=2), axis=2)# + 0.5
        # offsets.shape: [batch_size(1), num_timesteps, y(1), x(1), 2]
        # x, y will be broadcast
        # move mesh_grids at each timestep according to pred_points
        mesh_grids = mesh_grids - offsets
        # mesh_grids.shape: [batch_size(1), num_timesteps, max_y, max_x, 2]
        # permute mesh grids to desired shape for .prob() call
        mesh_grids = np.transpose(mesh_grids, axes=[2, 3, 0, 1, 4])

        # mesh_grids.shape: [max_y, max_x, batch_size(1), num_timesteps, event_size(2)]
        # schematically: [M1, ..., Mm, B1, ..., Bb, event_size]
        # or: [sample_shape, batch_shape, event_shape]

        dist_imgs = mixture.prob(mesh_grids)
        # dist_imgs.shape: [max_y, max_x, batch_size(1), num_timesteps]
        # schematically: [M1, ..., Mm, B1, ..., Bb]
        # or: [sample_shape, batch_shape]

        # squeeze batch dimension
        dist_imgs = np.squeeze(dist_imgs, axis=2)
        # dist_imgs.shape: [max_y, max_x, num_timesteps]

        # sum timesteps
        dist_img = np.sum(dist_imgs, axis=-1)
        # dist_img.shape: [max_y, max_x]
        return dist_img


def covar_mat_from_corr_and_stddev(corrs, devs):
    flat_corrs = tf.reshape(corrs, [-1])
    flat_corr_mats = tf.map_fn(lambda x: tf.constant([[1, x.numpy()], [x.numpy(), 1]]), flat_corrs)
    corr_mats = tf.reshape(flat_corr_mats, [corrs.shape[0], corrs.shape[1], corrs.shape[2], 2, 2])

    devs_mats = tf.reshape(devs, [devs.shape[0], devs.shape[1], int(devs.shape[2]/2), 1, 2])

    trans_devs_mats = tf.transpose(devs_mats, perm=[0, 1, 2, 4, 3])

    covar_matrices = corr_mats * devs_mats * trans_devs_mats
    return covar_matrices

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
            scale_tril=tf.linalg.cholesky(  # input must be nonzero
                covar_mat_from_corr_and_stddev(pred_params[:, :, k:2 * k], pred_params[:, :, 4 * k:6 * k]))
        )
    )

    bernoulli = tfd.Bernoulli(
        probs=pred_params[:, :, -1],  # eos_probs.shape: [batch_size, max_timesteps, 1]
        dtype=tf.float32
    )
    return mixture, bernoulli


class Loss(tf.keras.losses.Loss):
    def call(self, true_points, pred_params):
        # pred_params.shape: [batch_size, (num_timesteps), 1+6*k_components]
        # last dimension is made up of following values in order:
        # component_weights (k) + correlations (k) + means ((2)*k) + std_devs ((2)*k) + eos_prob (1)
        # true_points.shape: [batch_size, (num_timesteps), 3]
        #print(pred_params_and_mask)
        #pred_params = pred_params_and_mask[0]
        #mask = pred_params_and_mask[1]
        mixture, bernoulli = create_dists(pred_params)

        mixture_prob = mixture.log_prob(true_points[:, :, :2])
        bernoulli_prob = bernoulli.log_prob(true_points[:, :, 2])
        losses = - mixture_prob - bernoulli_prob  # [batch_size, num_timesteps]

        # masking
        #losses = apply_mask(losses, mask)
        # remove nans
        losses = tf.where(tf.math.is_nan(losses), tf.zeros_like(losses), losses)
        batch_losses = tf.math.reduce_sum(losses, axis=1, keepdims=True)  # [batch_size]
        total_loss = tf.math.reduce_mean(batch_losses, axis=0, keepdims=True)  # []

        return total_loss


class PredictCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, dataset, base_path, run_name):

        self.pred_model = Model(model.num_lstms, model.hidden_size)
        self.pred_model.compile(optimizer='adam',
                      loss=[Loss(), None, None],
                      metrics=[['accuracy'], [None, None]],
                      run_eagerly=True)
        self.base_path = base_path
        self.run_name = run_name
        self.pred_model.evaluate(dataset.unbatch().batch(batch_size=1).take(1), verbose=2)
        self.pred_model.load_weights(os.path.join(self.base_path, "checkpoints", self.run_name, "weights.hdf5"))
        self.pred_model.predict("hello", save="epoch{e:0>2}.png".format(e="test"))

    def on_epoch_end(self, epoch, logs=None):
        self.pred_model.load_weights(os.path.join(self.base_path, "checkpoints", self.run_name, "weights.hdf5"))
        self.pred_model.predict("hello", save="epoch{e:0>2}.svg".format(e=epoch))
