import scribemodel
import scribemodel as scribe
import tensorflow as tf
import numpy as np
import os
from inspect import getsourcefile
from os.path import abspath

BASE_DIR, _ = os.path.split(abspath(getsourcefile(lambda: 0)))


run_name = "week_run"


train_dir = "datasets/full_train"
test_dir = "datasets/test"

train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)


def datasets_from_files(files, directory):
    sets = []

    for file in files:
        path = os.path.join(directory, file)
        train_data = tf.data.Dataset.load(path)
        train_data = train_data.map(lambda base: ((base["strokes"][:-1], base["chars"]), base["strokes"][1:]))
        sets.append(train_data)

    # list of personal sets
    return sets


def bucket_sort(dataset):  # misuses bucketing as sort
    bucket_boundaries = list(range(2, 2001, 1))
    sorted_bucket_batches = dataset.bucket_by_sequence_length(element_length_func=lambda x, y: tf.shape(y)[0],
                                                      bucket_boundaries=bucket_boundaries,
                                                      bucket_batch_sizes=[100000] * (len(bucket_boundaries) + 1),
                                                      pad_to_bucket_boundary=False,
                                                      drop_remainder=False)
    # sorted_bucket_batches: each batch contains all samples of a particular length, batches are sorted by this length
    sorted_set = sorted_bucket_batches.unbatch()

    return sorted_set


def pad_person_sets(person_sets):
    # finding out max sequence lengths
    bucket_seq_lens = []
    bucket_char_lens = []
    c = 0
    for person_set in person_sets:
        person_numpy = list(person_set.as_numpy_iterator())
        for idx, sample in enumerate(reversed(person_numpy)):
            if len(bucket_seq_lens) == idx:  # if list isnt yet long enough
                bucket_seq_lens.append(0)
                bucket_char_lens.append(0)
            seq_len = sample[1].shape[0]
            char_len = sample[0][1].shape[0]
            if seq_len > bucket_seq_lens[idx]:
                bucket_seq_lens[idx] = seq_len
            if char_len > bucket_char_lens[idx]:
                bucket_char_lens[idx] = char_len

    bucket_seq_lens = list(reversed(bucket_seq_lens))
    # TODO: why does bucket_seq_lens have a zero in first place?
    # bec. 0 gets appended, then 1 padding thing gets appended..???
    bucket_char_lens = list(reversed(bucket_char_lens))
    bucket_char_lens = [max(3, l) for l in bucket_char_lens] # Do that for conwolution in window? todo or simply handle less than three in window

    # padding according to prev findings
    pad_exp_dsets = []
    for person_set in person_sets:
        set_as_numpy = list(person_set.as_numpy_iterator())
        pad_repeats = len(bucket_seq_lens) - len(set_as_numpy)
        # pad_repeats: difference of the sets cardinality to the biggest sets cardinality
        pad = [((np.zeros((1, 3)), np.zeros((1))), np.zeros((1, 3)))]
        # pad: [(([[0, 0, 0]], [0]), [[0, 0, 0]])]

        # exp_numpy: append pad_repeat pad samples in front, to bring all sets to same length
        exp_numpy = pad * pad_repeats + set_as_numpy

        def pad_numpy(x, seq_len, char_len):
            #seq_rep = seq_len-x[0][0].shape[0]
            seq_rep = seq_len-len(x[0][0])
            #char_rep = char_len-x[0][1].shape[0]
            char_rep = char_len-len(x[0][1])
            seq_pad = np.repeat(np.zeros((1, 3)), repeats=seq_rep, axis=0)
            #seq_pad = seq_rep * [[0, 0, 0]]
            char_pad = np.repeat(np.zeros((1)), repeats=char_rep, axis=0)
            #char_pad = char_rep * [0]
            return (np.concatenate([x[0][0], seq_pad]), np.concatenate([x[0][1], char_pad])), np.concatenate([x[1], seq_pad])
            #return (x[0][0] + seq_pad, x[0][1] + char_pad), x[1] + seq_pad

        pad_exp_numpy = list(map(pad_numpy, exp_numpy, bucket_seq_lens, bucket_char_lens))
        pad_exp_ds = tf.data.Dataset.from_tensors(pad_exp_numpy[0])
        for sample in pad_exp_numpy[1:]:
            pad_exp_ds = pad_exp_ds.concatenate(tf.data.Dataset.from_tensors(sample))
        pad_exp_dsets.append(pad_exp_ds)

    return pad_exp_dsets


def data_for_priming(datasets_list, batch_size):
    sorted_sets = []
    for set in datasets_list:
        sorted_sets.append(bucket_sort(set))

    pad_set = tf.data.Dataset.from_tensor_slices(
        ((np.array([[[0, 0, 0]]], dtype=np.float32), np.array([[0]])), np.array([[[0, 0, 0]]], dtype=np.float32)))
    for i in range(batch_size - (len(sorted_sets)%batch_size)):
        sorted_sets.append(pad_set)
    # sorted_sets: each set sorted by sequence-length
    set_batches = [sorted_sets[i:i+batch_size] for i in range(0, len(sorted_sets), batch_size)]
    # set_batches: list of lists of (batch_size) sets
    padded_set_batches = [pad_person_sets(set_batch) for set_batch in set_batches]
    # padded_chunked_sets: contains lists of batch_size sets which are padded according to the other elements in the lst
    padded_sets = [set for chunk in padded_set_batches for set in chunk]
    ds_for_interleave = tf.data.Dataset.from_tensor_slices(padded_sets)
    interleaved_ds = ds_for_interleave.interleave(map_func=lambda x: x,
                                                  cycle_length=batch_size,
                                                  block_length=1
                                                  )
    batched = interleaved_ds.batch(batch_size, drop_remainder=True)

    return batched


train_sets = datasets_from_files(train_files, train_dir)

test_sets = datasets_from_files(test_files, test_dir)

train_sets = train_sets + test_sets

batch_size = 17  # teiler von 238 (236 ds)

train_for_priming = data_for_priming(train_sets, batch_size)  # (len 14)
test_for_priming = data_for_priming(test_sets[:16], batch_size)[0]

model = scribe.Model()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join("logs", run_name),
                                                      histogram_freq=1,
                                                      write_images=True,
                                                      embeddings_freq=1)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(BASE_DIR, "checkpoints", run_name, "weights.hdf5"),
    save_weights_only=True,
    verbose=0,
    save_freq=20)


predict_callback = scribemodel.PredictCallback(model, test_for_priming, BASE_DIR, run_name)


model.compile(optimizer='adam',
              loss=[scribe.Loss(), None, None],
              metrics=[['accuracy'], [None, None]],
              run_eagerly=True)


if os.path.isfile(os.path.join(BASE_DIR, "checkpoints", run_name, "weights.hdf5")):

    model.evaluate(test_for_priming.unbatch().batch(batch_size=batch_size, drop_remainder=True).take(1), verbose=2)

    model.load_weights(os.path.join(BASE_DIR, "checkpoints", run_name, "weights.hdf5"))

model.fit(train_for_priming, validation_data=test_for_priming, epochs=100, callbacks=[tensorboard_callback, model_checkpoint_callback, predict_callback], verbose=1)
# validation_data=test_batched,
