import scribemodel
import scribemodel as scribe
import tensorflow as tf
import os

base_path = "C:/Users/miron/Git/scribeAI"

run_name = "miron"

train_dir = "datasets/train"
test_dir = "datasets/test"

train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)
print(train_files, test_files)

def datasets_from_files(files, dir ):
    sets = []

    for file in files:
        path = os.path.join(dir, file)
        train_data = tf.data.Dataset.load(path)
        train_data = train_data.map(lambda base: ((base["strokes"][:-1], base["chars"]), base["strokes"][1:]))
        sets.append(train_data)
    return sets


def bucket(dataset, bucket_size=20):  #acts as sort
    bucket_boundaries = list(range(bucket_size+1, 2001, bucket_size))
    batched_buckets = dataset.bucket_by_sequence_length(element_length_func=lambda x, y: tf.shape(y)[0],
                                                      bucket_boundaries=bucket_boundaries,
                                                      bucket_batch_sizes=[10000] * (len(bucket_boundaries) + 1),
                                                      pad_to_bucket_boundary=True,
                                                      drop_remainder=False)

    #buckets = batched_buckets.unbatch()
    return batched_buckets

def pad_person_sets(person_sets):
    # padding the number of elements of each person in each bucket-batch to the maximum number of elements in this bucket
    # buckets_sets_batch = tf.data.Dataset.from_tensor_slices(sets_batch)
    # buckets_sets_batch = buckets_sets_batch.map(lambda set: set.map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y))))
    sets_list = [[bucket for bucket in set] for set in person_sets]
    bucket_boundaries = []
    for person_set in sets_list:
        for bucket in person_set:
            bucket_boundaries.append(bucket[1].shape[1])
    bucket_boundaries = sorted(list(set(bucket_boundaries)))

    max_samples_bucketwise = [0] * len(bucket_boundaries)
    for person_set in sets_list:
        for bucket in person_set:
            bucket_idx = bucket_boundaries.index(bucket[1].shape[1])
            n_samples = bucket[1].shape[0]
            if n_samples > max_samples_bucketwise[bucket_idx]:
                max_samples_bucketwise[bucket_idx] = n_samples

    padding_num = 0
    pad_exp_dsets = []
    for person_set in sets_list:
        exp_set = []
        idx_offset = 0
        for idx, bucket in enumerate(person_set):
            while bucket[1].shape[1] != bucket_boundaries[idx+idx_offset]:
                seq_len = bucket_boundaries[idx+idx_offset]
                pad = ((tf.zeros((1, seq_len, 3)), tf.zeros((1, seq_len, 72))), tf.zeros((1, seq_len, 3)))
                exp_set.append(pad)
                padding_num += seq_len
                idx_offset += 1
            exp_set.append(bucket)

        pad_exp_set = []
        for idx, bucket in enumerate(exp_set):
            seq_len = bucket_boundaries[idx]
            repeats = max_samples_bucketwise[idx] - bucket[1].shape[0]
            pad_3 = tf.repeat(tf.zeros((1, seq_len, 3)), [repeats], axis=0)
            pad_72 = tf.repeat(tf.zeros((1, seq_len, 72)), [repeats], axis=0)

            pad_bucket = ((tf.concat([bucket[0][0], pad_3], axis=0), tf.concat([bucket[0][1], pad_72], axis=0)), tf.concat([bucket[1], pad_3], axis=0))
            pad_exp_set.append(tf.data.Dataset.from_tensor_slices(pad_bucket))
            padding_num += seq_len*repeats

        concat_buckets_ds = pad_exp_set[0]
        for bucket in pad_exp_set[1:]:
            concat_buckets_ds.concatenate(bucket)

        pad_exp_dsets.append(concat_buckets_ds)
    # [num_blocks, num_persons-perblock(5), num_buckets, ((, ), )]
    #pad_exp_buckets_sets_batch = tf.data.Dataset.from_tensor_slices(pad_exp_sets)
    # padded_sets_batch = pad_exp_buckets_sets_batch.interleave(lambda x: x,
    #                                                           cycle_length=1,
    #                                                          block_length=max_samples)
    # batched_sets = padded_sets_batch.batch(batch_size, drop_remainder=True)
    return pad_exp_dsets, padding_num

def data_for_priming(datasets_list, batch_size):
    #datasets = tf.data.Dataset.from_tensor_slices(datasets)
    #print(datasets)
    #for s in datasets:
    #    print(s)
    bucket_sizes = [50]#list(range(40, 60, 2))
    print("entering for")
    for bucket_size in bucket_sizes:
        #bucketed_sets = datasets.map(lambda x: bucket(x, bucket_size))
        bucketed_sets = []
        for set in datasets_list:
            bucketed_sets.append(bucket(set, bucket_size))
        #print(bucketed_sets)

        chunked_sets = [bucketed_sets[i:i+batch_size] for i in range(0, len(bucketed_sets), batch_size)]
        #sets_batches = bucketed_sets.batch(batch_size, drop_remainder=True)
        #print(sets_batches)
        #for i in sets_batches:
        #    print(i)
        #    for s in i:
        #        print(s)
        #padded_sets_batches = sets_batches.map(lambda sets_batch: pad_sets_batch(sets_batch, batch_size))

        padded_chunked_sets, padding_nums = tuple(zip(*[pad_person_sets(chunk) for chunk in chunked_sets]))
        total_padding = sum(padding_nums)
        print(total_padding)
        unchunked_sets = [set for chunk in padded_chunked_sets for set in chunk]
        ds_for_interleave = tf.data.Dataset.from_tensor_slices(unchunked_sets)
        #unbatched_sets = padded_sets_batches.map(lambda ds: ds.unbatch())
    print("interleaveing")
    interleaved_ds = ds_for_interleave.interleave(map_func=lambda x: x,
                                                  cycle_length=batch_size,
                                                  block_length=1
                                                  )
    batched = interleaved_ds.batch(batch_size, drop_remainder=True)

    #bucket_boundaries = []
    #for batch in batched:
     #   bucket_boundaries.append(batch[1].shape[1]+1)

    #batched = batched.bucket_by_sequence_length(lambda x, y: y.shape[0],
     #                                           bucket_boundaries=bucket_boundaries,
      #                                          bucket_batch_sizes=[batch_size]*(len(bucket_boundaries)+1),
       #                                         pad_to_bucket_boundary=True,
        #                                        drop_remainder=True)
    return batched


train_sets = datasets_from_files(train_files, train_dir)
test_sets = datasets_from_files(test_files, test_dir)
batch_size = 20
train_for_priming = data_for_priming(train_sets, batch_size)
test_for_priming = data_for_priming(test_sets, batch_size)

model = scribe.Model()
print("instantiate model")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join("logs", run_name),
                                                      histogram_freq=1,
                                                      update_freq=5)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(base_path, "checkpoints", run_name, "weights.hdf5"),
    save_weights_only=True,
    verbose=1,
    save_freq=20)

predict_callback = scribemodel.PredictCallback(model, test_for_priming, base_path, run_name)

model.compile(optimizer='adam',
              loss=[scribe.Loss(), None, None],
              metrics=[['accuracy'], [None, None]],
              run_eagerly=True)

if os.path.isfile(os.path.join(base_path, "checkpoints", run_name, "weights.hdf5")):
    print("evaluating")
    test_eval_x = (tf.constant([[[0., 0., 0.],
        [0., 0., 0.]]], dtype=tf.float32), tf.constant([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0.]]], dtype=tf.float32))
    test_eval_y = tf.constant([[[0., 0., 0.],
                                 [0., 0., 0.]]], dtype=tf.float32)
    model.evaluate(test_eval_x, test_eval_y, verbose=2)
    model.evaluate(test_for_priming.unbatch().batch(batch_size=batch_size, drop_remainder=True).take(1), verbose=2)
    print("evaluated")
    model.load_weights(os.path.join(base_path, "checkpoints", run_name, "weights.hdf5"))
    print("loaded")
print("fitting")
model.fit(train_for_priming, validation_data=test_for_priming, epochs=50, callbacks=[tensorboard_callback, model_checkpoint_callback, predict_callback], verbose=1)
# validation_data=test_batched,
