import scribemodel as scribe
import tensorflow as tf
import os
import datetime

train_dir = "datasets/train"
test_dir = "datasets/test"

train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

train_set = None
test_set = None

for file in train_files:
    path = os.path.join(train_dir, file)
    train_data = tf.data.experimental.load(path)
    train_data = train_data.map(lambda base: ((base["strokes"][:-1], base["chars"]), base["strokes"][1:]))
    if train_set is None:
        train_set = train_data
    else:
        train_set = train_set.concatenate(train_data)

for file in test_files:
    path = os.path.join(test_dir, file)
    test_data = tf.data.experimental.load(path)
    test_data = test_data.map(lambda base: ((base["strokes"][:-1], base["chars"]), base["strokes"][1:]))
    if test_set is None:
        test_set = test_data
    else:
        test_set = train_set.concatenate(test_data)

model = scribe.Model()

model.compile(optimizer='adam',
              loss=scribe.Loss(),
              metrics=['accuracy'],
              run_eagerly=True)

log_dir = "logs/hw/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

bucket_boundaries = [101, 201, 301, 401, 501, 601]
batch_sizes = [32] * (len(bucket_boundaries) + 1)

train_batched = train_set.bucket_by_sequence_length(element_length_func=lambda x, y: tf.shape(x[0])[0],
                                                    bucket_boundaries=bucket_boundaries,
                                                    bucket_batch_sizes=batch_sizes,
                                                    pad_to_bucket_boundary=True,
                                                    drop_remainder=True)
test_batched = test_set.bucket_by_sequence_length(element_length_func=lambda x, y: tf.shape(x[0])[0],
                                                  bucket_boundaries=bucket_boundaries,
                                                  bucket_batch_sizes=batch_sizes,
                                                  pad_to_bucket_boundary=True,
                                                  drop_remainder=True)

model.fit(train_batched, epochs=10, validation_data=test_batched, callbacks=[tensorboard_callback])
