import scribemodel as scribe
import tensorflow as tf
import os

base_path = "C:/Users/miron/Git/scribeAI"

run_name = "initial"
ckpt_file = "weights-{epoch:02d}.hdf5"

train_dir = "datasets/miron"
test_dir = "datasets/miron"

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
        test_set = test_set.concatenate(test_data)


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join("logs", run_name),
                                                      histogram_freq=1,
                                                      update_freq=5)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(base_path, "checkpoints", run_name, ckpt_file),
    save_weights_only=True,
    verbose=1,
    save_freq=20)

bucket_boundaries = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 1101]
batch_sizes = [5] * (len(bucket_boundaries) + 1)

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

model = scribe.Model()
print(model.weights)

model.compile(optimizer='adam',
              loss=[scribe.Loss(), None, None],
              metrics=[['accuracy'], [None, None]],
              run_eagerly=True)


model.evaluate(test_batched, verbose=2)
model.load_weights(os.path.join(base_path, "checkpoints", run_name, ckpt_file.format(epoch=2)))
model.fit(train_batched, epochs=1, validation_data=test_batched, callbacks=[tensorboard_callback, model_checkpoint_callback], verbose=1)
model.evaluate(test_batched, verbose=2)
model.load_weights(os.path.join(base_path, "checkpoints", run_name, ckpt_file.format(epoch=2)))