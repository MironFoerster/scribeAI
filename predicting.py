import scribemodel as scribe
import tensorflow as tf
import os
import matplotlib.pyplot as plt

base_path = "/home/miron/scribeAI"

run_name = "full_run"

test_dir = "datasets/full_train"
test_files = os.listdir(test_dir)
test_set = None

for file in test_files:
    path = os.path.join(test_dir, file)
    test_data = tf.data.Dataset.load(path)
    test_data = test_data.map(lambda base: ((base["strokes"][:-1], base["chars"]), base["strokes"][1:]))
    if test_set is None:
        test_set = test_data
    else:
        test_set = test_set.concatenate(test_data)

bucket_boundaries = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 1101]
batch_sizes = [1] * (len(bucket_boundaries) + 1)
test_batched = test_set.bucket_by_sequence_length(element_length_func=lambda x, y: tf.shape(x[0])[0],
                                                  bucket_boundaries=bucket_boundaries,
                                                  bucket_batch_sizes=batch_sizes,
                                                  pad_to_bucket_boundary=True,
                                                  drop_remainder=True)

model = scribe.Model()
model.compile(optimizer='adam',
              loss=[scribe.Loss(), None, None],
              metrics=[['accuracy'], [None, None]],
              run_eagerly=True)
#model.evaluate(test_set.batch(batch_size=1).take(1), verbose=2)
#pred_points, strokes, full, dists, word_wins, alph_wins = model.predict("freak")
#plt.show()
model.evaluate(test_set.batch(batch_size=1).take(1), verbose=2)

model.load_weights(os.path.join(base_path, "checkpoints", run_name, "weights.hdf5"))

pred_points = model.predict("hello")
plt.show()
