import csv
import os
import tensorflow as tf
import json

examples = {}
means = []

read_path = "csv/split/"
write_path = "records/train/"

# alphabet = "0123456789,.!?'():- ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# alphabet = " !'(),-.0123456789:?ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxyz"
alphabet = ""
char_dist = {}

read_dir = os.scandir(read_path)

# check alphabet
for obj in read_dir:
    if obj.is_dir():
        with open(os.path.join(read_path, obj.name, "submits.csv")) as f:
            reader = csv.DictReader(f, fieldnames=["strokes", "text", "person"], delimiter=';')
            next(reader)  # skip the header
            for line in reader:
                for c in line["text"]:
                    if c == " ":
                        continue
                    if c not in alphabet:
                        char_dist[c] = 0
                        alphabet += c
                    char_dist[c] += 1


print(dict(sorted(char_dist.items(), key=lambda e: e[1])))

print("".join(sorted(alphabet)))

read_dir = os.scandir(read_path)
for obj in read_dir:
    if obj.is_dir():
        with open(os.path.join(read_path, obj.name, "submits.csv")) as f:
            reader = csv.DictReader(f, fieldnames=["strokes", "text", "person"], delimiter=';')
            next(reader)  # skip the header
            for line in reader:
                # preprocess line
                raw_strokes = json.loads(line["strokes"].replace("'", '"'))
                strokes_list = []
                prev_x = 0
                prev_y = 0
                for point in raw_strokes:
                    strokes_list.append([point["x"]-prev_x, point["y"]-prev_y, point["eos"]])
                    prev_x = point["x"]
                    prev_y = point["y"]

                strokes = tf.constant(strokes_list)
                serialized_strokes = tf.io.serialize_tensor(strokes).numpy()
                means.append(tf.reduce_mean(strokes[:, :2], axis=0).numpy())

                chars = line["text"].replace(" ", "")
                indices = [alphabet.index(char) for char in chars]
                one_hot_chars = tf.one_hot(indices, depth=len(alphabet))
                serialized_one_hot_chars = tf.io.serialize_tensor(one_hot_chars).numpy()

                # create features dictionary
                feature = {
                    "strokes": tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_strokes])),
                    "chars": tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_one_hot_chars]))
                }
                # create example from features
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                if line["person"] not in examples.keys():
                    examples[line["person"]] = []

                examples[line["person"]].append(example.SerializeToString())


for person in examples.keys():
    print(person)
    # write to personal records-file
    filename = os.path.join(write_path, person + ".tfrecords")
    with tf.io.TFRecordWriter(filename) as writer:
        for example in examples[person]:
            writer.write(example)
