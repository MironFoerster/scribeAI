import csv
import os
import tensorflow as tf
import numpy as np
import json

examples = {}
means = []

read_path = "csv/split/"
write_path = "datasets/full_train"

# alphabet = "0123456789,.!?'():- ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# alphabet = " !'(),-.0123456789:?ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxyz"
alphabet = ""
char_dist = {}

read_dir = os.scandir(read_path)

# check alphabet
if False:
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
alphabet = "!'(),-./0123456789:?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

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

                chars = line["text"].replace(" ", "")
                indices = [alphabet.index(char)+1 for char in chars]
                # one_hot_chars = tf.one_hot(indices, depth=len(alphabet)).numpy()

                if line["person"] not in examples.keys():
                    examples[line["person"]] = {"strokes": [], "chars": []}

                examples[line["person"]]["strokes"].append(strokes_list)
                examples[line["person"]]["chars"].append(indices)


for person in examples.keys():
    # write to personal dataset-file
    examples[person]["strokes"] = tf.ragged.constant(examples[person]["strokes"], ragged_rank=1)
    examples[person]["chars"] = tf.ragged.constant(examples[person]["chars"], ragged_rank=1)
    dataset = tf.data.Dataset.from_tensor_slices(examples[person])
    path = os.path.join(write_path, person+".ds")
    tf.data.Dataset.save(dataset, path)

