import csv
import json
import os

root_path = "./dataset/AGNews"
raw_paths = f"{root_path}/raw/%s.csv"
raw_texts_paths = f"{root_path}/raw-texts/%s"
raw_texts_file_paths = f"{root_path}/raw-texts/%s/sample%s.json"

labels = dict()

for SPLIT in ["train", "val", "test"]:
    try:
        os.makedirs(raw_texts_paths % SPLIT)
    except FileExistsError:
        pass
    FILE_SPLIT = "train" if SPLIT == "val" else SPLIT
    labels[SPLIT] = dict()
    with open(raw_paths % FILE_SPLIT, mode='r', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        COUNTER = 0
        rows = list(csvreader)
        if SPLIT == "train":
            rows = rows[:int(len(rows)*0.9)]
        elif SPLIT == "val":
            rows = rows[int(len(rows)*0.9):]
        for row in rows:
            sample = {
                "Topic": row[0],
                "Title": row[1],
                "Details": row[2]
            }
            json_object = json.dumps(sample, indent=4)
            with open(raw_texts_file_paths % (SPLIT, COUNTER), mode='w') as raw_texts_file:
                raw_texts_file.write(json_object)
                labels[SPLIT]["sample%s" % COUNTER] = row[0]
                COUNTER += 1

json_object = json.dumps(labels, indent=4)
with open(f"{root_path}/topics.json", mode='w') as topics_json:
    topics_json.write(json_object)
