"""utility and helper functions / classes."""
import json
import logging
import os
import pickle
import random
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_num_classes(DATASET: str) -> int:
    """
    Get the number of classes to be classified by dataset.
    """
    if DATASET == "MELD":
        NUM_CLASSES = 7
    elif DATASET == "IEMOCAP":
        NUM_CLASSES = 6
    elif DATASET == "AGNews":
        NUM_CLASSES = 4
    elif DATASET == "IMDb":
        NUM_CLASSES = 2
    else:
        raise ValueError

    return NUM_CLASSES


def compute_metrics(eval_predictions) -> dict:
    """
    Return f1_weighted, f1_micro, and f1_macro scores.
    """
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)

    f1_weighted = f1_score(label_ids, preds, average="weighted")
    f1_micro = f1_score(label_ids, preds, average="micro")
    f1_macro = f1_score(label_ids, preds, average="macro")

    return {"f1_weighted": f1_weighted, "f1_micro": f1_micro, "f1_macro": f1_macro}


def set_seed(seed: int) -> None:
    """
    Set random seed to a fixed value.
    Make everything deterministic.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_label2id(DATASET: str) -> Tuple[dict, dict]:
    """
    Get a dict that converts string class to numbers.
    """
    if DATASET == "MELD":
        # MELD has 7 classes
        labels = [
            "neutral",
            "joy",
            "surprise",
            "anger",
            "sadness",
            "disgust",
            "fear",
        ]
    elif DATASET == "IEMOCAP":
        # IEMOCAP originally has 11 classes but we'll only use 6 of them.
        labels = [
            "neutral",
            "frustration",
            "sadness",
            "anger",
            "excited",
            "happiness",
        ]
    elif DATASET == "AGNews":
        # AGNews has 4 classes
        labels = [
            "1",
            "2",
            "3",
            "4",
        ]
    elif DATASET == "IMDb":
        # IMDb has 2 classes
        labels = [
            0,
            1
        ]

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {val: key for key, val in label2id.items()}

    return label2id, id2label


class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        DATASET="MELD",
        SPLIT="train",
        speaker_mode="upper",
        num_past_utterances=0,
        num_future_utterances=0,
        model_checkpoint="roberta-base",
        ROOT_DIR="dataset/",
        ONLY_UPTO=False,
        SEED=0,
    ):
        """
        Initialize emotion recognition in conversation text modality dataset class.
        """

        self.DATASET = DATASET
        self.ROOT_DIR = ROOT_DIR
        self.SPLIT = SPLIT
        self.speaker_mode = speaker_mode
        self.num_past_utterances = num_past_utterances
        self.num_future_utterances = num_future_utterances
        self.model_checkpoint = model_checkpoint
        self.ONLY_UPTO = ONLY_UPTO
        self.SEED = SEED
        self._load_classes()
        self.label2id, self.id2label = get_label2id(self.DATASET)

        if self.DATASET == "MELD" or self.DATASET == "IEMOCAP":
            self._load_utterance_ordered()
            self._string2tokens()
        if self.DATASET == "AGNews":
            self._load_agnews_topics()
            self._string2tokens_agnews()
        if self.DATASET == "IMDb":
            self._load_imdb_reviews()
            self._string2tokens_imdb()

    def _load_classes(self):
        """
        Load the supervised labels.
        """
        if self.DATASET in ["MELD", "IEMOCAP"]:
            with open(
                os.path.join(self.ROOT_DIR, self.DATASET, "emotions.json"), "r"
            ) as stream:
                self.labels = json.load(stream)[self.SPLIT]
        if self.DATASET in ["AGNews"]:
            with open(
                os.path.join(self.ROOT_DIR, self.DATASET, "topics.json"), "r"
            ) as stream:
                self.labels = json.load(stream)[self.SPLIT]

    def _load_utterance_ordered(self):
        """
        Load the ids of the utterances in order.
        """
        path = os.path.join(self.ROOT_DIR, self.DATASET, "utterance-ordered.json")

        with open(path, "r") as stream:
            self.utterance_ordered = json.load(stream)[self.SPLIT]

    def _load_agnews_topics(self):
        """
        Load the ids of the topics in order.
        """
        path = os.path.join(self.ROOT_DIR, self.DATASET, "topics.json")

        with open(path, "r") as stream:
            self.agnews_topics = json.load(stream)[self.SPLIT]

    def _load_imdb_reviews(self):
        """
        Load the ids of the utterances in order.
        """
        import datasets

        dataset = 'imdb'
        DATASET_DIR = os.path.join(self.ROOT_DIR, self.DATASET, ".data")
        revision = 'de29c68'

        if self.SPLIT == 'val':
            SPLIT = 'train'
        else:
            SPLIT = self.SPLIT

        self.imdb_reviews = datasets.load_dataset(dataset, split=SPLIT, cache_dir=DATASET_DIR, revision=revision)
        
        if self.SPLIT == 'train':
            self.imdb_reviews = self.imdb_reviews.train_test_split(test_size=0.1)['train']
        if self.SPLIT == 'val':
            self.imdb_reviews = self.imdb_reviews.train_test_split(test_size=0.1)['test']

    def __len__(self):
        return len(self.inputs_)

    def _load_utterance_speaker_emotion(self, uttid, speaker_mode) -> dict:
        """
        Load an speaker-name prepended utterance and emotion label.
        """
        text_path = os.path.join(self.ROOT_DIR, self.DATASET, "raw-texts", self.SPLIT, uttid + ".json")

        with open(text_path, "r", encoding="utf8") as stream:
            text = json.load(stream)

        utterance = text["Utterance"].strip()
        emotion = text["Emotion"]

        if self.DATASET == "MELD":
            speaker = text["Speaker"]
        elif self.DATASET == "IEMOCAP":
            sessid = text["SessionID"]
            # https: // www.ssa.gov/oact/babynames/decades/century.html
            speaker = {
                "Ses01": {"Female": "Mary", "Male": "James"},
                "Ses02": {"Female": "Patricia", "Male": "John"},
                "Ses03": {"Female": "Jennifer", "Male": "Robert"},
                "Ses04": {"Female": "Linda", "Male": "Michael"},
                "Ses05": {"Female": "Elizabeth", "Male": "William"},
            }[sessid][text["Speaker"]]
        else:
            raise ValueError(f"{self.DATASET} not supported!!!!!!")

        if speaker_mode is not None and speaker_mode.lower() == "upper":
            utterance = speaker.upper() + ": " + utterance
        elif speaker_mode is not None and speaker_mode.lower() == "title":
            utterance = speaker.title() + ": " + utterance

        return {"Utterance": utterance, "Emotion": emotion}

    def _load_agnews_sample(self, sample_id) -> dict:
        """
        Load a AGNews sample.
        """
        text_path = os.path.join(self.ROOT_DIR, self.DATASET, "raw-texts", self.SPLIT, sample_id + ".json")

        with open(text_path, "r", encoding="utf8") as stream:
            sample = json.load(stream)

        return sample

    def _create_input(self, diaids, speaker_mode, num_past_utterances, num_future_utterances):
        """
        Create an input which will be an input to RoBERTa.
        """
        args = {
            "diaids": diaids,
            "speaker_mode": speaker_mode,
            "num_past_utterances": num_past_utterances,
            "num_future_utterances": num_future_utterances,
        }

        logging.debug(f"arguments given: {args}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        max_model_input_size = tokenizer.max_model_input_sizes[self.model_checkpoint]
        num_truncated = 0

        inputs = []
        count = 0
        for diaid in tqdm(diaids):
            count += 1
            ues = [
                (self._load_utterance_speaker_emotion(uttid, speaker_mode), uttid)
                for uttid in self.utterance_ordered[diaid]
            ]

            num_tokens = [len(tokenizer(ue["Utterance"])["input_ids"]) for ue, _ in ues]

            for idx, (ue, dia_utt) in enumerate(ues):
                emotion = ue["Emotion"]

                if emotion not in list(self.label2id.keys()):
                    continue

                label = self.label2id[emotion]

                indexes = [idx]
                indexes_past = [
                    i for i in range(idx - 1, idx - num_past_utterances - 1, -1)
                ]
                indexes_future = [
                    i for i in range(idx + 1, idx + num_future_utterances + 1, 1)
                ]

                offset = 0
                if len(indexes_past) < len(indexes_future):
                    for _ in range(len(indexes_future) - len(indexes_past)):
                        indexes_past.append(None)
                elif len(indexes_past) > len(indexes_future):
                    for _ in range(len(indexes_past) - len(indexes_future)):
                        indexes_future.append(None)

                for i, j in zip(indexes_past, indexes_future):
                    if i is not None and i >= 0:
                        indexes.insert(0, i)
                        offset += 1
                        if (
                            sum([num_tokens[idx_] for idx_ in indexes])
                            > max_model_input_size
                        ):
                            del indexes[0]
                            offset -= 1
                            num_truncated += 1
                            break
                    if j is not None and j < len(ues):
                        indexes.append(j)
                        if (
                            sum([num_tokens[idx_] for idx_ in indexes])
                            > max_model_input_size
                        ):
                            del indexes[-1]
                            num_truncated += 1
                            break

                utterances = [ues[idx_][0]["Utterance"] for idx_ in indexes]

                if num_past_utterances == 0 and num_future_utterances == 0:
                    assert len(utterances) == 1
                    final_utterance = utterances[0]

                elif num_past_utterances > 0 and num_future_utterances == 0:
                    if len(utterances) == 1:
                        final_utterance = "</s></s>" + utterances[-1]
                    else:
                        final_utterance = (
                            " ".join(utterances[:-1]) + "</s></s>" + utterances[-1]
                        )

                elif num_past_utterances == 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = utterances[0] + "</s></s>"
                    else:
                        final_utterance = (
                            utterances[0] + "</s></s>" + " ".join(utterances[1:])
                        )

                elif num_past_utterances > 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = "</s></s>" + utterances[0] + "</s></s>"
                    else:
                        final_utterance = (
                            " ".join(utterances[:offset])
                            + "</s></s>"
                            + utterances[offset]
                            + "</s></s>"
                            + " ".join(utterances[offset + 1 :])
                        )
                else:
                    raise ValueError

                input_ids_attention_mask = tokenizer(final_utterance)
                input_ids = input_ids_attention_mask["input_ids"]
                attention_mask = input_ids_attention_mask["attention_mask"]
                label = label

                input_ = [
                    input_ids,
                    attention_mask,
                    label,
                ]

                inputs.append(input_)

        logging.info(f"number of truncated utterances: {num_truncated}")
        logging.info(f"number of inputs: {len(inputs)}")
        logging.info(f"inputs 0: {inputs[0]}")
        return inputs

    def _create_input_agnews(self):
        """
        Create an input which will be an input to RoBERTa.
        """
        file_name = os.path.join(self.ROOT_DIR, self.DATASET, f'{self.DATASET}_{self.model_checkpoint}_inputs_{self.SPLIT}')
        if not os.path.exists(file_name):
            tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
            inputs = []
            count = 0
            for agnews_sample_id in list(self.agnews_topics.keys()):
                count += 1
                agnews_sample = self._load_agnews_sample(agnews_sample_id)

                label = agnews_sample["Topic"]
                if label not in list(self.label2id.keys()):
                    continue
                label = self.label2id[label]

                input_ids_attention_mask = tokenizer(agnews_sample["Title"], agnews_sample["Details"])
                input_ids = input_ids_attention_mask["input_ids"]
                attention_mask = input_ids_attention_mask["attention_mask"]

                input_ = [
                    input_ids,
                    attention_mask,
                    label
                ]

                inputs.append(input_)

            with open(file_name, 'wb') as agnews_inputs:
                pickle.dump(inputs, agnews_inputs)
        else:
            logging.info(f"{file_name} existed")
            with open(file_name, 'rb') as agnews_inputs:
                inputs = pickle.load(agnews_inputs)

        logging.info(f"number of inputs: {len(inputs)}")
        logging.info(f"inputs 0: {inputs[0]}")

        return inputs

    def _create_input_imdb(self):
        """
        Create an input which will be an input to RoBERTa.
        """
        file_name = os.path.join(self.ROOT_DIR, self.DATASET, f'{self.DATASET}_{self.model_checkpoint}_inputs_{self.SPLIT}')
        if not os.path.exists(file_name):
            tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
            inputs = []
            count = 0
            for review in self.imdb_reviews:
                count += 1

                label = review["label"]
                if label not in list(self.label2id.keys()):
                    continue
                label = self.label2id[label]

                input_ids_attention_mask = tokenizer(review["text"], truncation=True, return_tensors="pt")
                input_ids = input_ids_attention_mask["input_ids"][0]
                attention_mask = input_ids_attention_mask["attention_mask"][0]

                input_ = [
                    input_ids,
                    attention_mask,
                    label,
                ]

                inputs.append(input_)

            with open(file_name, 'wb') as imdb_inputs:
                pickle.dump(inputs, imdb_inputs)
        else:
            logging.info(f"{file_name} existed")
            with open(file_name, 'rb') as imdb_inputs:
                inputs = pickle.load(imdb_inputs)

        logging.info(f"number of inputs: {len(inputs)}")
        logging.info(f"inputs 0: {inputs[0]}")

        return inputs


    def _string2tokens(self):
        """Convert string to (BPE) tokens."""
        logging.info(f"converting utterances into tokens ...")

        diaids = sorted(list(self.utterance_ordered.keys()))

        set_seed(self.SEED)
        # random.shuffle(diaids)

        if self.ONLY_UPTO:
            logging.info(f"Using only the first {self.ONLY_UPTO} dialogues ...")
            diaids = diaids[: self.ONLY_UPTO]

        logging.info(f"creating input utterance data ... ")
        self.inputs_ = self._create_input(
            diaids=diaids,
            speaker_mode=self.speaker_mode,
            num_past_utterances=self.num_past_utterances,
            num_future_utterances=self.num_future_utterances,
        )

    def _string2tokens_agnews(self):
        """Convert string to (BPE) tokens."""
        logging.info(f"converting text into tokens ...")

        set_seed(self.SEED)

        logging.info(f"creating input text data ... ")
        self.inputs_ = self._create_input_agnews()

    def _string2tokens_imdb(self):
        """Convert string to (BPE) tokens."""
        logging.info(f"converting text into tokens ...")

        set_seed(self.SEED)

        logging.info(f"creating input text data ... ")
        self.inputs_ = self._create_input_imdb()


    def __getitem__(self, index):
        return self.inputs_[index]
