# Extreme Fine-Tuning

This repository is a Python-based project designed for Extreme Fine-Tuning. This project leverages robust models like RoBERTa for fine-tuning on various datasets to achieve high efficient training.

## Requirements
- Python 3.9
- Cuda 11.3
- PyTorch 1.12.1+cu113

## Installation
To set up the project environment, run the following command:

```
pip install -r requirements.txt
```

## Preparing the Dataset
For detailed instructions on how to prepare the dataset, please refer to the README in the `dataset` directory.

## Usage
To use Extreme Fine-Tuning, run the `run.py` script with the desired command line arguments. Here are some example commands:

```
python run.py --seed 42 --bpe 3 --nh 400 --cp roberta-large --num-past-utterances 0 --num-future-utterances 0 --batch-size 32 --batch-size-elm 128 --dataset MELD --default-lr 0.000005
```

```
python run.py --seed 42 --bpe 3 --nh 400 --cp roberta-large --num-past-utterances 128 --num-future-utterances 128 --batch-size 32 --batch-size-elm 128 --dataset MELD --default-lr 0.000005
```

```
python run.py --seed 42 --bpe 3 --nh 400 --cp roberta-large --num-past-utterances 128 --num-future-utterances 0 --batch-size 32 --batch-size-elm 128 --dataset IEMOCAP --default-lr 0.000005
```

```
python run.py --seed 42 --bpe 2 --nh 400 --cp roberta-large --num-past-utterances 0 --num-future-utterances 0 --batch-size 32 --batch-size-elm 128 --dataset IMDb --default-lr 0.000005
```

## Command Line Arguments
- `--seed`: Set the seed for randomness control.
- `--bpe`: Set the number of backpropagation epochs.
- `--nh`: Set the number of hidden nodes.
- `--cp`: Set the model checkpoint (e.g., roberta-large).
- `--num-past-utterances`: Set the number of past utterances.
- `--num-future-utterances`: Set the number of future utterances.
- `--batch-size`: Set the batch size for backpropagation (BP).
- `--batch-size-elm`: Set the batch size for Extreme Learning Machine (ELM).
- `--dataset`: Choose the dataset to use (e.g., MELD, IEMOCAP, IMDb).
- `--default-lr`: Set the default learning rate.
