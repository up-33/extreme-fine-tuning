import os, sys, socket
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModelForSequenceClassification

from extreme_fine_tuning import *
from dataset import dataset


def main(seed=42, 
        hidden_size=1024, 
        epochs=3,
        save_path='save/',
        model_checkpoint='roberta-base',
        speaker_mode='upper',
        ROOT_DIR='dataset/',
        num_past_utterances=0,
        num_future_utterances=0,
        batch_size=2**4,
        batch_size_elm=2**8,
        device='cuda',
        DATASET='MELD',
        default_lr=0.00001,
        debug=False
    ):
    num_labels = dataset.get_num_classes(DATASET)

    dataset.set_seed(seed)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset_train_based = dataset.TextDataset(
        DATASET=DATASET,
        SPLIT='train',
        speaker_mode=speaker_mode,
        num_past_utterances=num_past_utterances,
        num_future_utterances=num_future_utterances,
        model_checkpoint=model_checkpoint,
        ROOT_DIR=ROOT_DIR,
        SEED=seed
    )
    train_data_loader_based = DataLoader(
        dataset_train_based,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: x
    )

    dataset_train = dataset.TextDataset(
        DATASET=DATASET,
        SPLIT='train',
        speaker_mode=speaker_mode,
        num_past_utterances=num_past_utterances,
        num_future_utterances=num_future_utterances,
        model_checkpoint=model_checkpoint,
        ROOT_DIR=ROOT_DIR,
        SEED=seed
    )
    train_data_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: x
    )

    dataset_train_elm = dataset.TextDataset(
        DATASET=DATASET,
        SPLIT='train',
        speaker_mode=speaker_mode,
        num_past_utterances=num_past_utterances,
        num_future_utterances=num_future_utterances,
        model_checkpoint=model_checkpoint,
        ROOT_DIR=ROOT_DIR,
        SEED=seed
    )
    train_data_loader_elm = DataLoader(
        dataset_train_elm,
        batch_size=batch_size_elm,
        shuffle=True,
        collate_fn=lambda x: x
    )

    dataset_var_based = dataset.TextDataset(
        DATASET=DATASET,
        SPLIT='val',
        speaker_mode=speaker_mode,
        num_past_utterances=num_past_utterances,
        num_future_utterances=num_future_utterances,
        model_checkpoint=model_checkpoint,
        ROOT_DIR=ROOT_DIR,
        SEED=seed
    )
    val_data_loader_based = DataLoader(
        dataset_var_based,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: x
    )

    dataset_val = dataset.TextDataset(
        DATASET=DATASET,
        SPLIT='val',
        speaker_mode=speaker_mode,
        num_past_utterances=num_past_utterances,
        num_future_utterances=num_future_utterances,
        model_checkpoint=model_checkpoint,
        ROOT_DIR=ROOT_DIR,
        SEED=seed
    )
    val_data_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: x
    )

    dataset_test_based = dataset.TextDataset(
        DATASET=DATASET,
        SPLIT='test',
        speaker_mode=speaker_mode,
        num_past_utterances=num_past_utterances,
        num_future_utterances=num_future_utterances,
        model_checkpoint=model_checkpoint,
        ROOT_DIR=ROOT_DIR,
        SEED=seed
    )
    test_data_loader_based = DataLoader(
        dataset_test_based,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: x
    )

    dataset_test = dataset.TextDataset(
        DATASET=DATASET,
        SPLIT='test',
        speaker_mode=speaker_mode,
        num_past_utterances=num_past_utterances,
        num_future_utterances=num_future_utterances,
        model_checkpoint=model_checkpoint,
        ROOT_DIR=ROOT_DIR,
        SEED=seed
    )
    test_data_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: x
    )

    based_model = BasedModel(
        AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels),
        num_labels=num_labels,
        lr=default_lr,
        debug=debug
    )
    for param in based_model.parameters():
        param.requires_grad = True
        
    trainer = Trainer(gpus=1, max_epochs=epochs, enable_checkpointing=False, logger=False)
    trainer.fit(based_model, train_data_loader_based)

    trainer.test(based_model, train_data_loader_based)
    trainer.test(based_model, val_data_loader_based)
    trainer.test(based_model, test_data_loader_based)

    if model_checkpoint == 'roberta-base' or model_checkpoint == 'roberta-large':
        features_extractor = based_model.model.roberta
        input_size = based_model.model.classifier.dense.in_features
    elif model_checkpoint == 'bert-base-uncased' or model_checkpoint == 'bert-large-uncased':
        features_extractor = based_model.model.bert
        input_size = based_model.model.classifier.in_features

    model = ExtremeFineTuning(
        features_extractor=features_extractor,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=num_labels,
        activation_function=nn.PReLU(init=0.5),
        dropout=nn.Dropout(p=0.1, inplace=False),
        lr=default_lr,
        debug=debug
    )
    for param in model.parameters():
        param.requires_grad = True

    ielm_trainer = ExtremeFineTuningTrainer(gpus=1, debug=debug)
    ielm_trainer.fit(model, train_data_loader_elm, force_hidden_size=True)

    trainer = Trainer(gpus=1, max_epochs=1)
    trainer.test(model, train_data_loader)
    trainer.test(model, val_data_loader)
    trainer.test(model, test_data_loader)

    with torch.no_grad():
        predict_output = list()
        label = list()
        for batch in test_data_loader:
            try:
                np_batch = np.array(list([e[0].tolist(), e[1].tolist(), e[2]] for e in batch))                
            except:
                np_batch = np.array(batch)
            y_pred = model(np_batch)
            predict_output.append(y_pred)
            label.append(list(e[2] for e in batch))
    

if __name__ == '__main__':
    import argparse
    
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("--seed", help="Set seed")
    parser.add_argument("--bpe", help="Set BP epochs")
    parser.add_argument("--nh", help="Set a number of hidden node")
    parser.add_argument("--cp", help="Set model checkpoint")
    parser.add_argument("--num-past-utterances", help="Set a number of past utterances")
    parser.add_argument("--num-future-utterances", help="Set a number of future utterances")
    parser.add_argument("--batch-size", help="Set batch size for BP")
    parser.add_argument("--batch-size-elm", help="Set batch size for ELM")
    parser.add_argument("--dataset", help="Set dataset")
    parser.add_argument("--default-lr", help="Set dataset")
    
    # Read arguments from command line
    args = parser.parse_args()
    
    if args.seed:
        seed = int(args.seed)
        epochs = int(args.bpe)
        hidden_size = int(args.nh)
        model_checkpoint = args.cp
        num_past_utterances = int(args.num_past_utterances or 0)
        num_future_utterances = int(args.num_future_utterances or 0)
        batch_size = int(args.batch_size)
        batch_size_elm = int(args.batch_size_elm)
        DATASET = args.dataset
        default_lr = float(args.default_lr)
        print(f"Seed: {seed}")
        print(f"BP epochs: {epochs}")
        print(f"Hidden size: {hidden_size}")
        print(f"Model checkpoint: {model_checkpoint}")
        print(f"Past utterances: {num_past_utterances}")
        print(f"Future utterances: {num_future_utterances}")
        print(f"Batch size (BP): {batch_size}")
        print(f"Batch size (ELM): {batch_size_elm}")
        print(f"Dataset: {DATASET}")
        print(f"Default learning rate: {default_lr}")
        main(seed=seed, 
            hidden_size=hidden_size, 
            epochs=epochs,
            model_checkpoint=model_checkpoint,
            num_past_utterances=num_past_utterances,
            num_future_utterances=num_future_utterances,
            batch_size=batch_size,
            batch_size_elm=batch_size_elm,
            device='cuda',
            DATASET=DATASET,
            default_lr=default_lr,
        )
    else:
        print("Invalid arguments!")
