import numpy as np
import time
import torch
import torchmetrics
from pytorch_lightning import LightningModule, Trainer
from torch import nn


class BasedModel(LightningModule):
    def __init__(self, model, num_labels, lr=1e-3, criterion=torch.nn.CrossEntropyLoss(), debug=False):
        super(BasedModel, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.lr = lr
        self.criterion = criterion
        self.debug = debug

        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.test_accuracy_ = ''
        self.train_weighted_f1score = torchmetrics.F1Score(num_classes=self.num_labels, average='weighted')
        self.test_weighted_f1score = torchmetrics.F1Score(num_classes=self.num_labels, average='weighted')

    def forward(self, batch):        
        if type(batch[:, 0][0]) == type(list()):
            max_seq = len(max(batch[:, 0], key=lambda input_id: len(input_id)))
            for idx in range(len(batch[:, 0])):
                batch[:, 0][idx] = np.pad(
                    batch[:, 0][idx],
                    (0, max_seq - len(batch[:, 0][idx])),
                    'constant',
                    constant_values=(0, 0)
                )
                batch[:, 1][idx] = np.pad(
                    batch[:, 1][idx],
                    (0, max_seq - len(batch[:, 1][idx])),
                    'constant',
                    constant_values=(0, 0)
                )
            input_ids = np.stack(batch[:, 0])
            attention_mask = np.stack(batch[:, 1])
            input_ids = torch.from_numpy(input_ids).to(self.device)
            attention_mask = torch.from_numpy(attention_mask).to(self.device)
        else:
            max_seq = max(batch[:, 0], key=lambda input_id: input_id.shape).shape[0]
            input_ids = list()
            attention_mask = list()
            for idx in range(len(batch[:, 0])):
                input_ids.append(torch.nn.functional.pad(
                    batch[:, 0][idx],
                    (0, max_seq - batch[:, 0][idx].shape[0]),
                    'constant'
                ))
                attention_mask.append(torch.nn.functional.pad(
                    batch[:, 1][idx],
                    (0, max_seq - batch[:, 1][idx].shape[0]),
                    'constant'
                ))
            input_ids = torch.stack(input_ids)
            attention_mask = torch.stack(attention_mask)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def step_helper(self, batch):
        batch = np.array(batch, dtype=object)
        label = np.stack(batch[:, 2])
        label_one_hot = np.zeros((label.size, self.num_labels), dtype=np.float32)
        label_one_hot[np.arange(label.size), label] = 1
        label_one_hot = torch.from_numpy(label_one_hot).to(self.device)
        output = self(batch)
        return output, label_one_hot

    def training_step(self, batch, batch_idx):
        output, label_one_hot = self.step_helper(batch)
        loss = self.criterion(output, label_one_hot)

        prediction = torch.argmax(output, dim=1).type(torch.IntTensor).to(self.device)
        actual = torch.argmax(label_one_hot, dim=1).type(torch.IntTensor).to(self.device)

        if self.debug:
            self.train_accuracy(prediction, actual)
            self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=False, prog_bar=True)
            self.train_weighted_f1score(prediction, actual)
            self.log('train_weighted_f1score', self.train_weighted_f1score, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            output, label_one_hot = self.step_helper(batch)

            prediction = torch.argmax(output, dim=1).type(torch.IntTensor).to(self.device)
            actual = torch.argmax(label_one_hot, dim=1).type(torch.IntTensor).to(self.device)

            self.test_accuracy(prediction, actual)
            self.test_accuracy_ = f'{self.test_accuracy.tp/(self.test_accuracy.tp+self.test_accuracy.fn)}'
            self.log('test_accuracy', self.test_accuracy, on_step=True, on_epoch=True, prog_bar=True)
            self.test_weighted_f1score(prediction, actual)
            self.log('test_weighted_f1score', self.test_weighted_f1score, on_step=True, on_epoch=True, prog_bar=True)


class ExtremeFineTuning(LightningModule):
    def __init__(self, features_extractor, input_size, hidden_size, output_size,
                 input_weights=None, output_weights=None, activation_function=None, dropout=None,
                 lr=1e-3, criterion=torch.nn.CrossEntropyLoss(), debug=False):
        super(ExtremeFineTuning, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = self.output_size = output_size
        self.activation_function = activation_function
        self.lr = lr
        self.criterion = criterion

        self.features_extractor = features_extractor
        self.classifier = ExtremeFineTuningClassificationHead(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation_function=activation_function,
            dropout=dropout,
            input_weights=input_weights,
            output_weights=output_weights
        )

        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.test_accuracy_ = ''
        self.train_weighted_f1score = torchmetrics.F1Score(num_classes=self.num_labels, average='weighted')
        self.test_weighted_f1score = torchmetrics.F1Score(num_classes=self.num_labels, average='weighted')

        self.debug = debug

    def forward(self, batch):        
        if type(batch[:, 0][0]) == type(list()):
            max_seq = len(max(batch[:, 0], key=lambda input_id: len(input_id)))
            for idx in range(len(batch[:, 0])):
                batch[:, 0][idx] = np.pad(
                    batch[:, 0][idx],
                    (0, max_seq - len(batch[:, 0][idx])),
                    'constant',
                    constant_values=(0, 0)
                )
                batch[:, 1][idx] = np.pad(
                    batch[:, 1][idx],
                    (0, max_seq - len(batch[:, 1][idx])),
                    'constant',
                    constant_values=(0, 0)
                )
            input_ids = np.stack(batch[:, 0])
            attention_mask = np.stack(batch[:, 1])
            input_ids = torch.from_numpy(input_ids).to(self.device)
            attention_mask = torch.from_numpy(attention_mask).to(self.device)
        else:
            max_seq = max(batch[:, 0], key=lambda input_id: input_id.shape).shape[0]
            input_ids = list()
            attention_mask = list()
            for idx in range(len(batch[:, 0])):
                input_ids.append(torch.nn.functional.pad(
                    batch[:, 0][idx],
                    (0, max_seq - batch[:, 0][idx].shape[0]),
                    'constant'
                ))
                attention_mask.append(torch.nn.functional.pad(
                    batch[:, 1][idx],
                    (0, max_seq - batch[:, 1][idx].shape[0]),
                    'constant'
                ))
            input_ids = torch.stack(input_ids).to(self.device)
            attention_mask = torch.stack(attention_mask).to(self.device)

        outputs = self.features_extractor(
            input_ids,
            attention_mask=attention_mask,
        )

        features = outputs[0][:, 0, :]

        output = self.classifier(features)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def step_helper(self, batch):
        batch = np.array(batch, dtype=object)
        label = np.stack(batch[:, 2])
        label_one_hot = np.zeros((label.size, self.num_labels), dtype=np.float64)
        label_one_hot[np.arange(label.size), label] = 1
        label_one_hot = torch.from_numpy(label_one_hot).to(self.device)
        output = self(batch)
        return output, label_one_hot

    def training_step(self, batch, batch_idx):
        output, label_one_hot = self.step_helper(batch)
        loss = self.criterion(output, label_one_hot)

        prediction = torch.argmax(output, dim=1).type(torch.IntTensor).to(self.device)
        actual = torch.argmax(label_one_hot, dim=1).type(torch.IntTensor).to(self.device)

        if self.debug:
            self.train_accuracy(prediction, actual)
            self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=False, prog_bar=True)
            self.train_weighted_f1score(prediction, actual)
            self.log('train_weighted_f1score', self.train_weighted_f1score, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            output, label_one_hot = self.step_helper(batch)

            prediction = torch.argmax(output, dim=1).type(torch.IntTensor).to(self.device)
            actual = torch.argmax(label_one_hot, dim=1).type(torch.IntTensor).to(self.device)

            self.test_accuracy(prediction, actual)
            self.test_accuracy_ = f'{self.test_accuracy.tp / (self.test_accuracy.tp + self.test_accuracy.fn)}'
            self.log('test_accuracy', self.test_accuracy, on_step=True, on_epoch=True, prog_bar=True)
            self.test_weighted_f1score(prediction, actual)
            self.log('test_weighted_f1score', self.test_weighted_f1score, on_step=True, on_epoch=True, prog_bar=True)


class ExtremeFineTuningClassificationHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_function=None, dropout=None, elm_mode=False,
                 input_weights=None, output_weights=None):
        super(ExtremeFineTuningClassificationHead, self).__init__()
        if input_weights is not None:
            self.input_weights = input_weights
        else:
            self.input_weights = nn.Linear(input_size, hidden_size)
        if output_weights is not None:
            self.output_weights = output_weights
        else:
            self.output_weights = nn.Linear(hidden_size, output_size)
        if not activation_function:
            self.activation_function = nn.Tanh()
        else:
            self.activation_function = activation_function
        self.dropout = dropout
        self.elm_mode = elm_mode

    def forward(self, features, **kwargs):
        x = features
        if not self.elm_mode:
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.input_weights(x)
            x = self.activation_function(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.output_weights(x)
        return x

    def set_elm_mode(self):
        self.elm_mode = True

    def unset_elm_mode(self):
        self.elm_mode = False


class ExtremeFineTuningTrainer(Trainer):
    def __init__(self, debug=False, *args, **kwargs):
        super(ExtremeFineTuningTrainer, self).__init__(*args, **kwargs)
        self.debug = debug #TODO
        self.device = 'cuda' if self.gpus else 'cpu'

        self.model = None
        self.l2_gamma = None
        self.l2_weight = None
        self.input_weights = None

        self.input_size = None
        self.hidden_size = None
        self.output_size = None

        self.input_weights = None
        self.input_bias = None
        self.activation_function = None

        self.output_weights = None
        self.H = None
        self.Gamma = None
        self.Tau = None

    def fit(self, model, train_dataloaders=None, val_dataloaders=None, datamodule=None, ckpt_path=None,
            l2_gamma=1., reset=True, force_hidden_size=False):
        self.model = model.to(self.device)
        self.l2_gamma = l2_gamma

        if reset:
            self.input_size = self.model.classifier.input_weights.in_features
            self.hidden_size = self.model.hidden_size if force_hidden_size else \
                self.model.classifier.input_weights.out_features
            self.output_size = self.model.classifier.output_weights.out_features

            self.input_weights = torch.nn.init.orthogonal_(torch.empty(
                self.hidden_size, self.input_size), gain=1).type(torch.float64).to(self.device)
            self.input_bias = torch.nn.init.uniform_(torch.empty(self.hidden_size)).type(torch.float64).to(self.device)
            self.input_bias = self.input_bias / torch.norm(self.input_bias)
            self.activation_function = self.model.classifier.activation_function

            self.H = None
            self.Gamma = None
            self.Tau = None
            self.n_Gamma = None
            self.n_Tau = None
            self.output_weights = None
            self.l2_weight = 0.1

        start = time.time()

        with torch.no_grad():
            self.model.classifier.set_elm_mode()
            for batch_idx, batch in enumerate(train_dataloaders):
                input_features, label_one_hot = self.model.step_helper(batch)
                self._iterative_elm(input_features.type(torch.float64).to(self.device).T, label_one_hot.T)
                if self.debug:
                    print(f'\rI-ELM Batch#{batch_idx}', end='')

            new_input_weights = torch.nn.Linear(
                in_features=self.input_size,
                out_features=self.hidden_size,
                bias=True,
                device=model.device,
                dtype=model.dtype
            )
            new_input_weights.weight.copy_(self.input_weights)
            new_input_weights.bias.copy_(self.input_bias)

            new_output_weights = torch.nn.Linear(
                in_features=self.hidden_size,
                out_features=self.output_size,
                bias=False,
                device=model.device,
                dtype=model.dtype
            )
            new_output_weights.weight.copy_(self.output_weights)

            model.classifier.input_weights = new_input_weights
            model.classifier.output_weights = new_output_weights

            self.model.classifier.unset_elm_mode()

        end = time.time()
        print(f'\nI-ELM Training Time: {end - start}')

    def _iterative_elm(self, input_features, label_one_hot):
        ones = torch.ones(input_features.shape[1]).to(self.device)
        self.H = (self.input_weights @ input_features + torch.outer(self.input_bias, ones)).type(torch.float32).to(self.device)
        self.H = self.activation_function(self.H).type(torch.float64).to(self.device)
        if self.Gamma is not None or self.Tau is not None:
            self.Gamma += (self.H @ self.H.T)
            self.Tau += (label_one_hot @ self.H.T)
        else:
            self.Gamma = self.H @ self.H.T
            self.Tau = label_one_hot @ self.H.T

        weighted_gamma = (torch.eye(self.H.shape[0], dtype=torch.float64).to(self.device) / self.l2_weight) @ self.Gamma
        inverted_weighted_gamma = np.linalg.inv(weighted_gamma.cpu())
        self.output_weights = self.Tau @ torch.tensor(inverted_weighted_gamma).to(self.device)
