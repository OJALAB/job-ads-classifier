from typing import List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
try:
    from lightning.pytorch import LightningModule
except ImportError:  # pragma: no cover - compatibility fallback
    from pytorch_lightning import LightningModule
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup
from torchmetrics import Accuracy, MetricCollection, Recall

from job_offers_classifier.classification_utils import *

class FullyConnectedOutput(nn.Module):
    def __init__(self, input_size, output_size, layer_units=(10,), nonlin=nn.ReLU(), hidden_dropout=0, output_nonlin=nn.Softmax(dim=1), criterion=nn.CrossEntropyLoss(), labels_groups=None, labels_groups_mapping=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nonlin = nonlin
        self.layer_units = layer_units
        self.output_nonlin = output_nonlin
        self.criterion = criterion
        self.hidden_dropout = hidden_dropout
        self.labels_groups = labels_groups
        self.labels_groups_mapping = labels_groups_mapping

        if labels_groups_mapping is not None:
            self.register_buffer('_labels_groups_mapping', torch.tensor(labels_groups_mapping, dtype=torch.long))
            self.register_buffer('_labels_groups_mask_mapping', self._labels_groups_mapping != -1)
            self._labels_groups_mapping[~self._labels_groups_mask_mapping] += 1

        sequence = []
        units = [self.input_size] + list(self.layer_units) + [self.output_size]
        for in_size, out_size in zip(units, units[1:]):
            sequence.extend([nn.Linear(in_size, out_size), self.nonlin, nn.Dropout(self.hidden_dropout)])

        sequence = sequence[:-2]
        self.sequential = nn.Sequential(*sequence)

    def forward(self, batch, labels=None):
        logits_output = self.sequential(batch)
            
        if self.labels_groups is None:
            output = self.output_nonlin(logits_output)
            if labels is not None:
                return self.criterion(logits_output, labels).mean(), output
            else:
                return output
            
        else:
            output = torch.zeros_like(logits_output, dtype=torch.float32)
            for group in self.labels_groups:
                output[:, group] = self.output_nonlin(logits_output[:, group])
        
            if labels is not None:
                labels_groups = self._labels_groups_mapping[labels]
                labels_groups_mask = self._labels_groups_mask_mapping[labels]
                criterion = 0
                for g_idx, group in enumerate(self.labels_groups):
                    assert (labels_groups[:, g_idx] < len(group)).all()
                    criterion += self.criterion(logits_output[:, group], labels_groups[:, g_idx]) * labels_groups_mask[:, g_idx]
                criterion = criterion.sum() / labels_groups_mask.sum()
                return criterion, output
            else:
                return output
            

class TransformerClassifier(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        output_size: int,
        labels_groups: Optional[List] = None,
        labels_paths: Optional[List] = None,
        labels_groups_mapping: Optional[np.ndarray] = None,
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: float = 50,  # from 0-1 for % of training steps, >1 for number of steps
        weight_decay: float = 0.01,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        hidden_dropout: float = 0.0,
        eval_top_k: int = 10,
        freeze_transformer: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.eval_num_labels = output_size if labels_paths is None else len(labels_paths)

        if self.hparams.verbose:
            print(f"""Initializing TransformerClassifier with model_name={model_name_or_path}, 
  output_size={output_size}, labels_groups={labels_groups is not None}, 
  learning_rate={learning_rate}, weight_decay={weight_decay}, warmup_steps={warmup_steps}, train_batch_size={train_batch_size}, eval_batch_size={eval_batch_size}, freeze_transformer={freeze_transformer}
  loss={'hierarchical cross entropy' if labels_groups is not None else 'flat cross entropy'}, hidden_dropout={hidden_dropout}, hierarchy_leaves={self.eval_num_labels} ...""")

        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            finetuning_task=None
        )
        self.transformer = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.output = FullyConnectedOutput(self.config.hidden_size, output_size, layer_units=(), hidden_dropout=hidden_dropout, output_nonlin=nn.Softmax(dim=1), criterion=nn.CrossEntropyLoss(reduction='none'), labels_groups=labels_groups, labels_groups_mapping=labels_groups_mapping) 
        
        metric_dict = {
            "acc_at_1": Accuracy(task="multiclass", num_classes=self.eval_num_labels),
            "macro_acc": Accuracy(task="multiclass", num_classes=self.eval_num_labels, average='macro'),
        }

        for i in range(2, min(self.eval_num_labels, eval_top_k + 1)):
            metric_dict[f"recall_at_{i}"] = Recall(task="multiclass", num_classes=self.eval_num_labels, top_k=i)

        self.val_metrics = MetricCollection(metric_dict, prefix="val_")
        self.test_metrics = MetricCollection(metric_dict, prefix="test_")

    def forward(self, batch, labels=None):
        transformer_output = self.transformer(batch['input_ids'], attention_mask=batch['attention_mask'])
        transformer_output = transformer_output.last_hidden_state[:, 0, :]
        if self.output is None:
            return transformer_output
        else:
            return self.output.forward(transformer_output, labels=labels)
        
    def training_step(self, batch, batch_idx):
        if batch['labels'] is not None:  # Windows fix
            batch['labels'] = batch['labels'].long().to(self.device)
    
        loss, scores = self.forward(batch, batch['labels'])
        self.log('train_loss', loss, on_epoch=True, logger=True)
        return loss

    def _eval_step(self, batch, eval_name='val'):
        if batch['labels'] is not None:  # Windows fix
            batch['labels'] = batch['labels'].long().to(self.device)
    
        loss, scores = self.forward(batch, batch['labels'])
        if self.hparams.labels_paths is not None:
            new_scores = torch.zeros((scores.shape[0], self.eval_num_labels)).to(self.device)
            for i, path in enumerate(self.hparams.labels_paths):
                path_score = scores[:,path]
                new_scores[:,i] = path_score.prod(axis=1)
            scores = new_scores
            assert scores.shape == (scores.shape[0], self.eval_num_labels)
            true_loss = F.cross_entropy(scores, batch['labels'])
            self.log(f'{eval_name}_true_loss', true_loss, on_epoch=True, logger=True)

        metrics = self.val_metrics if eval_name == "val" else self.test_metrics
        self.log_dict(metrics(scores, batch['labels']), on_epoch=True, on_step=False, logger=True)
        self.log(f'{eval_name}_loss', loss, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, eval_name='val')

    @staticmethod
    def _format_metric_values(metrics):
        formatted = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                formatted[key] = value.item() if value.numel() == 1 else value.tolist()
            else:
                formatted[key] = value
        return formatted

    def on_validation_epoch_end(self):
        if self.hparams.verbose:
            print("Validation performance:")
            print(self._format_metric_values(self.val_metrics.compute()))
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, eval_name='test')

    def on_test_epoch_end(self):
        if self.hparams.verbose:
            print("Test performance:")
            print(self._format_metric_values(self.test_metrics.compute()))
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self.forward(batch)
        if isinstance(output, tuple):
            _, output = output
        return output.detach().cpu()

    def load_state_dict(self, state_dict, strict=True):
        metric_prefixes = ("metrics.", "val_metrics.", "test_metrics.")
        has_legacy_metric_keys = any(key.startswith(metric_prefixes) for key in state_dict)

        if has_legacy_metric_keys:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith(metric_prefixes)
            }
            strict = False

        return super().load_state_dict(state_dict, strict=strict)

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return

        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        ab_size = self.trainer.accumulate_grad_batches
        self.total_steps = max(1, len(train_loader) * self.trainer.max_epochs // max(1, ab_size))

        self.num_warmup_steps = self.hparams.warmup_steps
        if self.hparams.warmup_steps < 1:
            self.num_warmup_steps = int(self.total_steps * self.hparams.warmup_steps)

        if self.hparams.verbose:
            print("Warmup_steps:", self.num_warmup_steps)
            print("Total steps:", self.total_steps)

    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer = AdamW(
            self._get_optimizer_grouped_parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def save_transformer(self, ckpt_dir):
        self.transformer.save_pretrained(ckpt_dir)

    def _get_optimizer_grouped_parameters(self, layer_wise_lr=False, layer_wise_lr_mutli=1.1):
        # It is suggested to not use any decay for bias, LayerNorm.weight and LayerNorm.weight layers.
        no_decay = ["bias", "LayerNorm.weight"]

        if layer_wise_lr:
            optimizer_grouped_parameters = []
            for name, params in self.named_parameters():
                weight_decay = 0.0 if any(nd in name for nd in no_decay) else self.hparams.weight_decay
                learning_rate = self.hparams.learning_rate

                if 'embeddings' in name or 'encoder' in name:
                    learning_rate /= 10

                    for i in range(0, 20):
                        if f'layer.{i}' in name:
                            learning_rate *= layer_wise_lr_mutli ** (i + 1)

                print(name, learning_rate)
                optimizer_grouped_parameters.append({
                    "params": params,
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                })

        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

        return optimizer_grouped_parameters
