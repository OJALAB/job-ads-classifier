try:
    from lightning.pytorch import LightningDataModule
except ImportError:  # pragma: no cover - compatibility fallback
    from pytorch_lightning import LightningDataModule

from torch.utils.data import Dataset, DataLoader
from job_offers_classifier.collators import DynamicPaddingCollator

class TransformerDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        model_name_or_path: str,
        max_seq_length: int = 256,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        predict_batch_size: int | None = None,
        num_workers: int = 0,
        num_workers_train: int | None = None,
        num_workers_eval: int | None = None,
        num_workers_predict: int | None = None,
        pin_memory: bool = False,
        persistent_workers: bool | None = None,
        shuffle_train: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dataset = dataset
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.predict_batch_size = predict_batch_size or eval_batch_size
        self.num_workers = num_workers
        self.num_workers_train = num_workers if num_workers_train is None else num_workers_train
        self.num_workers_eval = num_workers if num_workers_eval is None else num_workers_eval
        self.num_workers_predict = num_workers if num_workers_predict is None else num_workers_predict
        self.pin_memory = pin_memory
        self.persistent_workers = num_workers > 0 if persistent_workers is None else persistent_workers
        self.train_shuffle = shuffle_train
        self.tokenizer = None
        self.collator = None
        self.verbose = verbose

        if self.verbose:
            print(f"Initializing TransformerDataModule with model_name={model_name_or_path}, max_seq_length={max_seq_length}, train/eval/predict_batch_size={train_batch_size}/{eval_batch_size}/{self.predict_batch_size}, num_workers(train/eval/predict)={self.num_workers_train}/{self.num_workers_eval}/{self.num_workers_predict} ...")

    def _get_dataloader(self, dataset_key, batch_size=32, shuffle=False, num_workers=0):
        if dataset_key in self.dataset:
            return DataLoader(self.dataset[dataset_key],
                              batch_size=batch_size,
                              num_workers=num_workers,
                              persistent_workers=self.persistent_workers and num_workers > 0,
                              pin_memory=self.pin_memory,
                              collate_fn=self.collator,
                              shuffle=shuffle)
        else:
            return None

    def setup(self, stage=None):
        if self.verbose:
            print("Setting up TransformerDataModule ...")

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.collator = DynamicPaddingCollator(self.tokenizer)
        for subset in self.dataset.values():
            subset.setup(self.tokenizer, self.max_seq_length)

    def train_dataloader(self):
        return self._get_dataloader("train", batch_size=self.train_batch_size, shuffle=self.train_shuffle, num_workers=self.num_workers_train)

    def val_dataloader(self):
        return self._get_dataloader("val", batch_size=self.eval_batch_size, num_workers=self.num_workers_eval)

    def test_dataloader(self):
        return self._get_dataloader("test", batch_size=self.eval_batch_size, num_workers=self.num_workers_eval)

    def predict_dataloader(self):
        return self._get_dataloader("test", batch_size=self.predict_batch_size, num_workers=self.num_workers_predict)
