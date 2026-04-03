import torch


class DynamicPaddingCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        token_features = []
        labels = []

        for feature in features:
            item = dict(feature)
            labels.append(item.pop("labels", None))
            token_features.append(item)

        batch = self.tokenizer.pad(
            token_features,
            padding=True,
            return_tensors="pt",
        )

        if labels and labels[0] is not None:
            first_label = labels[0]
            if isinstance(first_label, torch.Tensor):
                batch["labels"] = torch.stack([
                    label if isinstance(label, torch.Tensor) else torch.as_tensor(label)
                    for label in labels
                ])
            else:
                batch["labels"] = torch.as_tensor(labels)
        else:
            batch["labels"] = None

        return batch
