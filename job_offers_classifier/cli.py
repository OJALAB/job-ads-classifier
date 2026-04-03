import random
from datetime import datetime

import click
import numpy as np

from job_offers_classifier.job_offers_utils import create_hierarchy
from job_offers_classifier.load_save import load_texts, load_to_df, save_as_text
from job_offers_classifier.runtime import normalize_threads, resolve_hardware


CLASSIFIERS = click.Choice(
    ["LinearJobOffersClassifier", "TransformerJobOffersClassifier"],
    case_sensitive=True,
)
COMMANDS = click.Choice(["fit", "predict"], case_sensitive=True)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_classifier(classifier):
    from job_offers_classifier.job_offers_classfier import (
        LinearJobOffersClassifier,
        TransformerJobOffersClassifier,
    )

    classifier_map = {
        "LinearJobOffersClassifier": LinearJobOffersClassifier,
        "TransformerJobOffersClassifier": TransformerJobOffersClassifier,
    }
    return classifier_map[classifier]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("command", type=COMMANDS)
@click.argument("classifier", type=CLASSIFIERS)
@click.option("-x", "--x-data", "--x_data", "x_data", type=str, required=True)
@click.option("-y", "--y-data", "--y_data", "y_data", type=str, default="")
@click.option("--hierarchy-data", "--hierarchy_data", "hierarchy_data", type=str, default="")
@click.option("-m", "--model-dir", "--model_dir", "model_dir", type=str, required=True)
@click.option("-t", "--transformer-model", "--transformer_model", "transformer_model", type=str, default="allegro/herbert-base-cased")
@click.option("-k", "--transformer-ckpt-path", "--transformer_ckpt_path", "transformer_ckpt_path", type=str, default="")
@click.option("-mm", "--modeling-mode", "--modeling_mode", "modeling_mode", type=str, default="bottom-up")
@click.option("-l", "--learning-rate", "--learning_rate", "learning_rate", type=float, default=1e-5)
@click.option("-w", "--weight-decay", "--weight_decay", "weight_decay", type=float, default=0.01)
@click.option("-e", "--max-epochs", "--max_epochs", "max_epochs", type=int, default=20)
@click.option("-b", "--batch-size", "--batch_size", "batch_size", type=int, default=64)
@click.option("-s", "--max-sequence-length", "--max_sequence_length", "max_sequence_length", type=int, default=128)
@click.option("--early-stopping", "--early_stopping", "early_stopping", type=bool, default=False)
@click.option("--early-stopping-delta", "--early_stopping_delta", "early_stopping_delta", type=float, default=0.001)
@click.option("--early-stopping-patience", "--early_stopping_patience", "early_stopping_patience", type=int, default=1)
@click.option("--pooling", type=click.Choice(["cls", "mean"], case_sensitive=True), default="cls")
@click.option("--gradient-checkpointing/--no-gradient-checkpointing", "gradient_checkpointing", default=False)
@click.option("--tokenization-mode", "tokenization_mode", type=click.Choice(["batched", "lazy"], case_sensitive=True), default="batched")
@click.option("-T", "--threads", type=int, default=8)
@click.option("-D", "--devices", type=int, default=1)
@click.option("-P", "--precision", type=str, default="16")
@click.option("-A", "--accelerator", type=str, default="auto")
@click.option("--eps", type=float, default=0.001)
@click.option("-c", "--cost", type=float, default=10)
@click.option("-n", "--ensemble", type=int, default=1)
@click.option("--use-provided-hierarchy", "--use_provided_hierarchy", "use_provided_hierarchy", type=int, default=1)
@click.option("--tfidf-vectorizer-min-df", "--tfidf_vectorizer_min_df", "tfidf_vectorizer_min_df", type=int, default=2)
@click.option("-p", "--pred-path", "--pred_path", "pred_path", type=str, default="")
@click.option("-r", "--seed", type=int, default=1993)
@click.option("-v", "--verbose", type=bool, default=True)
def main(command,
         classifier,
         x_data,
         y_data,
         hierarchy_data,
         model_dir,
         transformer_model,
         transformer_ckpt_path,
         modeling_mode,
         learning_rate,
         weight_decay,
         max_epochs,
         batch_size,
         max_sequence_length,
         early_stopping,
         early_stopping_delta,
         early_stopping_patience,
         pooling,
         gradient_checkpointing,
         tokenization_mode,
         threads,
         devices,
         precision,
         accelerator,
         eps,
         cost,
         ensemble,
         use_provided_hierarchy,
         tfidf_vectorizer_min_df,
         pred_path,
         seed,
         verbose):

    if command == "fit":
        if not y_data:
            raise click.UsageError("--y-data is required for fit")
        if not hierarchy_data:
            raise click.UsageError("--hierarchy-data is required for fit")
    elif command == "predict" and not pred_path:
        raise click.UsageError("--pred-path is required for predict")

    threads = normalize_threads(threads)
    _seed_everything(seed)

    runtime = {
        "threads": threads,
        "devices": devices,
        "precision": precision,
        "accelerator": accelerator,
    }

    if classifier == "TransformerJobOffersClassifier":
        runtime.update(resolve_hardware(accelerator=accelerator, devices=devices, precision=precision))
        if verbose and runtime["accelerator"] == "cpu" and runtime["requested_accelerator"] not in {"cpu", "CPU"}:
            click.echo("CUDA accelerator was not available, falling back to CPU.")

    classifier_class = _resolve_classifier(classifier)

    click.echo(f"Starting command {command} with {classifier}, time: {datetime.now()}")

    if command == "fit":
        hierarchy = create_hierarchy(load_to_df(hierarchy_data))
        X = load_texts(x_data)
        y = load_texts(y_data)

        if classifier == "LinearJobOffersClassifier":
            model = classifier_class(
                model_dir=model_dir,
                hierarchy=hierarchy,
                eps=eps,
                c=cost,
                use_provided_hierarchy=bool(use_provided_hierarchy),
                ensemble=ensemble,
                threads=runtime["threads"],
                tfidf_vectorizer_min_df=tfidf_vectorizer_min_df,
                verbose=verbose,
            )
        else:
            model = classifier_class(
                model_dir=model_dir,
                hierarchy=hierarchy,
                transformer_model=transformer_model,
                transformer_ckpt_path=transformer_ckpt_path,
                modeling_mode=modeling_mode,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_epochs=max_epochs,
                batch_size=batch_size,
                max_sequence_length=max_sequence_length,
                early_stopping=early_stopping,
                early_stopping_delta=early_stopping_delta,
                early_stopping_patience=early_stopping_patience,
                pooling=pooling,
                gradient_checkpointing=gradient_checkpointing,
                tokenization_mode=tokenization_mode,
                devices=runtime["devices"],
                accelerator=runtime["accelerator"],
                threads=runtime["threads"],
                precision=runtime["precision"],
                verbose=verbose,
            )

        model.fit(y, X)

    else:
        X = load_texts(x_data)

        if classifier == "LinearJobOffersClassifier":
            model = classifier_class(threads=runtime["threads"], verbose=verbose)
        else:
            model = classifier_class(
                batch_size=batch_size,
                devices=runtime["devices"],
                threads=runtime["threads"],
                precision=runtime["precision"],
                accelerator=runtime["accelerator"],
                pooling=pooling,
                gradient_checkpointing=gradient_checkpointing,
                tokenization_mode=tokenization_mode,
                verbose=verbose,
            )

        model.load(model_dir)
        pred, pred_map = model.predict(X)
        np.savetxt(pred_path, pred)
        save_as_text(f"{pred_path}.map", pred_map.values())

    click.echo("All done")


if __name__ == "__main__":
    main()
