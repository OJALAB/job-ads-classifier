import os
from datetime import datetime
import click
import numpy as np

from job_offers_classifier.datasets import *
from job_offers_classifier.job_offers_classfier import *
from job_offers_classifier.job_offers_utils import *
from job_offers_classifier.load_save import *


@click.command()
# Command
@click.argument("command", type=str)
@click.argument("classifier", type=str)
# General settings
@click.option("-x", "--x_data", type=str, required=True)
@click.option("-y", "--y_data", type=str, required=True, default="")
@click.option("-h", "--hierarchy_data", type=str, required=True, default="")
@click.option("-m", "--model_dir", type=str, required=True, default="model")
# Transformer model settings
@click.option("-t", "--transformer_model", type=str, required=True, default="allegro/herbert-base-cased")
@click.option("-tc", "--transformer_ckpt_path", type=str, required=True, default="")
@click.option("-mm", "--modeling_mode", type=str, required=True, default="bottom-up")
# Training parameters
@click.option("-l", "--learning_rate", type=float, required=True, default=1e-5)
@click.option("-w", "--weight_decay", type=float, required=True, default=0.01)
@click.option("-e", "--max_epochs", type=int, required=True, default=20)
@click.option("-b", "--batch_size", type=int, required=True, default=64)
@click.option("-s", "--max_sequence_length", type=int, required=True, default=128)
# Early stopping
@click.option("--early_stopping", type=bool, required=True, default=False)
@click.option("--early_stopping_delta", type=float, required=True, default=0.001)
@click.option("--early_stopping_patience", type=int, required=True, default=1)
# Hardware
@click.option("-T", "--threads", type=int, required=True, default=8)
@click.option("-D", "--devices", type=int, required=True, default=1)
@click.option("-P", "--precision", type=int, required=True, default=16)
@click.option("-A", "--accelerator", type=str, required=True, default="auto")
# Linear model
@click.option("--eps", type=float, required=True, default=0.001)
@click.option("-c", "--cost", type=float, required=True, default=10)
@click.option("-e", "--ensemble", type=int, required=True, default=1)
@click.option("--use_provided_hierarchy", type=int, required=True, default=1)
@click.option("--tfidf_vectorizer_min_df", type=int, required=True, default=2)
# Prediction
@click.option("-p", "--pred_path", type=str, required=True, default="")
@click.option("-s", "--seed", type=int, required=True, default=1993)
@click.option("-v", "--verbose", type=bool, required=True, default=True)
def main(command: str,
         classifier: str,
         x_data: str,
         y_data: str,
         hierarchy_data: str,
         model_dir: str,

         transformer_model: str,
         transformer_ckpt_path: str,
         modeling_mode: str,

         learning_rate: float,
         weight_decay: float,
         max_epochs: int,
         batch_size: int,
         max_sequence_length: int,

         early_stopping: bool,
         early_stopping_delta: float,
         early_stopping_patience: int,

         threads: int,
         devices: int,
         precision: int,
         accelerator: str,

         eps: float,
         cost: float,
         ensemble: int,
         use_provided_hierarchy: int,
         tfidf_vectorizer_min_df: int,

         pred_path: str,
         seed: int,
         verbose: int,
         ):

    if threads <= 0:
        threads = min(os.cpu_count() - threads, 1)
    devices = min(devices, torch.cuda.device_count())
    print(f"Starting command {command} with {classifier}, time: {datetime.now()}")

    if command == 'fit':
        # Load data
        hierarchy_df = load_to_df(hierarchy_data)
        hierarchy = create_hierarchy(hierarchy_df)

        X = load_texts(x_data)
        y = load_texts(y_data)

        # Create model
        if classifier == "LinearJobOffersClassifier":
            model = LinearJobOffersClassifier(
                model_dir=model_dir,  # folder gdzie wszystkie elementy modelu będą zapisywane
                hierarchy=hierarchy,  # hierarchia klas w formacie słownika <etykieta>: {'label': <etykieta>, 'level': <numer poziomu hierarchii, 'name': <nazwa etykiety> (opcjonalne), 'parents': <lista zawierająca wszystkich rodziców etykiety>}
                eps=eps,  # warunek stopu uczenia
                c=cost,  # kontroluje regularyzacje, większa wartość = mniejsza regularyzacja
                use_provided_hierarchy=use_provided_hierarchy,  # jeżeli ensemble = 1 i use_provided_hierarchy = True, zostanie użyta podana hierarchia w agumencie hierarchy
                ensemble=ensemble,  # zespół ilu klasyfikatorów użyć, jeśli wartość > 1, zostanie użyty zespół klasyfikatorów, które same spróbują okryć dobrą hierarchię klass aby wprowadzić element losowy
                threads=threads,  # ilość wątków procesora wykorzystywana przy uczeniu i predykcji
                tfidf_vectorizer_min_df=tfidf_vectorizer_min_df,  # minimalna ilość wystąpień token w zbiorze treningowym
                verbose=verbose
            )
        elif classifier == "TransformerJobOffersClassifier":
            model = TransformerJobOffersClassifier(
                model_dir=model_dir,  # folder gdzie wszystkie elementy modelu będą zapisywane
                hierarchy=hierarchy,  # hierarchia klas w formacie słownika <etykieta>: {'label': <etykieta>, 'level': <numer poziomu hierarchii, 'name': <nazwa etykiety> (opcjonalne), 'parents': <lista zawierająca wszystkich rodziców etykiety>})
                transformer_model=transformer_model,  # podstawowy model transformera (encodera) do użycia, może być to dowolny model pochodzący z repo https://huggingface.co
                transformer_ckpt_path=transformer_ckpt_path,  # ścieżka do modelu transformera, od której rozpocząć uczenie, może służyć do tego, by szybciej trenować nowy model na podstawie już wytrenowanego, np. kiedy zmieni się hierarchia ogłoszeń
                modeling_mode=modeling_mode,  # `cascade_loss` albo `flat_loss`, pierwszy uczy poziom po poziomie w sumie przez max_epochs (max_epochs / liczba poziomów) per poziom, drugi uczy model na ostatnim poziomie hierarchi
                learning_rate=learning_rate,  # początkowy rozmiar kroku uczenia
                weight_decay=weight_decay,  # stała regularyzacyjna, im większa, tym większa regularyzacja
                max_epochs=max_epochs,  # maksymalna ilość epok uczenia
                batch_size=batch_size,  # wielkość batcha podczas trenowania/testowana, większy rozmiar wymaga więcej pamięci
                max_sequence_length=max_sequence_length,  # ilość słów od początku tekstu, która zostanie uwzględniona przez klasyfikator (512 to maksymalna ilość dla tej architektury), im większa wartość tym większe zapotrzebowanie pamięciowe i wolniejsze uczenie/predykcja
                early_stopping=early_stopping,  # zastosuj wcześniejsze kończenie treningu, jeśli nie zostanie osiągnięta wystarczająca poprawa na stracie
                early_stopping_delta=early_stopping_delta,  # próg tolerancja na warunek stopu
                early_stopping_patience=early_stopping_patience,  # zakończ po tej ilości epok bez poprawy o próg tolerancji
                devices=devices,  # id akceleratora gpu, który użyć podczas treningu, przy większej ilość gpu, można podać listę, obliczenia powinny się rozproszyć, ale w tym wypadku tego nie testowałem
                accelerator=accelerator,
                threads=threads,  # ilość wątków procesora wykorzystana przy uczeniu i predykcji
                precision=precision,  # precyzja obliczeń na GPU, niższa precyzja (16 bitów) pozwala na szybsze uczenie większego modelu
                verbose=verbose
            )
        else:
            raise ValueError(f'Unknown classifier type {classifier}')

        model.fit(y, X)

    elif command == 'predict':
        X = load_texts(x_data)

        if classifier == "LinearJobOffersClassifier":
            model = LinearJobOffersClassifier(
                threads=threads,
            )
        elif classifier == "TransformerJobOffersClassifier":
            model = TransformerJobOffersClassifier(
                batch_size=batch_size,
                devices=devices,
                threads=threads,
                precision=precision,
                accelerator=accelerator,
            )
        else:
            raise ValueError(f'Unknown classifier type f{classifier}')
        model.load(model_dir)
        pred, pred_map = model.predict(X)
        np.savetxt(pred_path, pred)
        save_as_text(f"{pred_path}.map", pred_map.values())
    else:
        raise ValueError(f'Unknown command type {command}')

    print(f"All done")


if __name__ == "__main__":
    main()
