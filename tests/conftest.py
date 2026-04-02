from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from job_offers_classifier.job_offers_utils import create_hierarchy
from job_offers_classifier.load_save import load_texts, load_to_df


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_hierarchy(test_data_dir):
    return create_hierarchy(load_to_df(str(test_data_dir / "classes.tsv")))


@pytest.fixture
def sample_texts(test_data_dir):
    return {
        "x_train": load_texts(str(test_data_dir / "x_train.txt")),
        "y_train": load_texts(str(test_data_dir / "y_train.txt")),
        "x_test": load_texts(str(test_data_dir / "x_test.txt")),
    }
