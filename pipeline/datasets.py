from enum import StrEnum

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError as HFDatasetNotFoundError

from .exceptions import DatasetNotFoundError, InvalidDatasetError


class HuggingfaceDataset(StrEnum):
    AG_NEWS = "fancyzhx/ag_news"
    R8 = "dxgp/R8"


def load_from_huggingface(dataset_path: HuggingfaceDataset | str):
    if isinstance(dataset_path, HuggingfaceDataset):
        dataset_path = dataset_path.value
    try:
        ds = load_dataset(dataset_path)
    except HFDatasetNotFoundError:
        raise DatasetNotFoundError(f"{dataset_path} not found in huggingface")
    if "test" not in ds or "train" not in ds:
        raise InvalidDatasetError(
            f"We need both train and test but you have {', '.join(ds.keys())}"
        )

    return ds
