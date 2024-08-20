from unittest import TestCase
from unittest.mock import patch

import pyarrow as pa
import pytest
from datasets import Dataset, DatasetDict

from pipeline.datasets import load_from_huggingface
from pipeline.exceptions import DatasetNotFoundError, InvalidDatasetError


class TestDataset(TestCase):
    @pytest.mark.integration_test
    def test_load_from_huggingface_when_invalid_name_then_raises_err(self):
        with self.assertRaises(DatasetNotFoundError):
            load_from_huggingface("bad-dataset-name-9rmvoitkj")

    @patch("pipeline.datasets.load_dataset")
    def test_load_from_hugginface_when_return_dataset_wo_train_then_raises_err(
        self, mock_load_dataset
    ):
        mock_load_dataset.return_value = DatasetDict(
            {
                "test": Dataset(
                    pa.Table.from_arrays(
                        [["blah", "blai", "blaj"], [0, 1, 0]],
                        names=["text", "label"],
                    )
                )
            }
        )

        with self.assertRaises(InvalidDatasetError):
            load_from_huggingface("bad-dataset-name-290bajatijg")

    @patch("pipeline.datasets.load_dataset")
    def test_load_from_hugginface_when_return_dataset_wo_test_then_raises_err(
        self, mock_load_dataset
    ):
        mock_load_dataset.return_value = DatasetDict(
            {
                "train": Dataset(
                    pa.Table.from_arrays(
                        [["blah", "blai", "blaj"], [0, 1, 0]],
                        names=["text", "label"],
                    )
                )
            }
        )

        with self.assertRaises(InvalidDatasetError):
            load_from_huggingface("bad-dataset-name-290bajatijg")
