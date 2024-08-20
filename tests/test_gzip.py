from unittest import TestCase

from pipeline.exceptions import ModelNotTrainedError
from pipeline.models.gzip import GzipClassifier


class TestGzip(TestCase):
    def test_when_predict_with_no_training_then_error(self):
        model = GzipClassifier(5)
        X = None
        with self.assertRaises(ModelNotTrainedError):
            model.predict(X)
