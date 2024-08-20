import gzip

from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from ..exceptions import ModelNotTrainedError
from .base import BaseModel


def _gzip_distance(a: str, b: str):
    Ca = len(gzip.compress(a.encode()))
    Cb = len(gzip.compress(b.encode()))
    ab = " ".join([a, b])
    Cab = len(gzip.compress(ab.encode()))
    ncd = (Cab - min(Ca, Cb)) / max(Ca, Cb)
    return ncd


class GzipClassifier(BaseModel):
    def __init__(self, k):
        self.k = k
        self.knn = KNeighborsClassifier(n_neighbors=self.k, metric=_gzip_distance)

    def train(self, X, y):
        self.knn.fit(X, y)

    def predict(self, X):
        try:
            check_is_fitted(self.knn)
        except NotFittedError:
            raise ModelNotTrainedError
        return self.knn.predict(X)

    def serialize(self):
        pass
