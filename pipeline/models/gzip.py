import gzip

from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from ..exceptions import ModelNotTrainedError
from .base import BaseModel


def _gzip_distance(a: str, b: str):
    Ca = len(gzip.compress(a))
    Cb = len(gzip.compress(b))
    ab = b" ".join([a, b])
    Cab = len(gzip.compress(ab))
    ncd = (Cab - min(Ca, Cb)) / max(Ca, Cb)
    return ncd


class GzipClassifier(BaseModel):
    def __init__(self, k):
        self.k = k
        self.knn = KNeighborsClassifier(
            n_neighbors=self.k, metric="precomputed"
        )

    def train(self, X, y):
        self.X_train = self._encode_input(X)
        affinity_matrix = squareform(pdist(self.X_train, _gzip_distance))
        self.knn.fit(affinity_matrix, y)

    def predict(self, X):
        try:
            check_is_fitted(self.knn)
        except NotFittedError:
            raise ModelNotTrainedError
        X_encoded = self._encode_input(X)
        return self.knn.predict(cdist(X_encoded, self.X_train, _gzip_distance))

    def _encode_input(self, X):
        return [[val.encode()] for val in X]
