from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, balanced_accuracy_score


class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def serialize(self):
        pass

    @classmethod
    def deserialize(cls, bytes):
        pass

    def evaluate(self, X, y):
        y_pred = self.predict(X)

        return {"accuracy": accuracy_score(y, y_pred), "balanced_accuracy": balanced_accuracy_score(y, y_pred)}
