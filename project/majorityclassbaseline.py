# TODO: build the majority class baseline model.
# TODO: find the majority class in the labels. 🤔
# TODO: score the model on valdf with a 2D metric space: sklearn.metrics.accuracy_score, sklearn.metrics.roc_auc_score
# Documentation on suggested model-scoring approach: https://scikit-learn.org/stable/modules/model_evaluation.html

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, roc_auc_score

class MajorityClassBaseline(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.majority_class = y.value_counts().idxmax()

    def predict(self, X):
        return [self.majority_class] * len(X)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        roc_auc = roc_auc_score(y, predictions)
        return accuracy, roc_auc

