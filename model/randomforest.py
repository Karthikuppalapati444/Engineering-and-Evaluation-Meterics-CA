import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class RandomForest(BaseModel):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(
            n_estimators=1000,
            random_state=0,
            class_weight='balanced_subsample'
        )
        self.predictions = None

    def train(self, X_train, y_train) -> None:
        self.mdl.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray):
        self.predictions = self.mdl.predict(X_test)
        return self.predictions

    def print_results(self, y_test):
        if self.predictions is None:
            print("[⚠️] No predictions were made yet. Call predict() first.")
        else:
            print(classification_report(y_test, self.predictions))

    def data_transform(self):
        # Required by BaseModel abstract class
        pass
