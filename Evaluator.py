import pandas as pd
from typing import List, Callable
from dg_models.DGModel import DGModel
from ml_models.MachineLearner import MachineLearner
from utils.Data import Data
from sklearn.metrics import accuracy_score, auc, average_precision_score, balanced_accuracy_score


class Evaluator():
    def __init__(self, ml_models: List[MachineLearner], dg_models: List[DGModel], real_models: List[DGModel],
                 scores: List[Callable], n_learning: int, n_train: int, n_test: int, outcome_name: str = "Y"):
        self.scores = scores
        self.real_models = real_models
        self.ml_models = ml_models
        self.dg_models = dg_models
        self.n_learning = n_learning
        self.n_train = n_train
        self.n_test = n_test
        self.outcome_name = outcome_name

    def generate_from_world(self, dg_model: DGModel, num_samples: int):
        return dg_model.generate(num_samples, self.outcome_name)

    def evaluate_ml_model(self, ml_model: MachineLearner, test_data: Data):
        y_pred = ml_model.predict(test_data)
        metrics = {}
        for score in self.scores:
            metrics.update({"accuracy": score(y_pred=y_pred, y_true=test_data.y)})
        return metrics

    def evaluate_ml_models(self, train_data: Data, test_data: Data):  # level 3 repetition
        metrics = {}
        for model in self.ml_models:
            model.learn(train_data)
            metrics[f'{model.name}'] = self.evaluate_ml_model(model, test_data)
        return metrics

    def evaluate_dg_model(self, dg_model: DGModel, n_learning: int, test_data: Data = None):  # level 2 repetition
        train_data = dg_model.generate(self.n_train, self.outcome_name)
        # if dg_model.name == "real_model":
        #     print(train_data.y)
        #     print((sum(train_data.y)/len(train_data.y))*100, "% out of ", len(train_data.y))
        #     exit(33)

        if test_data is None:
            test_data = dg_model.generate(self.n_test, self.outcome_name)
        metrics = self.evaluate_ml_models(train_data, test_data)
        metrics = {f'{dg_model.name}': metrics}
        learning_data = train_data[0:n_learning:1] if n_learning > 0 else None
        return metrics, learning_data, test_data

    def run_pipeline(self, real_model: DGModel):
        metrics = {}
        dg_metrics, learning_data, test_data = self.evaluate_dg_model(real_model, self.n_learning)
        dg_metrics["real"] = dg_metrics.pop(real_model.name)
        metrics.update(dg_metrics)
        for dg_model in self.dg_models:
            dg_model.fit(learning_data)
            # todo fix issue with PC
            # assert len(dg_model.model.nodes) == len(test_data.all.columns)
            if len(dg_model.model.nodes) != len(test_data.all.columns):
                continue
            dg_metrics, _, _ = self.evaluate_dg_model(dg_model, -1, test_data)
            metrics.update(dg_metrics)
        return metrics

    def run_all(self):  # level 1 repetition
        metrics = {}
        for real_model in self.real_models:
            metrics[real_model.name] = self.run_pipeline(real_model)
        return metrics
