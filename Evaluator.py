import pandas as pd
from typing import List, Callable
from dg_models.DGModel import DGModel
from ml_models.MachineLearner import MachineLearner
from utils.Data import Data
import numpy as np


class Evaluator:
    def __init__(self, ml_models: List[MachineLearner], dg_models: List[DGModel], real_models: List[DGModel],
                 scores: List[Callable], outcome_name: str = "Y"):
        self.scores = scores
        self.real_models = real_models
        self.ml_models = ml_models
        self.dg_models = dg_models
        self.outcome_name = outcome_name

    def analysis_1_per_dg_model(self, dg_model_real: DGModel, n_samples: int, tr_frac: float, n_btstrps: int):
        """

        :param dg_model_real: a (pretended) real-world DagSim model
        :param n_samples: number of samples to generate for training + testing
        :param tr_frac: fraction of data for training
        :param n_btstrps: numer of times to  perform bootstrap
        :return: btstrp_results: dict of shape {model_name: {score_name: list_of_score_values], ...}, ...}
        """
        orig_train_data, test_data = self._get_train_and_test_from_dg(dg_model_real, n_samples, tr_frac)

        assert len(orig_train_data) + len(test_data) == n_samples

        btstrp_results = pd.DataFrame(data=[[[] for _ in range(len(self.ml_models))] for __ in range(len(self.scores))],
                                      index=[sc.__name__ for sc in self.scores],
                                      columns=[md.name for md in self.ml_models])
        for _ in range(n_btstrps):
            btstr_tr = orig_train_data.bootstrap()
            for ml_model in self.ml_models:
                result_per_btsrtp_per_model = self._develop_ml_model(ml_model, btstr_tr, test_data)
                for score_name in btstrp_results.index.values.tolist():
                    btstrp_results[ml_model.name][score_name].append(result_per_btsrtp_per_model[score_name])
        return btstrp_results

    def analysis_2_per_dg_model(self, dg_model_real: DGModel, n_learning: int = 100):
        corr_dict = {}
        real_data, corr_dict[dg_model_real.name] = self._get_corr(dg_model_real)

        for dg_model in self.dg_models:
            dg_model.fit(real_data[0:n_learning])
            _, corr_dict[dg_model.name] = self._get_corr(dg_model)

        return corr_dict

    def analysis_3_per_dg_model(self, dg_model_real: DGModel, n_learning: int = 100, n_train: int = 100,
                                n_test: int = 100):
        metrics = {}
        dg_metrics, learning_data, test_data = self._evaluate_dg_model(dg_model_real, n_learning=n_learning,
                                                                       n_train=n_train, n_test=n_test)
        metrics.update(dg_metrics)
        for dg_model in self.dg_models:
            dg_model.fit(learning_data)
            # todo fix issue with PC
            # assert len(dg_model_real.model.nodes) == len(test_data.all.columns)
            # todo change for non pgmpy models
            if dg_model.num_vars != len(test_data.all.columns):
                continue
            dg_metrics, _, _ = self._evaluate_dg_model(dg_model=dg_model, n_learning=-1, n_train=n_train, n_test=n_test,
                                                       test_data=test_data)
            metrics.update(dg_metrics)
        return metrics

    def analysis_4_per_dg_model(self, dg_model_real: DGModel, n_samples: int, tr_frac: float, n_reps: int):
        scores_per_dg_model = {}
        dg_model_real_scores, train_data, test_data = self._get_performance_by_repetition(dg_model_real, n_samples,
                                                                                          tr_frac, n_reps)
        scores_per_dg_model[dg_model_real.name] = dg_model_real_scores
        for dg_model in self.dg_models:
            dg_model.fit(train_data)
            if dg_model.num_vars != len(test_data.all.columns):
                continue
            dg_model_scores, *_ = self._get_performance_by_repetition(dg_model, n_samples, tr_frac, n_reps,
                                                                      test_data=test_data)
            scores_per_dg_model[dg_model.name] = dg_model_scores
        return scores_per_dg_model

    def _evaluate_ml_model(self, ml_model: MachineLearner, test_data: Data):
        '''
        Given a **trained** ml model, evaluate its performance using all the defined metrics on the test set.

        '''
        y_pred = ml_model.predict(test_data)
        metrics = {}
        for score in self.scores:
            metrics.update({score.__name__: score(y_pred=y_pred, y_true=test_data.y)})
        return metrics

    def _develop_all_ml_models(self, train_data: Data, test_data: Data):  # level 3 repetition
        '''
        Given a training set and a test set, train each machine learning model on the training data and test it on the
        test set.
        '''
        metrics = {}
        for ml_model in self.ml_models:
            metrics[f'{ml_model.name}'] = self._develop_ml_model(ml_model=ml_model, train_data=train_data,
                                                                 test_data=test_data)
        return metrics

    def _evaluate_dg_model(self, dg_model: DGModel, n_learning: int, n_train: int, n_test: int,
                           test_data: Data = None):  # level 2 repetition
        train_data = dg_model.generate(n_train, self.outcome_name)

        if test_data is None:
            test_data = dg_model.generate(n_test, self.outcome_name)
        metrics = self._develop_all_ml_models(train_data, test_data)
        metrics = {f'{dg_model.name}': metrics}
        learning_data = train_data[0:n_learning:1] if n_learning > 0 else None
        return metrics, learning_data, test_data

    def _get_performance_by_repetition(self, dg_model: DGModel, n_samples_real: int, tr_frac: float, n_reps: int,
                                       test_data=None):
        all_scores = {ml_model.name: {score.__name__: [] for score in self.scores} for ml_model in self.ml_models}
        train_data = None

        for rep in range(n_reps):
            train_data = dg_model.generate(int(n_samples_real * tr_frac), self.outcome_name)
            if test_data is None:
                test_data = dg_model.generate(int(n_samples_real * (1 - tr_frac)), self.outcome_name)
            for ml_model in self.ml_models:
                scores = self._develop_ml_model(ml_model, train_data, test_data)
                for score_name in scores.keys():
                    all_scores[ml_model.name][score_name].append(scores[score_name])
        return all_scores, train_data, test_data

    def _develop_ml_model(self, ml_model: MachineLearner, train_data: Data, test_data: Data):
        '''
        For a given ml model, train it on the provided training set and evaluate it on the test set.

        '''
        ml_model.learn(train_data)
        scores = self._evaluate_ml_model(ml_model, test_data)
        return scores

    def _get_corr(self, dg_model):
        data = dg_model.generate(num_samples=100000, outcome_name=self.outcome_name)
        up_tri_mat = np.triu(np.ones(data.all.corr().shape), k=1).astype(bool)
        corr_df = data.all.corr().where(up_tri_mat)
        corr_df = corr_df.stack().reset_index()
        return data, corr_df

    def _get_train_and_test_from_dg(self, dg_model_real, n_samples, tr_frac):
        train_data = dg_model_real.generate(int(n_samples * tr_frac), self.outcome_name)
        test_data = dg_model_real.generate(int(n_samples * (1 - tr_frac)), self.outcome_name)
        return train_data, test_data
