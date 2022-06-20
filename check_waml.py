import random
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from pgmpy.estimators import *
import warnings
from dg_models.PgmpyLearner import PgmpyModel
from dg_models.DagsimModel import DagsimModel
from ml_models.SklearnModel import SklearnModel
import numpy as np
from dagsim.base import Graph, Node
from Evaluator import Evaluator
from utils.Postprocressing import Postprocessing
from sklearn.metrics import accuracy_score, balanced_accuracy_score, auc, average_precision_score

warnings.simplefilter(action='ignore', category=FutureWarning)

list_pgmpy = [PgmpyModel(f'{learner.__name__}', learner, "Y") for learner in
              [PC, HillClimbSearch, TreeSearch]]  # , ExhaustiveSearch]]

list_sklearn = [SklearnModel(f'{learner.__name__}', learner) for learner in
                [LogisticRegression, BernoulliNB, KNeighborsClassifier]]


def log_transformation(params0, params1, params2, params3):
    sum = params0 * 2 + params1 - params2 + params3 + random.randint(0, 1)
    y = 1 / (1 + np.exp(-sum))
    y = 1 if y > 0.75 else 0
    return y


Prior1 = Node(name="A", function=np.random.binomial, kwargs={"n": 1, "p": 0.5}, size_field="size")
Prior2 = Node(name="B", function=np.random.binomial, kwargs={"n": 1, "p": 0.2}, size_field="size")
Prior3 = Node(name="C", function=np.random.binomial, kwargs={"n": 1, "p": 0.7}, size_field="size")
Prior4 = Node(name="D", function=np.random.binomial, kwargs={"n": 1, "p": 0.5}, size_field="size")
Node1 = Node(name="Y", function=log_transformation,
             kwargs={"params0": Prior1, "params1": Prior2, "params2": Prior3, "params3": Prior4})

listNodes = [Prior1, Prior2, Prior3, Prior4, Node1]
my_graph = Graph(name="Logistic Regression - Real-world", list_nodes=listNodes)

ds_model = DagsimModel("pipeline1", my_graph)

evaluator = Evaluator(ml_models=list_sklearn, dg_models=list_pgmpy, real_models=[ds_model],
                      scores=[accuracy_score, balanced_accuracy_score], n_learning=100, n_train=1000, n_test=100,
                      outcome_name="Y")

pp = Postprocessing()

# analysis1_results = evaluator.analysis_1_per_dg_model(dg_model_real=ds_model, n_samples=200, tr_frac=0.5, n_btstrps=20)
# pp.plot_analysis1(analysis1_results)

analysis2 = evaluator.analysis_2_per_dg_model(ds_model)
# pp.plot_analysis2(analysis2)
pp.plot_analysis2_gks(analysis2)
#
# analysis3 = evaluator.analysis_2_per_dg_model(ds_model)
# pp.plot_analysis3(pipeline="pipeline1", analysis3_results=analysis3)
#
# analysis4 = evaluator.analysis_4_per_dg_model(ds_model, 200, 0.5, 5)
# pp.plot_analysis4(analysis4)
