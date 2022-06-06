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
from sklearn.metrics import accuracy_score


warnings.simplefilter(action='ignore', category=FutureWarning)

list_pgmpy = [PgmpyModel(f'{learner.__name__}', learner, "Y") for learner in
              [PC, HillClimbSearch, TreeSearch]]  # , ExhaustiveSearch]]

list_sklearn = [SklearnModel(f'{learner.__name__}', learner) for learner in
                [LogisticRegression, BernoulliNB, KNeighborsClassifier]]

x = Node("X", function=np.random.binomial, args=[1, 0.5])
z = Node("Z", function=np.random.binomial, args=[1, 0.5])
y = Node("Y", function=np.random.binomial, args=[1, 0.5])


def log_transformation(params0, params1, params2, params3):
    sum = params0 * 2 + params1 - params2 + params3 + random.randint(0, 1)
    y = 1 / (1 + np.exp(-sum))
    # print("-> ", y)
    y = 1 if y > 0.75 else 0
    return y


Prior1 = Node(name="A", function=np.random.binomial, kwargs={"n": 1, "p": 0.5})
Prior2 = Node(name="B", function=np.random.binomial, kwargs={"n": 1, "p": 0.2})
Prior3 = Node(name="C", function=np.random.binomial, kwargs={"n": 1, "p": 0.7})
Prior4 = Node(name="D", function=np.random.binomial, kwargs={"n": 1, "p": 0.5})
Node1 = Node(name="Y", function=log_transformation,
             kwargs={"params0": Prior1, "params1": Prior2, "params2": Prior3, "params3": Prior4})

listNodes = [Prior1, Prior2, Prior3, Prior4, Node1]
my_graph = Graph(name="Logistic Regression - Real-world", list_nodes=listNodes)

ds_model = DagsimModel("pipeline1", my_graph)

evaluator = Evaluator(ml_models=list_sklearn, dg_models=list_pgmpy, real_models=[ds_model], scores=[accuracy_score],
                      n_learning=100, n_train=1000, n_test=200, outcome_name="Y")

results = evaluator.run_all()
print(results)

pp = Postprocessing(results)
# df = pp.dict_to_dataframe()
# ll = pp.dict_to_list("pipeline1")
pp.plot_scores("pipeline1")
