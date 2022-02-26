import random
import numpy as np
from baseDS import Graph, Generic

# Ground truth definition using a logit (sigmoid) function
#Linear problem
def log_transformation(params0, params1, params2, params3):
    sum = params0 * 2 + params1 - params2 + params3 + random.randint(0,1)
    y = 1 / (1 + np.exp(-sum))
    y = 1 if y > 0.5 else 0
    return y

#Non-linear problem
def log_transformation_nonlinear(params0, params1, params2, params3):
    sum = params0 * pow(2, 4) + params1 + params2 + params3 + random.randint(0,1)
    y = 1 / (1 + np.exp(-sum))
    y = 1 if y > 0.5 else 0
    return y

#Non-linear variant
#def log_transformation_sparse(params0, params1, params2, params3):
#    sum = params0 * pow(2, 4) + params1 + params2 + params3 + random.randint(0,1)
#    y = 1 / (1 + np.exp(-sum))
#    y = 1 if y > 0.5 else 0
#    return y


#2-Dimnesion variant
def log_transformation_dimensionality(params0, params1, params2, params3, params4, params5, params6, params7, params8, params9):
    sum = params0 * 2 + params1 + params2 - 2 + params3 + params4 - 2 + params5 + params6 + 2 + params7 + params8 - 2 + params9 + random.randint(0,1)
    y = 1 / (1 + np.exp(-sum))
    y = 1 if y > 0.5 else 0
    return y

def from_A_get_B(A):
    if A:
        return np.random.binomial(n=1, p=0.4)
    else:
        return np.random.binomial(n=1, p=0.7)

# DAG setup
def setup_realworld(pipeline_type, training_n, test_n):
    if(pipeline_type==1):
        Prior1 = Generic(name="A", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior2 = Generic(name="B", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior3 = Generic(name="C", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior4 = Generic(name="D", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Node1 = Generic(name="E", function=log_transformation,
                        arguments={"params0": Prior1, "params1": Prior2, "params2": Prior3, "params3": Prior4})

        listNodes = [Prior1, Prior2, Prior3, Prior4, Node1]
        my_graph = Graph("Logistic Regression - Real-world", listNodes)
    elif(pipeline_type==2):
        Prior1 = Generic(name="A", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior2 = Generic(name="B", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior3 = Generic(name="C", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior4 = Generic(name="D", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Node1 = Generic(name="E", function=log_transformation_nonlinear,
                        arguments={"params0": Prior1, "params1": Prior2, "params2": Prior3, "params3": Prior4})

        listNodes = [Prior1, Prior2, Prior3, Prior4, Node1]
        my_graph = Graph("Logistic Regression - Real-world", listNodes)
    elif(pipeline_type==3):
        Prior1 = Generic(name="A", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior2 = Generic(name="B", function=from_A_get_B, arguments={"A": Prior1})
        Prior3 = Generic(name="C", function=from_A_get_B, arguments={"A": Prior1})
        Prior4 = Generic(name="D", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Node1 = Generic(name="E", function=log_transformation,
                        arguments={"params0": Prior1, "params1": Prior2, "params2": Prior3, "params3": Prior4})
        listNodes = [Prior1, Prior2, Prior3, Prior4, Node1]
        my_graph = Graph("Logistic Regression - Real-world", listNodes)
    elif(pipeline_type==4):
        Prior1 = Generic(name="A", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior2 = Generic(name="B", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior3 = Generic(name="C", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior4 = Generic(name="D", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior5 = Generic(name="E", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior6 = Generic(name="F", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior7 = Generic(name="G", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior8 = Generic(name="H", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior9 = Generic(name="I", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Prior10 = Generic(name="J", function=np.random.binomial, arguments={"n": 1, "p": 0.5})
        Node1 = Generic(name="K", function=log_transformation_dimensionality,
                        arguments={"params0": Prior1, "params1": Prior2, "params2": Prior3, "params3": Prior4, "params4": Prior5, "params5": Prior6, "params6": Prior7, "params7": Prior8, "params8": Prior9, "params9": Prior10})
        listNodes = [Prior1, Prior2, Prior3, Prior4,Prior5, Prior6,Prior7, Prior8,Prior9, Prior10,Node1]
        my_graph = Graph("Logistic Regression - Real-world", listNodes)
    train = my_graph.simulate(training_n, csv_name="train")
    test = my_graph.simulate(test_n, csv_name="test")
    return train