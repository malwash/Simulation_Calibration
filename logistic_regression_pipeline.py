import random
import importlib

from pgmpy.estimators import BayesianEstimator, PC, HillClimbSearch, BicScore, TreeSearch, MmhcEstimator
from pgmpy.models import BayesianModel
from pomegranate import BayesianNetwork
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import simulation_notears
import simulation_dagsim
import simulation_models
from sklearn import metrics
from sklearn import svm
from notears import utils
from notears.linear import notears_linear
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import DataFrame, StrVector
from rpy2.robjects.packages import importr

TRAIN_SIZE = 1000
TEST_SIZE = 1000

def slice_data(pipeline_type, train_data, test_data):
    '''
    splits and reshapes simulated data into appropriate size
    :param pipeline_type: indicates the problem type being executed (pipeline 1-3, 4 variables, 1 outcome, pipelines 4, 10 variables, 1 outcome)
    :param train_data: the training set
    :param test_data: the test set
    :return:
    '''
    #print(train_data.shape)
    #print(test_data.shape)
    # todo get shape automatically
    if(pipeline_type==4):
        x_train = train_data.iloc[:, 0:10].to_numpy().reshape([-1, 10])  # num predictors
        y_train = train_data.iloc[:, 10].to_numpy().reshape([-1])  # outcome
        x_test = test_data.iloc[:, 0:10].to_numpy().reshape([-1, 10])  # num predictors
        y_test = test_data.iloc[:, 10].to_numpy().reshape([-1])  # outcome
    else:
        x_train = train_data.iloc[:, 0:4].to_numpy().reshape([-1, 4])  # num predictors
        y_train = train_data.iloc[:, 4].to_numpy().reshape([-1])  # outcome
        x_test = test_data.iloc[:, 0:4].to_numpy().reshape([-1, 4])  # num predictors
        y_test = test_data.iloc[:, 4].to_numpy().reshape([-1])  # outcome
    return x_train, y_train, x_test, y_test


def get_data_from_real_world(pipeline_type, num_train, num_test, world=None, data=None):
    '''
    Simulate data using dagsim based on the pipeline type
    :param pipeline_type: indicates the problem type being executed (pipeline 1-3, 4 variables, 1 outcome, pipelines 4, 10 variables, 1 outcome)
    :param num_train the number of samples for training
    :param num_test the number of samples for testing
    :return: sliced train and test data
    '''
    # if data is None:
    #     if world!="real":
    #             raise("For learning a world, data are needed")
    #     else:
    #         train, test = simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
    # else:
    #     if world=="real":
    #         raise("world==real with data are not allowed")
    #     elif world=="notears":
    #         train, test = simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
    #     elif world=="bnlearn":
    #         train, test = simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
    train, test = simulation_dagsim.setup_realworld(pipeline_type, num_train, num_test)
    x_train, y_train, x_test, y_test = slice_data(pipeline_type, train, test)
    return x_train, y_train, x_test, y_test


def get_data_from_learned_world(pkg_name, config, real_data, num_train, num_test, pipeline_type):
    '''
    :param pkg_name: Bayesian Structural Learning framework used to sample data
    :param config: the parameterisation at the simulation-level
    :param real_data: subset of the real data to learn from and perform parameter learning
    :param num_train: the number of samples for training
    :param num_test: the number of samples for testing
    :param pipeline_type: indicates the problem type being executed (pipeline 1-3, 4 variables, 1 outcome, pipelines 4, 10 variables, 1 outcome)
    :return: sliced train and test data from the simulated framework
    '''
    # module = importlib.import_module(pkg_name)
    # function = getattr(module, f'{pkg_name}_setup_{config}')
    learned_data_train = None
    learned_data_test = None
    if pkg_name=="notears":
        model = notears_linear(real_data, lambda1=0.01, loss_type=config)
        learned_data_train = utils.simulate_linear_sem(model, num_train, 'logistic')
        learned_data_test = utils.simulate_linear_sem(model, num_test, 'logistic')
    elif pkg_name=="pgmpy":
        model_learn = None
        model = None
        real_data = pd.DataFrame(real_data)
        #if config=="pc":
        #    model_learn = PC(real_data)
        #    model = model_learn.estimate()
        if config=="hc":
            model_learn = HillClimbSearch(real_data)
            model = model_learn.estimate(scoring_method=BicScore(real_data))
        elif config=="tree":
            model_learn = TreeSearch(real_data)
            model = model_learn.estimate(estimator_type='chow-liu')
        elif config=="mmhc":
            model_learn = MmhcEstimator(real_data)
            model = model_learn.estimate()
        construct = BayesianModel(model)
        estimator = BayesianEstimator(construct, real_data)
        cpds = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=1000)
        for cpd in cpds:
            construct.add_cpds(cpd)
        construct.check_model()
        learned_data_train = construct.simulate(n_samples=int(1000))
        learned_data_test = construct.simulate(n_samples=int(1000))
    elif pkg_name=="pomegranate":
        real_data = pd.DataFrame(real_data)
        #print(real_data)
        model = BayesianNetwork.from_samples(real_data, algorithm=config)
        learned_data_train = model.sample(1000)
        learned_data_test = model.sample(1000)
    elif pkg_name=="bnlearn": #bnlearn R emulation
        if config=="hc":
            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                robjects.r('''
                        library(bnlearn)
                        bn_hillclimbing <- function(r, verbose=FALSE) {
                        databn <-read.csv("train.csv", header=FALSE)
                        databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
                        my_bn <- hc(databn)
                        fit = bn.fit(my_bn, databn)
                        training_output = rbn(my_bn, 1000, databn)
                        }
                        bn_hillclimbing()
                        ''')
            elif (pipeline_type == 4):
                robjects.r('''
                        library(bnlearn)
                        bn_hillclimbing <- function(r, verbose=FALSE) {
                        databn <-read.csv("train.csv", header=FALSE)
                        databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
                        my_bn <- hc(databn)
                        fit = bn.fit(my_bn, databn)
                        training_output = rbn(my_bn, 1000, databn)
                        }
                        bn_hillclimbing()
                        ''')
            bn_hc = robjects.r['bn_hillclimbing']
            bn_train_output = bn_hc()

            result = np.array(bn_train_output)
            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                result_edit = result[:, :].reshape([-1, 5])
            elif (pipeline_type == 4):
                result_edit = result[:, :].reshape([-1, 11])
            learned_data_train = pd.DataFrame(result_edit)

            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                robjects.r('''
                            library(bnlearn)
                            bn_hillclimbing <- function(r, verbose=FALSE) {
                            databn <-read.csv("train.csv", header=FALSE)
                            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
                            my_bn <- hc(databn)
                            fit = bn.fit(my_bn, databn)
                            training_output = rbn(my_bn, 1000, databn)
                            }
                            bn_hillclimbing()
                            ''')
            elif (pipeline_type == 4):
                robjects.r('''
                            library(bnlearn)
                            bn_hillclimbing <- function(r, verbose=FALSE) {
                            databn <-read.csv("train.csv", header=FALSE)
                            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
                            my_bn <- hc(databn)
                            fit = bn.fit(my_bn, databn)
                            training_output = rbn(my_bn, 1000, databn)
                            }
                            bn_hillclimbing()
                            ''')
            bn_hc = robjects.r['bn_hillclimbing']
            bn_test_output = bn_hc()

            result = np.array(bn_test_output)
            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                result_edit_test = result[:, :].reshape([-1, 5])
            elif (pipeline_type == 4):
                result_edit_test = result[:, :].reshape([-1, 11])
            learned_data_test = pd.DataFrame(result_edit_test)
        elif config=="tabu":
            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                robjects.r('''
                        library(bnlearn)
                        bn_tabu <- function(r, verbose=FALSE) {
                        databn <-read.csv("train.csv", header=FALSE)
                        databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
                        my_bn <- tabu(databn)
                        fit = bn.fit(my_bn, databn)
                        training_output = rbn(my_bn, 1000, databn)
                        }
                        bn_tabu()
                        ''')
            elif (pipeline_type == 4):
                robjects.r('''
                        library(bnlearn)
                        bn_tabu <- function(r, verbose=FALSE) {
                        databn <-read.csv("train.csv", header=FALSE)
                        databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
                        my_bn <- tabu(databn)
                        fit = bn.fit(my_bn, databn)
                        training_output = rbn(my_bn, 1000, databn)
                        }
                        bn_tabu()
                        ''')
            bn_tabu = robjects.r['bn_tabu']
            bn_train_output = bn_tabu()

            result = np.array(bn_train_output)
            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                result_edit = result[:, :].reshape([-1, 5])
            elif (pipeline_type == 4):
                result_edit = result[:, :].reshape([-1, 11])
            learned_data_train = pd.DataFrame(result_edit)

            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                robjects.r('''
                        library(bnlearn)
                        bn_tabu <- function(r, verbose=FALSE) {
                        databn <-read.csv("train.csv", header=FALSE)
                        databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
                        my_bn <- tabu(databn)
                        fit = bn.fit(my_bn, databn)
                        training_output = rbn(my_bn, 1000, databn)
                        }
                        bn_tabu()
                        ''')
            elif (pipeline_type == 4):
                robjects.r('''
                        library(bnlearn)
                        bn_tabu <- function(r, verbose=FALSE) {
                        databn <-read.csv("train.csv", header=FALSE)
                        databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
                        my_bn <- tabu(databn)
                        fit = bn.fit(my_bn, databn)
                        training_output = rbn(my_bn, 1000, databn)
                        }
                        bn_tabu()
                        ''')
            bn_tabu = robjects.r['bn_tabu']
            bn_test_output = bn_tabu()

            result = np.array(bn_test_output)
            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                result_edit_test = result[:, :].reshape([-1, 5])
            elif (pipeline_type == 4):
                result_edit_test = result[:, :].reshape([-1, 11])
            learned_data_test = pd.DataFrame(result_edit_test)
        elif config=="mmhc":
            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                robjects.r('''
                        library(bnlearn)
                        bn_mmhc <- function(r, verbose=FALSE) {
                        databn <-read.csv("train.csv", header=FALSE)
                        databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
                        my_bn <- mmhc(databn)
                        fit = bn.fit(my_bn, databn)
                        training_output = rbn(my_bn, 1000, databn)
                        }
                        bn_mmhc()
                        ''')
            elif (pipeline_type == 4):
                robjects.r('''
                        library(bnlearn)
                        bn_mmhc <- function(r, verbose=FALSE) {
                        databn <-read.csv("train.csv", header=FALSE)
                        databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
                        my_bn <- mmhc(databn)
                        fit = bn.fit(my_bn, databn)
                        training_output = rbn(my_bn, 1000, databn)
                        }
                        bn_mmhc()
                        ''')
            bn_mmhc = robjects.r['bn_mmhc']
            bn_train_output = bn_mmhc()

            result = np.array(bn_train_output)
            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                result_edit = result[:, :].reshape([-1, 5])
            elif (pipeline_type == 4):
                result_edit = result[:, :].reshape([-1, 11])
            learned_data_train = pd.DataFrame(result_edit)

            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                robjects.r('''
                        library(bnlearn)
                        bn_mmhc <- function(r, verbose=FALSE) {
                        databn <-read.csv("train.csv", header=FALSE)
                        databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
                        my_bn <- mmhc(databn)
                        fit = bn.fit(my_bn, databn)
                        training_output = rbn(my_bn, 1000, databn)
                        }
                        bn_mmhc()
                        ''')
            elif (pipeline_type == 4):
                robjects.r('''
                        library(bnlearn)
                        bn_mmhc <- function(r, verbose=FALSE) {
                        databn <-read.csv("train.csv", header=FALSE)
                        databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
                        my_bn <- mmhc(databn)
                        fit = bn.fit(my_bn, databn)
                        training_output = rbn(my_bn, 1000, databn)
                        }
                        bn_mmhc()
                        ''')
            bn_mmhc = robjects.r['bn_mmhc']
            bn_test_output = bn_mmhc()

            result = np.array(bn_test_output)
            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                result_edit_test = result[:, :].reshape([-1, 5])
            elif (pipeline_type == 4):
                result_edit_test = result[:, :].reshape([-1, 11])
            learned_data_test = pd.DataFrame(result_edit_test)
        elif config=="rsmax2":
            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                robjects.r('''
                        library(bnlearn)
                        bn_rsmax2 <- function(r, verbose=FALSE) {
                        databn <-read.csv("train.csv", header=FALSE)
                        databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
                        my_bn <- rsmax2(databn)
                        fit = bn.fit(my_bn, databn)
                        training_output = rbn(my_bn, 1000, databn)
                        }
                        bn_rsmax2()
                        ''')
            elif (pipeline_type == 4):
                robjects.r('''
                        library(bnlearn)
                        bn_rsmax2 <- function(r, verbose=FALSE) {
                        databn <-read.csv("train.csv", header=FALSE)
                        databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
                        my_bn <- rsmax2(databn)
                        fit = bn.fit(my_bn, databn)
                        training_output = rbn(my_bn, 1000, databn)
                        }
                        bn_rsmax2()
                        ''')
            bn_rsmax2 = robjects.r['bn_rsmax2']
            bn_train_output = bn_rsmax2()

            result = np.array(bn_train_output)
            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                result_edit = result[:, :].reshape([-1, 5])
            elif (pipeline_type == 4):
                result_edit = result[:, :].reshape([-1, 11])
            learned_data_train = pd.DataFrame(result_edit)

            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                robjects.r('''
                        library(bnlearn)
                        bn_rsmax2 <- function(r, verbose=FALSE) {
                        databn <-read.csv("train.csv", header=FALSE)
                        databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
                        my_bn <- rsmax2(databn)
                        fit = bn.fit(my_bn, databn)
                        training_output = rbn(my_bn, 1000, databn)
                        }
                        bn_rsmax2()
                        ''')
            elif (pipeline_type == 4):
                robjects.r('''
                        library(bnlearn)
                        bn_rsmax2 <- function(r, verbose=FALSE) {
                        databn <-read.csv("train.csv", header=FALSE)
                        databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
                        my_bn <- rsmax2(databn)
                        fit = bn.fit(my_bn, databn)
                        training_output = rbn(my_bn, 1000, databn)
                        }
                        bn_rsmax2()
                        ''')
            bn_rsmax2 = robjects.r['bn_rsmax2']
            bn_test_output = bn_rsmax2()

            result = np.array(bn_test_output)
            if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
                result_edit_test = result[:, :].reshape([-1, 5])
            elif (pipeline_type == 4):
                result_edit_test = result[:, :].reshape([-1, 11])
            learned_data_test = pd.DataFrame(result_edit_test)
    learned_data_train = pd.DataFrame(learned_data_train)
    learned_data_test = pd.DataFrame(learned_data_test)
    x_train, y_train, x_test, y_test = slice_data(pipeline_type, learned_data_train, learned_data_test)
    return x_train, y_train, x_test, y_test


# Evaluate function for all ML algorithms
def world_evaluate(world, pipeline_type, x_train, y_train, x_test, y_test):
    '''
    Performs Machine Learning benchmarks on the data provided
    :param world: indicates the source of the data, i.e., real, notears, bnlearn
    :param pipeline_type: indicates the problem type being executed (pipeline 1-3, 4 variables, 1 outcome, pipelines 4, 10 variables, 1 outcome)
    :param x_train: x variables from the training set
    :param y_train: y outcome from the training set
    :param x_test: x variables from the test set
    :param y_test: y outcome from the test set
    :return: dict with accuracy scores of ML methods with the configuration specified as the key (i.e., world_pipeline_method)
    '''
    # todo what does y_train do
    scores = {}
    pipeline_name = ["linear", "non-linear", "sparse", "dimension"][pipeline_type-1]
    MLModels = {"DTCgini": DecisionTreeClassifier(criterion='gini'),
                "DTCent": DecisionTreeClassifier(criterion='entropy'),
                "RFCgini": RandomForestClassifier(criterion='gini'),
                "RFCent": RandomForestClassifier(criterion='entropy'),
                "LRnone": LogisticRegression(penalty='none'),
                "LRl1": LogisticRegression(penalty='l1', solver='liblinear'),
                "LRl2": LogisticRegression(penalty='l2'),
                "LRmix": LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5),
                "BNB": BernoulliNB(), "GNB": GaussianNB(), "MnNB": MultinomialNB(), "CNB": ComplementNB(),
                "SVMsig": svm.SVC(kernel="sigmoid"), "SVMpoly": svm.SVC(kernel="poly"), "SVMrbf": svm.SVC(kernel="rbf"),
                "KNCunif": KNeighborsClassifier(weights='uniform'), "KNCdist": KNeighborsClassifier(weights='distance')}
    x_train_rdy = x_train
    x_test_rdy = x_test
    for key in MLModels.keys():
        if key in ["MnNB", "CNB"]:
            min_max_scaler = MinMaxScaler()
            x_train_rdy = min_max_scaler.fit_transform(x_train)
            x_test_rdy = min_max_scaler.transform(x_test)
        clf = MLModels[key]
        clf = clf.fit(x_train_rdy, y_train)
        y_pred = clf.predict(x_test_rdy)
        scores[f'{world}_{pipeline_name}_{key}'] = metrics.accuracy_score(y_test,y_pred)
    return scores

def evaluate_on_learned_world(pipeline_type, x_train, y_train, x_test, y_test):
    '''
    Runs through all parameter sets in structural learners, and performs benchmarking on learned data
    :param pipeline_type:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    '''
    learners = ["notears","pgmpy","pomegranate", "bnlearn"]
    notears_loss = ["logistic", "l2", "poisson"]
    pgmpy_algorithms = ["hc","tree", "mmhc"]
    pomegranate_algorithms = ["exact", "greedy"]
    bnlearn_algorithms = ["hc", "tabu", "mmhc", "rsmax2"]
    results = {}
    for loss in notears_loss:
        print(loss)
        x_train_lr, y_train_lr, _, _ = get_data_from_learned_world("notears", loss, np.concatenate([x_train[:100], y_train.reshape([-1,1])[:100]], axis=1), TRAIN_SIZE, TEST_SIZE, pipeline_type)
        scores = world_evaluate("notears"+"-"+loss, pipeline_type, x_train_lr, y_train_lr, x_test, y_test)
        results.update(scores)
    for algorithm in pgmpy_algorithms:
        print(algorithm)
        x_train_lr, y_train_lr, _, _ = get_data_from_learned_world("pgmpy", algorithm, np.concatenate([x_train[:100], y_train.reshape([-1,1])[:100]], axis=1), TRAIN_SIZE, TEST_SIZE, pipeline_type)
        scores = world_evaluate("pgmpy"+"-"+algorithm, pipeline_type, x_train_lr, y_train_lr, x_test, y_test)
        results.update(scores)
    for algorithm in pomegranate_algorithms:
        print(algorithm)
        x_train_lr, y_train_lr, _, _ = get_data_from_learned_world("pomegranate", algorithm, np.concatenate([x_train[:100], y_train.reshape([-1,1])[:100]], axis=1), TRAIN_SIZE, TEST_SIZE, pipeline_type)
        scores = world_evaluate("pomegranate"+"-"+algorithm, pipeline_type, x_train_lr, y_train_lr, x_test, y_test)
        results.update(scores)
    for algorithm in bnlearn_algorithms:
        print(algorithm)
        x_train_lr, y_train_lr, _, _ = get_data_from_learned_world("bnlearn", algorithm, np.concatenate([x_train[:100], y_train.reshape([-1,1])[:100]], axis=1), TRAIN_SIZE, TEST_SIZE, pipeline_type)
        scores = world_evaluate("bnlearn"+"-"+algorithm, pipeline_type, x_train_lr, y_train_lr, x_test, y_test)
        results.update(scores)
    return results

def run_all():
    '''
    Executes all benchmarking for both the real and learned world's
    :return: results_real is a dict for the real world, results_learned is a dict for the learned world
    '''
    results_real = {}
    results_learned = {}
    for pipeline_type in range(1,5):
        print(f'pipeline type: {pipeline_type}')
        x_train, y_train, x_test, y_test = get_data_from_real_world(pipeline_type, TRAIN_SIZE, TEST_SIZE)
        scores_real = world_evaluate("real", pipeline_type, x_train, y_train, x_test, y_test)
        print("Finished benchmarking Real-world scores")
        results_real.update(scores_real)
        scores_learned = evaluate_on_learned_world(pipeline_type, x_train, y_train, x_test, y_test)
        print("Finished benchmarking Learned-world scores")
        results_learned.update(scores_learned)
    return results_real, results_learned

def write_results_to_csv():
    '''
    input benchmarks of SimCal experiments to csv format
    :return: saved csv's of benchmarks in the format (learner_problemtype_mlestimator)
    '''

    with open('simulation_experiments_summary.csv', 'w', newline='') as csvfile:
        fieldnames = ['ProblemType', 'Algorithm', 'Method', 'Accuracy']
        thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        thewriter.writeheader()
        for k, v in real_results.items():
            if k.split('_')[-3] == "real":
                thewriter.writerow({fieldnames[0]: k.split('_')[-2], fieldnames[1]: k.split('_')[-3], fieldnames[2]: k.split('_')[-1], fieldnames[3]: real_results[k]})
        for k, v in learned_results.items():
            if k.split('_')[-2] == "linear":
                thewriter.writerow({fieldnames[0]: k.split('_')[-2], fieldnames[1]: k.split('_')[-3],fieldnames[2]: k.split('_')[-1], fieldnames[3]: learned_results[k]})
            elif k.split('_')[-2] == "non-linear":
                thewriter.writerow({fieldnames[0]: k.split('_')[-2], fieldnames[1]: k.split('_')[-3], fieldnames[2]: k.split('_')[-1], fieldnames[3]: learned_results[k]})
            elif k.split('_')[-2] == "sparse":
                thewriter.writerow({fieldnames[0]: k.split('_')[-2], fieldnames[1]: k.split('_')[-3], fieldnames[2]: k.split('_')[-1], fieldnames[3]: learned_results[k]})
            elif k.split('_')[-2] == "dimension":
                thewriter.writerow({fieldnames[0]: k.split('_')[-2], fieldnames[1]: k.split('_')[-3], fieldnames[2]: k.split('_')[-1], fieldnames[3]: learned_results[k]})

global real_results, learned_results #move these to a main call section
real_results, learned_results = run_all()
print(real_results)
print(learned_results)
write_results_to_csv()

#refactor into list of calibrations and then iterate through list and designate benchmarks
def write_results_to_figures():
    '''
    input benchmarks of SimCal experiments to output figures for predictive performance and structural performance
    :return: saved png's of benchmarks grouped by problem type
    '''
    # Group by figure
    labels = ['DT_G', 'DT_E', 'RF_G', 'RF_E', 'LR', 'LR_L1', 'LR_L2', 'LR_E', 'NB_B', 'NB_G', 'NB_M', 'NB_C', 'SVM_S',
              'SVM_P', 'SVM_R', 'KNN_W', 'KNN_D']

    #linear benchmarks
    bn_hc_linear_means, bn_tabu_linear_means, bn_mmhc_linear_means, bn_rsmax2_linear_means = [], [], [], []
    nt_log_linear_means, nt_l2_linear_means, nt_p_linear_means = [], [], []
    p_e_linear_means, p_g_linear_means = [], []
    pgmpy_tree_linear_means, pgmpy_hc_linear_means, pgmpy_mmhc_linear_means = [], [], []

    # Nonlinear benchmarks
    bn_hc_nonlinear_means, bn_tabu_nonlinear_means, bn_mmhc_nonlinear_means, bn_rsmax2_nonlinear_means = [], [], [], []
    nt_log_nonlinear_means, nt_l2_nonlinear_means, nt_p_nonlinear_means = [], [], []
    p_e_nonlinear_means, p_g_nonlinear_means = [], []
    pgmpy_tree_nonlinear_means, pgmpy_hc_nonlinear_means, pgmpy_mmhc_nonlinear_means = [], [], []

    # Sparse benchmarks
    bn_hc_sparse_means, bn_tabu_sparse_means, bn_mmhc_sparse_means, bn_rsmax2_sparse_means = [], [], [], []
    nt_log_sparse_means, nt_l2_sparse_means, nt_p_sparse_means = [], [], []
    p_e_sparse_means, p_g_sparse_means = [], []
    pgmpy_tree_sparse_means, pgmpy_hc_sparse_means, pgmpy_mmhc_sparse_means = [], [], []

    # Dimension benchmarks
    bn_hc_dimension_means, bn_tabu_dimension_means, bn_mmhc_dimension_means, bn_rsmax2_dimension_means = [], [], [], []
    nt_log_dimension_means, nt_l2_dimension_means, nt_p_dimension_means = [], [], []
    p_e_dimension_means, p_g_dimension_means = [], []
    pgmpy_tree_dimension_means, pgmpy_hc_dimension_means, pgmpy_mmhc_dimension_means = [], [], []
    for k, v in learned_results.items():
        if k.split('_')[-2] == "linear":
            if k.split('_')[-3] == "bnlearn-hc":
                bn_hc_linear_means.append(learned_results[k])
            elif k.split('_')[-3] == "bnlearn-tabu":
                bn_tabu_linear_means.append(learned_results[k])
            elif k.split('_')[-3] == "bnlearn-mmhc":
                bn_mmhc_linear_means.append(learned_results[k])
            elif k.split('_')[-3] == "bnlearn-rsmax2":
                bn_rsmax2_linear_means.append(learned_results[k])
            elif k.split('_')[-3] == "notears-logistic":
                nt_log_linear_means.append(learned_results[k])
            elif k.split('_')[-3] == "notears-l2":
                nt_l2_linear_means.append(learned_results[k])
            elif k.split('_')[-3] == "notears-poisson":
                nt_p_linear_means.append(learned_results[k])
            elif k.split('_')[-3] == "pgmpy-hc":
                pgmpy_hc_linear_means.append(learned_results[k])
            elif k.split('_')[-3] == "pgmpy-tree":
                pgmpy_tree_linear_means.append(learned_results[k])
            elif k.split('_')[-3] == "pgmpy-mmhc":
                pgmpy_mmhc_linear_means.append(learned_results[k])
            elif k.split('_')[-3] == "pomegranate-exact":
                p_e_linear_means.append(learned_results[k])
            elif k.split('_')[-3] == "pomegranate-greedy":
                p_g_linear_means.append(learned_results[k])
        elif k.split('_')[-2] == "non-linear":
            if k.split('_')[-3] == "bnlearn-hc":
                bn_hc_nonlinear_means.append(learned_results[k])
            elif k.split('_')[-3] == "bnlearn-tabu":
                bn_tabu_nonlinear_means.append(learned_results[k])
            elif k.split('_')[-3] == "bnlearn-mmhc":
                bn_mmhc_nonlinear_means.append(learned_results[k])
            elif k.split('_')[-3] == "bnlearn-rsmax2":
                bn_rsmax2_nonlinear_means.append(learned_results[k])
            elif k.split('_')[-3] == "notears-logistic":
                nt_log_nonlinear_means.append(learned_results[k])
            elif k.split('_')[-3] == "notears-l2":
                nt_l2_nonlinear_means.append(learned_results[k])
            elif k.split('_')[-3] == "notears-poisson":
                nt_p_nonlinear_means.append(learned_results[k])
            elif k.split('_')[-3] == "pgmpy-hc":
                pgmpy_hc_nonlinear_means.append(learned_results[k])
            elif k.split('_')[-3] == "pgmpy-tree":
                pgmpy_tree_nonlinear_means.append(learned_results[k])
            elif k.split('_')[-3] == "pgmpy-mmhc":
                pgmpy_mmhc_nonlinear_means.append(learned_results[k])
            elif k.split('_')[-3] == "pomegranate-exact":
                p_e_nonlinear_means.append(learned_results[k])
            elif k.split('_')[-3] == "pomegranate-greedy":
                p_g_nonlinear_means.append(learned_results[k])
        elif k.split('_')[-2] == "sparse":
            if k.split('_')[-3] == "bnlearn-hc":
                bn_hc_sparse_means.append(learned_results[k])
            elif k.split('_')[-3] == "bnlearn-tabu":
                bn_tabu_sparse_means.append(learned_results[k])
            elif k.split('_')[-3] == "bnlearn-mmhc":
                bn_mmhc_sparse_means.append(learned_results[k])
            elif k.split('_')[-3] == "bnlearn-rsmax2":
                bn_rsmax2_sparse_means.append(learned_results[k])
            elif k.split('_')[-3] == "notears-logistic":
                nt_log_sparse_means.append(learned_results[k])
            elif k.split('_')[-3] == "notears-l2":
                nt_l2_sparse_means.append(learned_results[k])
            elif k.split('_')[-3] == "notears-poisson":
                nt_p_sparse_means.append(learned_results[k])
            elif k.split('_')[-3] == "pgmpy-hc":
                pgmpy_hc_sparse_means.append(learned_results[k])
            elif k.split('_')[-3] == "pgmpy-tree":
                pgmpy_tree_sparse_means.append(learned_results[k])
            elif k.split('_')[-3] == "pgmpy-mmhc":
                pgmpy_mmhc_sparse_means.append(learned_results[k])
            elif k.split('_')[-3] == "pomegranate-exact":
                p_e_sparse_means.append(learned_results[k])
            elif k.split('_')[-3] == "pomegranate-greedy":
                p_g_sparse_means.append(learned_results[k])
        elif k.split('_')[-2] == "dimension":
            if k.split('_')[-3] == "bnlearn-hc":
                bn_hc_dimension_means.append(learned_results[k])
            elif k.split('_')[-3] == "bnlearn-tabu":
                bn_tabu_dimension_means.append(learned_results[k])
            elif k.split('_')[-3] == "bnlearn-mmhc":
                bn_mmhc_dimension_means.append(learned_results[k])
            elif k.split('_')[-3] == "bnlearn-rsmax2":
                bn_rsmax2_dimension_means.append(learned_results[k])
            elif k.split('_')[-3] == "notears-logistic":
                nt_log_dimension_means.append(learned_results[k])
            elif k.split('_')[-3] == "notears-l2":
                nt_l2_dimension_means.append(learned_results[k])
            elif k.split('_')[-3] == "notears-poisson":
                nt_p_dimension_means.append(learned_results[k])
            elif k.split('_')[-3] == "pgmpy-hc":
                pgmpy_hc_dimension_means.append(learned_results[k])
            elif k.split('_')[-3] == "pgmpy-tree":
                pgmpy_tree_dimension_means.append(learned_results[k])
            elif k.split('_')[-3] == "pgmpy-mmhc":
                pgmpy_mmhc_dimension_means.append(learned_results[k])
            elif k.split('_')[-3] == "pomegranate-exact":
                p_e_dimension_means.append(learned_results[k])
            elif k.split('_')[-3] == "pomegranate-greedy":
                p_g_dimension_means.append(learned_results[k])

    plt.rcParams["figure.figsize"] = [18, 18]
    plt.rcParams["figure.autolayout"] = True
    x_axis = np.arange(len(labels))
    w = 0.05  # the width of the bars
    plt.bar(x_axis + w, bn_hc_linear_means, width=0.05, label="BN_LEARN (HC)", color="lightsteelblue")
    plt.bar(x_axis + w * 2, bn_tabu_linear_means, width=0.05, label="BN_LEARN (TABU)", color="cornflowerblue")
    plt.bar(x_axis + w * 3, bn_mmhc_linear_means, width=0.05, label="BN_LEARN (MMHC)", color="blue")
    plt.bar(x_axis + w * 4, bn_rsmax2_linear_means, width=0.05, label="BN_LEARN (RSMAX2)", color="mediumblue")
    plt.bar(x_axis + w * 5, nt_log_linear_means, width=0.05, label="NO_TEARS (logistic)", color="limegreen")
    plt.bar(x_axis + w * 6, nt_l2_linear_means, width=0.05, label="NO_TEARS (l2)", color="forestgreen")
    plt.bar(x_axis + w * 7, nt_p_linear_means, width=0.05, label="NO_TEARS (poisson)", color="darkgreen")
    plt.bar(x_axis + w * 8, p_e_linear_means, width=0.05, label="POMEGRANATE (exact)", color="darkviolet")
    plt.bar(x_axis + w * 9, p_g_linear_means, width=0.05, label="POMEGRANATE (greed)", color="rebeccapurple")
    plt.bar(x_axis + w * 10, pgmpy_mmhc_linear_means, width=0.05, label="PGMPY (MMHC)", color="#FA8072")
    plt.bar(x_axis + w * 11, pgmpy_hc_linear_means, width=0.05, label="PGMPY (HC)", color="#FF2400")
    plt.bar(x_axis + w * 12, pgmpy_tree_linear_means, width=0.05, label="PGMPY (TREE)", color="#7C0A02")

    plt.xticks(x_axis, labels)
    plt.legend()
    plt.style.use("fivethirtyeight")
    plt.ylabel('Accuracy')
    plt.xlabel('ML Technique', labelpad=15)
    plt.title('Linear Problem - Performance by library on ML technique')
    plt.ylim(0.0, 1)
    plt.savefig('pipeline_summary_benchmark_for_linear_by_library_groupbar.png', bbox_inches='tight')
    plt.show()

    # Produce Non-Linear Problem by Library on Problem Group by figure
    plt.bar(x_axis + w, bn_hc_nonlinear_means, width=0.05, label="BN_LEARN (HC)", color="lightsteelblue")
    plt.bar(x_axis + w * 2, bn_tabu_nonlinear_means, width=0.05, label="BN_LEARN (TABU)", color="cornflowerblue")
    plt.bar(x_axis + w * 3, bn_mmhc_nonlinear_means, width=0.05, label="BN_LEARN (MMHC)", color="blue")
    plt.bar(x_axis + w * 4, bn_rsmax2_nonlinear_means, width=0.05, label="BN_LEARN (RSMAX2)", color="mediumblue")
    plt.bar(x_axis + w * 5, nt_log_nonlinear_means, width=0.05, label="NO_TEARS (logistic)", color="limegreen")
    plt.bar(x_axis + w * 6, nt_l2_nonlinear_means, width=0.05, label="NO_TEARS (l2)", color="forestgreen")
    plt.bar(x_axis + w * 7, nt_p_nonlinear_means, width=0.05, label="NO_TEARS (poisson)", color="darkgreen")
    plt.bar(x_axis + w * 8, p_e_nonlinear_means, width=0.05, label="POMEGRANATE (exact)", color="darkviolet")
    plt.bar(x_axis + w * 9, p_g_nonlinear_means, width=0.05, label="POMEGRANATE (greed)", color="rebeccapurple")
    plt.bar(x_axis + w * 10, pgmpy_mmhc_nonlinear_means, width=0.05, label="PGMPY (MMHC)", color="#FA8072")
    plt.bar(x_axis + w * 11, pgmpy_hc_nonlinear_means, width=0.05, label="PGMPY (HC)", color="#FF2400")
    plt.bar(x_axis + w * 12, pgmpy_tree_nonlinear_means, width=0.05, label="PGMPY (TREE)", color="#7C0A02")

    plt.xticks(x_axis, labels)
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('ML Technique', labelpad=15)
    plt.title('Non-Linear Problem - Performance by library on ML technique')
    plt.ylim(0.0, 1)
    plt.savefig('pipeline_summary_benchmark_for_nonlinear_by_library_groupbar.png', bbox_inches='tight')
    plt.show()

    # Produce Sparse Problem by Library on Problem Group by figure
    plt.bar(x_axis + w, bn_hc_sparse_means, width=0.05, label="BN_LEARN (HC)", color="lightsteelblue")
    plt.bar(x_axis + w * 2, bn_tabu_sparse_means, width=0.05, label="BN_LEARN (TABU)", color="cornflowerblue")
    plt.bar(x_axis + w * 3, bn_mmhc_sparse_means, width=0.05, label="BN_LEARN (MMHC)", color="blue")
    plt.bar(x_axis + w * 4, bn_rsmax2_sparse_means, width=0.05, label="BN_LEARN (RSMAX2)", color="mediumblue")
    plt.bar(x_axis + w * 5, nt_log_sparse_means, width=0.05, label="NO_TEARS (logistic)", color="limegreen")
    plt.bar(x_axis + w * 6, nt_l2_sparse_means, width=0.05, label="NO_TEARS (l2)", color="forestgreen")
    plt.bar(x_axis + w * 7, nt_p_sparse_means, width=0.05, label="NO_TEARS (poisson)", color="darkgreen")
    plt.bar(x_axis + w * 8, p_e_sparse_means, width=0.05, label="POMEGRANATE (exact)", color="darkviolet")
    plt.bar(x_axis + w * 9, p_g_sparse_means, width=0.05, label="POMEGRANATE (greed)", color="rebeccapurple")
    plt.bar(x_axis + w * 10, pgmpy_mmhc_sparse_means, width=0.05, label="PGMPY (MMHC)", color="#FA8072")
    plt.bar(x_axis + w * 11, pgmpy_hc_sparse_means, width=0.05, label="PGMPY (HC)", color="#FF2400")
    plt.bar(x_axis + w * 12, pgmpy_tree_sparse_means, width=0.05, label="PGMPY (TREE)", color="#7C0A02")

    plt.xticks(x_axis, labels)
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('ML Technique', labelpad=15)
    plt.title('Sparse Problem - Performance by library on ML technique')
    plt.ylim(0.0, 1)
    plt.savefig('pipeline_summary_benchmark_for_sparse_by_library_groupbar.png', bbox_inches='tight')
    plt.show()

    # Produce Dimensional Problem by Library on Problem Group by figure
    plt.bar(x_axis + w, bn_hc_dimension_means, width=0.05, label="BN_LEARN (HC)", color="lightsteelblue")
    plt.bar(x_axis + w * 2, bn_tabu_dimension_means, width=0.05, label="BN_LEARN (TABU)", color="cornflowerblue")
    plt.bar(x_axis + w * 3, bn_mmhc_dimension_means, width=0.05, label="BN_LEARN (MMHC)", color="blue")
    plt.bar(x_axis + w * 4, bn_rsmax2_dimension_means, width=0.05, label="BN_LEARN (RSMAX2)", color="mediumblue")
    plt.bar(x_axis + w * 5, nt_log_dimension_means, width=0.05, label="NO_TEARS (logistic)", color="limegreen")
    plt.bar(x_axis + w * 6, nt_l2_dimension_means, width=0.05, label="NO_TEARS (l2)", color="forestgreen")
    plt.bar(x_axis + w * 7, nt_p_dimension_means, width=0.05, label="NO_TEARS (poisson)", color="darkgreen")
    plt.bar(x_axis + w * 8, p_e_dimension_means, width=0.05, label="POMEGRANATE (exact)", color="darkviolet")
    plt.bar(x_axis + w * 9, p_g_dimension_means, width=0.05, label="POMEGRANATE (greed)", color="rebeccapurple")
    plt.bar(x_axis + w * 10, pgmpy_mmhc_dimension_means, width=0.05, label="PGMPY (MMHC)", color="#FA8072")
    plt.bar(x_axis + w * 11, pgmpy_hc_dimension_means, width=0.05, label="PGMPY (HC)", color="#FF2400")
    plt.bar(x_axis + w * 12, pgmpy_tree_dimension_means, width=0.05, label="PGMPY (TREE)", color="#7C0A02")

    plt.xticks(x_axis, labels)
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('ML Technique', labelpad=15)
    plt.title('Dimensional Problem - Performance by library on ML technique')
    plt.ylim(0.0, 1)
    plt.savefig('pipeline_summary_benchmark_for_dimensional_by_library_groupbar.png', bbox_inches='tight')
    plt.show()

write_results_to_figures()

def prediction_real_learned():
    '''
    Reports the comparative recommendation of ML method from SimCal benchmarks between real and learned worlds
    '''
    #List with values and loop
    print("#### SimCal Real/Learned-world Predictions ####")
    print("-- Exact (1-1) max(rank) output")
    top_real_linear = top_learned_linear = top_real_nonlinear = top_learned_nonlinear = top_real_sparse = top_learned_sparse = top_real_dimensional = top_learned_dimensional = 0
    top_real_linear_label = top_learned_linear_label = top_real_nonlinear_label = top_learned_nonlinear_label = top_real_sparse_label = top_learned_sparse_label = top_real_dimensional_label = top_learned_dimensional_label = ""

    for k, v in real_results.items():
        if k.split('_')[-2] == "linear":
            if real_results[k] > top_real_linear:
                top_real_linear = real_results[k]
                top_real_linear_label = k
        if k.split('_')[-2] == "non-linear":
            if real_results[k] > top_real_nonlinear:
                top_real_nonlinear = real_results[k]
                top_real_nonlinear_label = k
        if k.split('_')[-2] == "sparse":
            if real_results[k] > top_real_sparse:
                top_real_sparse = real_results[k]
                top_real_sparse_label = k
        if k.split('_')[-2] == "dimension":
            if real_results[k] > top_real_dimensional:
                top_real_dimensional = real_results[k]
                top_real_dimensional_label = k

    for k, v in learned_results.items():
        if k.split('_')[-2] == "linear":
            if learned_results[k] > top_learned_linear:
                top_learned_linear = learned_results[k]
                top_learned_linear_label = k
        if k.split('_')[-2] == "non-linear":
            if learned_results[k] > top_learned_nonlinear:
                top_learned_nonlinear = learned_results[k]
                top_learned_nonlinear_label = k
        if k.split('_')[-2] == "sparse":
            if learned_results[k] > top_learned_sparse:
                top_learned_sparse = learned_results[k]
                top_learned_sparse_label = k
        if k.split('_')[-2] == "dimension":
            if learned_results[k] > top_learned_dimensional:
                top_learned_dimensional = learned_results[k]
                top_learned_dimensional_label = k

    print("Real world - Linear problem, Prediction: " + top_real_linear_label + " ("+ str(top_real_linear)+")")
    print("Learned world - Linear problem, Prediction: "+ top_learned_linear_label + " (" +str(top_learned_linear)+")")
    print("Real world - Non-Linear problem, Prediction: " + top_real_nonlinear_label + " (" + str(top_real_nonlinear) + ")")
    print("Learned world - Non-Linear problem, Prediction: " + top_learned_nonlinear_label + " (" + str(top_learned_nonlinear) + ")")
    print("Real world - Sparse problem, Prediction: " + top_real_sparse_label + " (" + str(top_real_sparse) + ")")
    print("Learned world - Sparse problem, Prediction: " + top_learned_sparse_label + " (" + str(top_learned_sparse) + ")")
    print("Real world - Dimensional problem, Prediction: " + top_real_dimensional_label + " (" + str(top_real_dimensional) + ")")
    print("Learned world - Dimensional problem, Prediction: " + top_learned_dimensional_label + " (" + str(top_learned_dimensional) + ")")

real_experiment_summary = pd.read_csv("real_experiments_summary.csv")
real_experiment_summary

learned_experiment_summary = pd.read_csv("simulation_experiments_summary.csv")
learned_experiment_summary

prediction_real_learned()
