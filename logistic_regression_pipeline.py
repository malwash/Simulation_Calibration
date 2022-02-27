import random

#1 - compact dictionary into a dict or dict 414-436 homer simpson
#2 - Change csv simulated dataset to return for the function, change w_est, inside simulation_bnlearn, simulation_notears return instead of np.savetxt
#3 - Compact a single execution of a pipeline into a class 445-764
import importlib

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
import simulation_bnlearn
import simulation_dagsim
import simulation_models
from sklearn import metrics
from sklearn import svm
from notears import utils
from notears.linear import notears_linear
from notears.nonlinear import notears_nonlinear, NotearsMLP

#Save linear, nonlinear, sparse, dimensional training set of the real-world for reproducablity
global pipeline_type
global linear_training
global nonlinear_training
global sparse_training
global dimensional_training

# Attampt at globalising the training set of all pipelines from real world
# pipeline_type = 1
# simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#pipeline_type = 2
#nonlinear_training = simulation_dagsim.setup_realworld(pipeline_type, 10000, 5000)
#pipeline_type = 3
#sparse_training = simulation_dagsim.setup_realworld(pipeline_type, 10000, 5000)
#pipeline_type = 4
#dimensional_training = simulation_dagsim.setup_realworld(pipeline_type, 10000, 5000)

# import the saved training and test data from DagSim's real world
# def import_real_world_csv(pipeline_type):
#     global train_data
#     train_data = pd.read_csv("train.csv")
#     global train_data_numpy
#     train_data_numpy = train_data.to_numpy()
#     global x_train
#     global y_train
#     if(pipeline_type==4):
#         x_train = train_data.iloc[:, 0:10].to_numpy().reshape([-1, 10])  # num predictors
#         y_train = train_data.iloc[:, 10].to_numpy().reshape([-1]).ravel()  # outcome
#     elif(pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
#         x_train = train_data.iloc[:, 0:4].to_numpy().reshape([-1, 4])  # num predictors
#         y_train = train_data.iloc[:, 4].to_numpy().reshape([-1]).ravel()  # outcome
#
#     global test_data
#     global x_test
#     global y_test
#     test_data = pd.read_csv("test.csv")
#     if(pipeline_type==4):
#         x_test = test_data.iloc[:, 0:10].to_numpy().reshape([-1, 10])
#         y_test = test_data.iloc[:, 10].to_numpy().reshape([-1]).ravel()
#     elif(pipeline_type==1 or pipeline_type==2 or pipeline_type==3 ):
#         x_test = test_data.iloc[:, 0:4].to_numpy().reshape([-1, 4])
#         y_test = test_data.iloc[:, 4].to_numpy().reshape([-1]).ravel()

def slice_data(pipeline_type, train_data, test_data):
    if(pipeline_type==4):
        x_train = train_data.iloc[:, 0:10].to_numpy().reshape([-1, 10])  # num predictors
        y_train = train_data.iloc[:, 10].to_numpy().reshape([-1]).ravel()  # outcome
        x_test = test_data.iloc[:, 0:10].to_numpy().reshape([-1, 10])  # num predictors
        y_test = test_data.iloc[:, 10].to_numpy().reshape([-1]).ravel()  # outcome
    else:
        x_train = train_data.iloc[:, 0:4].to_numpy().reshape([-1, 4])  # num predictors
        y_train = train_data.iloc[:, 4].to_numpy().reshape([-1]).ravel()  # outcome
        x_test = test_data.iloc[:, 0:4].to_numpy().reshape([-1, 10])  # num predictors
        y_test = test_data.iloc[:, 4].to_numpy().reshape([-1]).ravel()  # outcome
    return x_train, y_train, x_test, y_test

def get_data_from_real_world(pipeline_type, num_train, num_test, world=None, data=None):
    '''
    Simulate data using dagsim based on the pipeline type
    :param pipeline_type: (str)
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

# def learn_world(pkg_name, train_data, config):
#     if pkg_name=="notears":
#         learned_model = notears_linear(train_data[0:100], lambda1=0.01, loss_type=config)
#     return learned_model

def get_data_from_learned_world(pkg_name, config, real_data, num_train, num_test, pipeline_type):
    # module = importlib.import_module(pkg_name)
    # function = getattr(module, f'{pkg_name}_setup_{config}')
    learned_data_train = None
    learned_data_test = None
    if pkg_name=="notears":
        model = notears_linear(real_data[0:100], lambda1=0.01, loss_type=config)
        learned_data_train = utils.simulate_linear_sem(model, num_train, 'logistic')
        learned_data_test = utils.simulate_linear_sem(model, num_test, 'logistic')
    x_train, y_train, x_test, y_test = slice_data(pipeline_type, learned_data_train, learned_data_test)
    return x_train, y_train, x_test, y_test

# Evaluate function for all ML techniques in the real-world
def world_evaluate(world, pipeline_type, x_train, y_train, x_test, y_test):
    scores = {}
    pipeline_name = ["linear", "non-linear", "sparse", "dimension"][pipeline_type]
    MLModels = {"DTCgini": DecisionTreeClassifier(criterion='gini'),
                "DTCent": DecisionTreeClassifier(criterion='entropy'),
                "RFCgini": RandomForestClassifier(criterion='gini'),
                "RFCent": RandomForestClassifier(criterion='entropy'),
                "LRnone": LogisticRegression(penalty='none'),
                "LRl1": LogisticRegression(penalty='l1', solver='liblinear', l1_ratio=1),
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
        scores[f'{world}_{pipeline_name}_{key}'] = cross_val_score(clf, x_test_rdy, y_test, cv=10)
    return scores
# Decision Tree
#     clf = DecisionTreeClassifier(criterion='gini')
#     clf = clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#     if(pipeline_type==1):
#         global real_linear_dt_scores
#         real_linear_dt_scores = cross_val_score(clf, x_train, y_train, cv=10)
#     elif(pipeline_type==2):
#         global real_nonlinear_dt_scores
#         real_nonlinear_dt_scores = cross_val_score(clf, x_train, y_train, cv=10)
#     elif(pipeline_type==3):
#         global real_sparse_dt_scores
#         real_sparse_dt_scores = cross_val_score(clf, x_train, y_train, cv=10)
#     elif (pipeline_type == 4):
#         global real_dimension_dt_scores
#         real_dimension_dt_scores = cross_val_score(clf, x_train, y_train, cv=10)



print("This is the first occurance of the real-world benchmarks")

def evaluate_real(num_train, num_test):
    for pipeline_type in range(1,5):
        x_train, y_train, x_test, y_test = get_data_from_real_world(pipeline_type, num_train, num_test)
        world_evaluate("real", pipeline_type, x_train, y_train, x_test, y_test)

# pipeline_type = 2
# simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#
# realworld_evaluate(pipeline_type)

# import_real_world_csv(pipeline_type)
#
# # Simulation library structure learning section
#
# print("This is the first occurance of the simulated benchmarks")
# simulation_notears.notears_setup(train_data_numpy[0:100], 1000, 5000)

#simulation_notears.notears_nonlinear_setup(train_data_numpy[0:100], 10000, 5000)

# import the saved training and test data from the simulation framework's learned world
def import_simulated_csv():
    global no_tears_sample_train
    no_tears_sample_train= pd.read_csv('W_est_train.csv')
    #global no_tears_sample_test
    #no_tears_sample_test = pd.read_csv('W_est_test.csv')
    #global no_tears_nonlinear_sample_train
    #no_tears_nonlinear_sample_train = pd.read_csv('K_est_train.csv')
    #global no_tears_nonlinear_sample_test
    #no_tears_nonlinear_sample_test = pd.read_csv('K_est_test.csv')
    global bn_learn_sample_train
    bn_learn_sample_train = pd.read_csv('Z_est_train.csv')
    #global bn_learn_sample_test
    #bn_learn_sample_test = pd.read_csv('Z_est_test.csv')

import_simulated_csv()

def run_learned_workflows(x_train, y_train, pipeline_type, alg):
    print("alg:"+alg+", pipeline:"+str(pipeline_type))
    my_dict = {"alg": alg, "pl": pipeline_type, "dt": 0, "dt_e": 0, "rf": 0, "rf_E": 0,"lr": 0, "lr_l1": 0, "lr_l2": 0, "lr_e": 0, "nb": 0, "nb_g": 0,"nb_m": 0,"nb_c": 0,"svm": 0,"svm_l": 0,"svm_po": 0,"svm_r": 0,"svm_pr": 0, "knn": 0, "knn_d": 0}
    my_dict["dt"] = simulation_models.decision_tree(x_train, y_train, x_test, y_test)
    my_dict["dt_e"] = simulation_models.decision_tree_entropy(x_train, y_train, x_test, y_test)
    my_dict["rf"] = simulation_models.random_forest(x_train, y_train, x_test, y_test)
    my_dict["rf_e"] = simulation_models.random_forest_entropy(x_train, y_train, x_test, y_test)
    my_dict["lr"] = simulation_models.logistic_regression(x_train, y_train, x_test, y_test)
    my_dict["lr_l1"] = simulation_models.logistic_regression_l1(x_train, y_train, x_test, y_test)
    my_dict["lr_l2"] = simulation_models.logistic_regression_l2(x_train, y_train, x_test, y_test)
    my_dict["lr_e"] = simulation_models.logistic_regression_elastic(x_train, y_train, x_test, y_test)
    my_dict["nb"] = simulation_models.naive_bayes(x_train, y_train, x_test, y_test)
    my_dict["nb_g"] = simulation_models.naive_bayes_gaussian(x_train, y_train, x_test, y_test)
    my_dict["nb_m"] = simulation_models.naive_bayes_multinomial(x_train, y_train, x_test, y_test)
    my_dict["nb_c"] = simulation_models.naive_bayes_complement(x_train, y_train, x_test, y_test)
    my_dict["svm"] = simulation_models.support_vector_machines(x_train, y_train, x_test, y_test)
    #my_dict["svm_l"] = simulation_models.support_vector_machines_linear(x_train, y_train, x_test, y_test)
    my_dict["svm_po"] = simulation_models.support_vector_machines_poly(x_train, y_train, x_test, y_test)
    my_dict["svm_r"] = simulation_models.support_vector_machines_rbf(x_train, y_train, x_test, y_test)
    #my_dict["svm_pr"] = simulation_models.support_vector_machines_precomputed(x_train, y_train, x_test, y_test)
    my_dict["knn"] = simulation_models.k_nearest_neighbor(x_train, y_train, x_test, y_test)
    my_dict["knn_d"] = simulation_models.k_nearest_neighbor_distance(x_train, y_train, x_test, y_test)
    return my_dict

#helper function to execute one workflow with parameterised setup
#def execute_pipeline(x_train, y_train, run_pipeline_type, pipeline_title):
#    pipeline_type = 2
#    simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#    import_real_world_csv(pipeline_type)

#notears simulation scoring
# notears_linear_dict_scores = run_learned_workflows(no_tears_sample_train.iloc[:,0:4], no_tears_sample_train.iloc[:,4], pipeline_type, "NO TEARS (Logistic)")
#
# simulation_notears.notears_setup(train_data_numpy[0:100], 1000, 5000)
# import_simulated_csv()
#
# notears_nonlinear_dict_scores = run_learned_workflows(no_tears_sample_train.iloc[:,0:4], no_tears_sample_train.iloc[:,4], pipeline_type, "NO TEARS (Logistic)")

def evaluate_on_learned_world(num_train_rl, num_train_lr, num_test_lr):
    world = "notears"
    loss_names = ["logistic", "l2", "poisson"]
    pipelines = list(range(1, 5))
    results = {}
    for pipeline in pipelines:
        x_train_rl, y_train_rl, x_test_rl, y_test_rl = get_data_from_real_world(pipeline, num_train_rl, 0)
        for loss in loss_names:
            x_train_lr, y_train_lr, x_test_lr, y_test_lr = get_data_from_learned_world(world, loss, np.concatenate([x_train_rl, y_train_rl], axis=1), num_train_lr, num_test_lr, pipeline_type)
            scores = world_evaluate(world, pipeline, x_train_lr, y_train_lr, x_test_lr, y_test_lr)
            results.update(scores)
    return results

# pipeline_type = 3
# simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
# import_real_world_csv(pipeline_type)
# simulation_notears.notears_setup(train_data_numpy[0:100], 1000, 5000)
# import_simulated_csv()
#
# notears_sparse_dict_scores = run_learned_workflows(no_tears_sample_train.iloc[:,0:4], no_tears_sample_train.iloc[:,4], pipeline_type, "NO TEARS (Logistic)")
# pipeline_type = 4
# simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
# import_real_world_csv(pipeline_type)
# simulation_notears.notears_setup(train_data_numpy[0:100], 1000, 5000)
# import_simulated_csv()
#
# notears_dimension_dict_scores = run_learned_workflows(no_tears_sample_train.iloc[:,0:10], no_tears_sample_train.iloc[:,10], pipeline_type, "NO TEARS (Logistic)")
#
# #notears hyperparameter loss function l2
# pipeline_type = 1
# simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
# import_real_world_csv(pipeline_type)
# simulation_notears.notears_setup_b(train_data_numpy[0:100], 1000, 5000)
# import_simulated_csv()
# notears_l2_linear_dict_scores = run_learned_workflows(no_tears_sample_train.iloc[:,0:4], no_tears_sample_train.iloc[:,4], pipeline_type, "NO TEARS (L2)")
#
# pipeline_type = 2
# simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
# import_real_world_csv(pipeline_type)
# simulation_notears.notears_setup_b(train_data_numpy[0:100], 1000, 5000)
# import_simulated_csv()
#
# notears_l2_nonlinear_dict_scores = run_learned_workflows(no_tears_sample_train.iloc[:,0:4], no_tears_sample_train.iloc[:,4], pipeline_type, "NO TEARS (L2)")
# pipeline_type = 3
# simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
# import_real_world_csv(pipeline_type)
# simulation_notears.notears_setup_b(train_data_numpy[0:100], 1000, 5000)
# import_simulated_csv()
#
# notears_l2_sparse_dict_scores = run_learned_workflows(no_tears_sample_train.iloc[:,0:4], no_tears_sample_train.iloc[:,4], pipeline_type, "NO TEARS (L2)")
# pipeline_type = 4
# simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
# import_real_world_csv(pipeline_type)
# simulation_notears.notears_setup_b(train_data_numpy[0:100], 1000, 5000)
# import_simulated_csv()
#
# notears_l2_dimension_dict_scores = run_learned_workflows(no_tears_sample_train.iloc[:,0:10], no_tears_sample_train.iloc[:,10], pipeline_type, "NO TEARS (L2)")
#
# #notears hyperparameter loss function poisson
# pipeline_type = 1
# simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
# import_real_world_csv(pipeline_type)
# simulation_notears.notears_setup_c(train_data_numpy[0:100], 1000, 5000)
# import_simulated_csv()
# notears_poisson_linear_dict_scores = run_learned_workflows(no_tears_sample_train.iloc[:,0:4], no_tears_sample_train.iloc[:,4], pipeline_type, "NO TEARS (Poisson)")
#
# pipeline_type = 2
# simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
# import_real_world_csv(pipeline_type)
# simulation_notears.notears_setup_c(train_data_numpy[0:100], 1000, 5000)
# import_simulated_csv()
#
# notears_poisson_nonlinear_dict_scores = run_learned_workflows(no_tears_sample_train.iloc[:,0:4], no_tears_sample_train.iloc[:,4], pipeline_type, "NO TEARS (Poisson)")
# pipeline_type = 3
# simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
# import_real_world_csv(pipeline_type)
# simulation_notears.notears_setup_c(train_data_numpy[0:100], 1000, 5000)
# import_simulated_csv()
#
# notears_poisson_sparse_dict_scores = run_learned_workflows(no_tears_sample_train.iloc[:,0:4], no_tears_sample_train.iloc[:,4], pipeline_type, "NO TEARS (Poisson)")
# pipeline_type = 4
# simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
# import_real_world_csv(pipeline_type)
# simulation_notears.notears_setup_c(train_data_numpy[0:100], 1000, 5000)
# import_simulated_csv()
#
# notears_poisson_dimension_dict_scores = run_learned_workflows(no_tears_sample_train.iloc[:,0:10], no_tears_sample_train.iloc[:,10], pipeline_type, "NO TEARS (Poisson)")

#bnlearn simulation scoring
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_hc(train_data[0:100], pipeline_type)
import_simulated_csv()

bnlearn_linear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (HC)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_hc(train_data[0:100], pipeline_type)
import_simulated_csv()

bnlearn_nonlinear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (HC)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_hc(train_data[0:100], pipeline_type)
import_simulated_csv()

bnlearn_sparse_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (HC)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_hc(train_data[0:100], pipeline_type)
import_simulated_csv()
bnlearn_dimension_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:10], bn_learn_sample_train.iloc[:,10], pipeline_type, "BN LEARN (HC)")

#Run hyperparameter of bnlearn - tabu
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_tabu(train_data[0:100], pipeline_type)
import_simulated_csv()

bnlearn_tabu_linear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (TABU)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_tabu(train_data[0:100], pipeline_type)
import_simulated_csv()

bnlearn_tabu_nonlinear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (TABU)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_tabu(train_data[0:100], pipeline_type)
import_simulated_csv()

bnlearn_tabu_sparse_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (TABU)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_tabu(train_data[0:100], pipeline_type)
import_simulated_csv()
bnlearn_tabu_dimension_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:10], bn_learn_sample_train.iloc[:,10], pipeline_type, "BN LEARN (TABU)")
#end of tabu workflows

#Run hyperparameter of bnlearn - pc
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_pc(train_data[0:100], pipeline_type)
import_simulated_csv()

bnlearn_pc_linear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (PC)")
#pipeline_type = 2
#simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_bnlearn.bnlearn_setup_pc(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed.
#import_simulated_csv()

#bnlearn_pc_nonlinear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (PC)")
#pipeline_type = 3
#simulation_dagsim.setup_realworld(pipeline_type, 10000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_bnlearn.bnlearn_setup_pc(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed.
#import_simulated_csv()

#bnlearn_pc_sparse_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (PC)")
#pipeline_type = 4
#simulation_dagsim.setup_realworld(pipeline_type, 10000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_bnlearn.bnlearn_setup_pc(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
#import_simulated_csv()
#bnlearn_pc_dimension_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:2], bn_learn_sample_train.iloc[:,2], pipeline_type, "BN LEARN (PC)")
#end of pc workflows

#Run hyperparameter of bnlearn - gs
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_gs(train_data[0:100], pipeline_type)
import_simulated_csv()

bnlearn_gs_linear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (GS)")
#pipeline_type = 2
#simulation_dagsim.setup_realworld(pipeline_type, 10000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_bnlearn.bnlearn_setup_gs(train_data[0:100], pipeline_type)
#import_simulated_csv()

#bnlearn_gs_nonlinear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (GS)")
#pipeline_type = 3
#simulation_dagsim.setup_realworld(pipeline_type, 10000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_bnlearn.bnlearn_setup_gs(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
#import_simulated_csv()

#bnlearn_gs_sparse_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (GS)")
#pipeline_type = 4
#simulation_dagsim.setup_realworld(pipeline_type, 10000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_bnlearn.bnlearn_setup_gs(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
#import_simulated_csv()
#bnlearn_gs_dimension_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:2], bn_learn_sample_train.iloc[:,2], pipeline_type, "BN LEARN (GS)")
#end of gs workflows

#Run hyperparameter of bnlearn - iamb
#pipeline_type = 1
#simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_bnlearn.bnlearn_setup_iamb(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed.
#import_simulated_csv()

#bnlearn_iamb_linear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (IAMB)")
#pipeline_type = 2
#simulation_dagsim.setup_realworld(pipeline_type, 10000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_bnlearn.bnlearn_setup_iamb(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
#import_simulated_csv()

#bnlearn_iamb_nonlinear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (IAMB)")
#pipeline_type = 3
#simulation_dagsim.setup_realworld(pipeline_type, 10000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_bnlearn.bnlearn_setup_iamb(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
#import_simulated_csv()

#bnlearn_iamb_sparse_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (IAMB)")
#pipeline_type = 4
#simulation_dagsim.setup_realworld(pipeline_type, 10000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_bnlearn.bnlearn_setup_iamb(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
#import_simulated_csv()
#bnlearn_iamb_dimension_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:2], bn_learn_sample_train.iloc[:,2], pipeline_type, "BN LEARN (IAMB)")
#end of pc workflows

#Run hyperparameter of bnlearn - mmhc
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_mmhc(train_data[0:100], pipeline_type)
import_simulated_csv()

bnlearn_mmhc_linear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (MMHC)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_mmhc(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
import_simulated_csv()

bnlearn_mmhc_nonlinear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (MMHC)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_mmhc(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
import_simulated_csv()

bnlearn_mmhc_sparse_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (MMHC)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_mmhc(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
import_simulated_csv()
bnlearn_mmhc_dimension_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:10], bn_learn_sample_train.iloc[:,10], pipeline_type, "BN LEARN (MMHC)")
#end of mmhc workflows

#Run hyperparameter of bnlearn - rsmax2
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_rsmax2(train_data[0:100], pipeline_type)
import_simulated_csv()

bnlearn_rsmax2_linear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (RSMAX2)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_rsmax2(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
import_simulated_csv()

bnlearn_rsmax2_nonlinear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (RSMAX2)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_rsmax2(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
import_simulated_csv()

bnlearn_rsmax2_sparse_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (RSMAX2)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_rsmax2(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
import_simulated_csv()
bnlearn_rsmax2_dimension_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:10], bn_learn_sample_train.iloc[:,10], pipeline_type, "BN LEARN (RSMAX2)")
#end of rsmax2 workflows

#Run hyperparameter of bnlearn - h2pc
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_h2pc(train_data[0:100], pipeline_type)
import_simulated_csv()

bnlearn_h2pc_linear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (H2PC)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_h2pc(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
import_simulated_csv()

bnlearn_h2pc_nonlinear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (H2PC)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_h2pc(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
import_simulated_csv()

bnlearn_h2pc_sparse_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (H2PC)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulation_bnlearn.bnlearn_setup_h2pc(train_data[0:100], pipeline_type) #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
import_simulated_csv()
bnlearn_h2pc_dimension_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:10], bn_learn_sample_train.iloc[:,10], pipeline_type, "BN LEARN (H2PC)")
#end of h2pc workflows

def write_learned_to_csv():
    experiments = ['Algorithm', 'Model', 'Linear', 'Non-linear', 'Sparsity', 'Dimensionality']
    with open('simulation_experiments_summary.csv', 'w', newline='') as csvfile:
        fieldnames = ['Algorithm', 'Model', 'Linear', 'Non-linear', 'Sparsity', 'Dimensionality']
        thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        thewriter.writeheader()

        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Decision Tree (gini)','Linear': str(round(mean(notears_l2_linear_dict_scores["dt"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["dt"])) + "," + str(max(notears_l2_linear_dict_scores["dt"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["dt"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["dt"])) + "," + str(max(notears_l2_nonlinear_dict_scores["dt"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["dt"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["dt"])) + "," + str(max(notears_l2_sparse_dict_scores["dt"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["dt"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["dt"])) + "," + str(max(notears_l2_dimension_dict_scores["dt"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Decision Tree (entropy)','Linear': str(round(mean(notears_l2_linear_dict_scores["dt_e"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["dt_e"])) + "," + str(max(notears_l2_linear_dict_scores["dt_e"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["dt_e"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["dt_e"])) + "," + str(max(notears_l2_nonlinear_dict_scores["dt_e"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["dt_e"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["dt_e"])) + "," + str(max(notears_l2_sparse_dict_scores["dt_e"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["dt_e"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["dt_e"])) + "," + str(max(notears_l2_dimension_dict_scores["dt_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Random Forest (gini)','Linear': str(round(mean(notears_l2_linear_dict_scores["rf"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["rf"])) + "," + str(max(notears_l2_linear_dict_scores["rf"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["rf"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["rf"])) + "," + str(max(notears_l2_nonlinear_dict_scores["rf"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["rf"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["rf"])) + "," + str(max(notears_l2_sparse_dict_scores["rf"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["rf"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["rf"])) + "," + str(max(notears_l2_dimension_dict_scores["rf"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Random Forest (entropy)','Linear': str(round(mean(notears_l2_linear_dict_scores["rf_e"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["rf_e"])) + "," + str(max(notears_l2_linear_dict_scores["rf_e"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["rf_e"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["rf_e"])) + "," + str(max(notears_l2_nonlinear_dict_scores["rf_e"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["rf_e"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["rf_e"])) + "," + str(max(notears_l2_sparse_dict_scores["rf_e"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["rf_e"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["rf_e"])) + "," + str(max(notears_l2_dimension_dict_scores["rf_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(round(mean(notears_l2_linear_dict_scores["lr"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["lr"])) + "," + str(max(notears_l2_linear_dict_scores["lr"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["lr"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["lr"])) + "," + str(max(notears_l2_nonlinear_dict_scores["lr"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["lr"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["lr"])) + "," + str(max(notears_l2_sparse_dict_scores["lr"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["lr"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["lr"])) + "," + str(max(notears_l2_dimension_dict_scores["lr"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Logistic Regression (l1)','Linear': str(round(mean(notears_l2_linear_dict_scores["lr_l1"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["lr_l1"])) + "," + str(max(notears_l2_linear_dict_scores["lr_l1"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["lr_l1"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["lr_l1"])) + "," + str(max(notears_l2_nonlinear_dict_scores["lr_l1"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["lr_l1"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["lr_l1"])) + "," + str(max(notears_l2_sparse_dict_scores["lr_l1"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["lr_l1"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["lr_l1"])) + "," + str(max(notears_l2_dimension_dict_scores["lr_l1"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Logistic Regression (l2)','Linear': str(round(mean(notears_l2_linear_dict_scores["lr_l2"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["lr_l2"])) + "," + str(max(notears_l2_linear_dict_scores["lr_l2"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["lr_l2"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["lr_l2"])) + "," + str(max(notears_l2_nonlinear_dict_scores["lr_l2"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["lr_l2"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["lr_l2"])) + "," + str(max(notears_l2_sparse_dict_scores["lr_l2"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["lr_l2"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["lr_l2"])) + "," + str(max(notears_l2_dimension_dict_scores["lr_l2"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(round(mean(notears_l2_linear_dict_scores["lr_e"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["lr_e"])) + "," + str(max(notears_l2_linear_dict_scores["lr_e"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["lr_e"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["lr_e"])) + "," + str(max(notears_l2_nonlinear_dict_scores["lr_e"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["lr_e"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["lr_e"])) + "," + str(max(notears_l2_sparse_dict_scores["lr_e"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["lr_e"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["lr_e"])) + "," + str(max(notears_l2_dimension_dict_scores["lr_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(round(mean(notears_l2_linear_dict_scores["nb"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["nb"])) + "," + str(max(notears_l2_linear_dict_scores["nb"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["nb"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["nb"])) + "," + str(max(notears_l2_nonlinear_dict_scores["nb"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["nb"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["nb"])) + "," + str(max(notears_l2_sparse_dict_scores["nb"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["nb"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["nb"])) + "," + str(max(notears_l2_dimension_dict_scores["nb"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(round(mean(notears_l2_linear_dict_scores["nb_m"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["nb_m"])) + "," + str(max(notears_l2_linear_dict_scores["nb_m"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["nb_m"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["nb_m"])) + "," + str(max(notears_l2_nonlinear_dict_scores["nb_m"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["nb_m"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["nb_m"])) + "," + str(max(notears_l2_sparse_dict_scores["nb_m"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["nb_m"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["nb_m"])) + "," + str(max(notears_l2_dimension_dict_scores["nb_m"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(round(mean(notears_l2_linear_dict_scores["nb_g"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["nb_g"])) + "," + str(max(notears_l2_linear_dict_scores["nb_g"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["nb_g"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["nb_g"])) + "," + str(max(notears_l2_nonlinear_dict_scores["nb_g"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["nb_g"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["nb_g"])) + "," + str(max(notears_l2_sparse_dict_scores["nb_g"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["nb_g"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["nb_g"])) + "," + str(max(notears_l2_dimension_dict_scores["nb_g"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Naive Bayes (Complement)','Linear': str(round(mean(notears_l2_linear_dict_scores["nb_c"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["nb_c"])) + "," + str(max(notears_l2_linear_dict_scores["nb_c"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["nb_c"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["nb_c"])) + "," + str(max(notears_l2_nonlinear_dict_scores["nb_c"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["nb_c"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["nb_c"])) + "," + str(max(notears_l2_sparse_dict_scores["nb_c"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["nb_c"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["nb_c"])) + "," + str(max(notears_l2_dimension_dict_scores["nb_c"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm"])) + "," + str(max(notears_l2_linear_dict_scores["svm"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm"])) + "," + str(max(notears_l2_sparse_dict_scores["svm"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm"])) + "," + str(max(notears_l2_dimension_dict_scores["svm"])) + "}"})
        #thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_l"])) + "," + str(max(notears_l2_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_l"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_l"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_l"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_l"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (poly)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_po"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_po"])) + "," + str(max(notears_l2_linear_dict_scores["svm_po"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_po"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_po"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_po"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_po"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_po"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_po"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_po"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_po"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_po"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (rbf)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_r"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_r"])) + "," + str(max(notears_l2_linear_dict_scores["svm_r"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_r"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_r"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_r"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_r"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_r"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_r"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_r"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_r"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_r"])) + "}"})
        #thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_pr"])) + "," + str(max(notears_l2_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_pr"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_pr"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_pr"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_pr"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(round(mean(notears_l2_linear_dict_scores["knn"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["knn"])) + "," + str(max(notears_l2_linear_dict_scores["knn"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["knn"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["knn"])) + "," + str(max(notears_l2_nonlinear_dict_scores["knn"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["knn"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["knn"])) + "," + str(max(notears_l2_sparse_dict_scores["knn"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["knn"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["knn"])) + "," + str(max(notears_l2_dimension_dict_scores["knn"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(round(mean(notears_l2_linear_dict_scores["knn_d"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["knn_d"])) + "," + str(max(notears_l2_linear_dict_scores["knn_d"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["knn_d"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["knn_d"])) + "," + str(max(notears_l2_nonlinear_dict_scores["knn_d"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["knn_d"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["knn_d"])) + "," + str(max(notears_l2_sparse_dict_scores["knn_d"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["knn_d"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["knn_d"])) + "," + str(max(notears_l2_dimension_dict_scores["knn_d"])) + "}"})

        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Decision Tree (gini)','Linear': str(round(mean(notears_linear_dict_scores["dt"]),2))+" {"+str(min(notears_linear_dict_scores["dt"]))+","+str(max(notears_linear_dict_scores["dt"]))+"}", 'Non-linear': str(round(mean(notears_nonlinear_dict_scores["dt"]),2))+" {"+str(min(notears_nonlinear_dict_scores["dt"]))+","+str(max(notears_nonlinear_dict_scores["dt"]))+"}", 'Sparsity': str(round(mean(notears_sparse_dict_scores["dt"]),2))+" {"+str(min(notears_sparse_dict_scores["dt"]))+","+str(max(notears_sparse_dict_scores["dt"]))+"}", 'Dimensionality': str(round(mean(notears_dimension_dict_scores["dt"]),2))+" {"+str(min(notears_dimension_dict_scores["dt"]))+","+str(max(notears_dimension_dict_scores["dt"]))+"}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Decision Tree (entropy)','Linear': str(round(mean(notears_linear_dict_scores["dt_e"]), 2)) + " {" + str(min(notears_linear_dict_scores["dt_e"])) + "," + str(max(notears_linear_dict_scores["dt_e"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["dt_e"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["dt_e"])) + "," + str(max(notears_nonlinear_dict_scores["dt_e"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["dt_e"]), 2)) + " {" + str(min(notears_sparse_dict_scores["dt_e"])) + "," + str(max(notears_sparse_dict_scores["dt_e"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["dt_e"]), 2)) + " {" + str(min(notears_dimension_dict_scores["dt_e"])) + "," + str(max(notears_dimension_dict_scores["dt_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Random Forest (gini)', 'Linear': str(round(mean(notears_linear_dict_scores["rf"]),2))+" {"+str(min(notears_linear_dict_scores["rf"]))+","+str(max(notears_linear_dict_scores["rf"]))+"}", 'Non-linear': str(round(mean(notears_nonlinear_dict_scores["rf"]),2))+" {"+str(min(notears_nonlinear_dict_scores["rf"]))+","+str(max(notears_nonlinear_dict_scores["rf"]))+"}", 'Sparsity': str(round(mean(notears_sparse_dict_scores["rf"]),2))+" {"+str(min(notears_sparse_dict_scores["rf"]))+","+str(max(notears_sparse_dict_scores["rf"]))+"}", 'Dimensionality': str(round(mean(notears_dimension_dict_scores["rf"]),2))+" {"+str(min(notears_dimension_dict_scores["rf"]))+","+str(max(notears_dimension_dict_scores["rf"]))+"}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Random Forest (entropy)','Linear': str(round(mean(notears_linear_dict_scores["rf_e"]), 2)) + " {" + str(min(notears_linear_dict_scores["rf_e"])) + "," + str(max(notears_linear_dict_scores["rf_e"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["rf_e"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["rf_e"])) + "," + str(max(notears_nonlinear_dict_scores["rf_e"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["rf_e"]), 2)) + " {" + str(min(notears_sparse_dict_scores["rf_e"])) + "," + str(max(notears_sparse_dict_scores["rf_e"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["rf_e"]), 2)) + " {" + str(min(notears_dimension_dict_scores["rf_e"])) + "," + str(max(notears_dimension_dict_scores["rf_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Logistic Regression (penalty-none)', 'Linear': str(round(mean(notears_linear_dict_scores["lr"]),2))+" {"+str(min(notears_linear_dict_scores["lr"]))+","+str(max(notears_linear_dict_scores["lr"]))+"}", 'Non-linear': str(round(mean(notears_nonlinear_dict_scores["lr"]),2))+" {"+str(min(notears_nonlinear_dict_scores["lr"]))+","+str(max(notears_nonlinear_dict_scores["lr"]))+"}", 'Sparsity': str(round(mean(notears_sparse_dict_scores["lr"]),2))+" {"+str(min(notears_sparse_dict_scores["lr"]))+","+str(max(notears_sparse_dict_scores["lr"]))+"}", 'Dimensionality': str(round(mean(notears_dimension_dict_scores["lr"]),2))+" {"+str(min(notears_dimension_dict_scores["lr"]))+","+str(max(notears_dimension_dict_scores["lr"]))+"}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Logistic Regression (l1)','Linear': str(round(mean(notears_linear_dict_scores["lr_l1"]), 2)) + " {" + str(min(notears_linear_dict_scores["lr_l1"])) + "," + str(max(notears_linear_dict_scores["lr_l1"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["lr_l1"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["lr_l1"])) + "," + str(max(notears_nonlinear_dict_scores["lr_l1"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["lr_l1"]), 2)) + " {" + str(min(notears_sparse_dict_scores["lr_l1"])) + "," + str(max(notears_sparse_dict_scores["lr_l1"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["lr_l1"]), 2)) + " {" + str(min(notears_dimension_dict_scores["lr_l1"])) + "," + str(max(notears_dimension_dict_scores["lr_l1"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Logistic Regression (l2)','Linear': str(round(mean(notears_linear_dict_scores["lr_l2"]), 2)) + " {" + str(min(notears_linear_dict_scores["lr_l2"])) + "," + str(max(notears_linear_dict_scores["lr_l2"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["lr_l2"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["lr_l2"])) + "," + str(max(notears_nonlinear_dict_scores["lr_l2"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["lr_l2"]), 2)) + " {" + str(min(notears_sparse_dict_scores["lr_l2"])) + "," + str(max(notears_sparse_dict_scores["lr_l2"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["lr_l2"]), 2)) + " {" + str(min(notears_dimension_dict_scores["lr_l2"])) + "," + str(max(notears_dimension_dict_scores["lr_l2"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(round(mean(notears_linear_dict_scores["lr_e"]), 2)) + " {" + str(min(notears_linear_dict_scores["lr_e"])) + "," + str(max(notears_linear_dict_scores["lr_e"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["lr_e"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["lr_e"])) + "," + str(max(notears_nonlinear_dict_scores["lr_e"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["lr_e"]), 2)) + " {" + str(min(notears_sparse_dict_scores["lr_e"])) + "," + str(max(notears_sparse_dict_scores["lr_e"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["lr_e"]), 2)) + " {" + str(min(notears_dimension_dict_scores["lr_e"])) + "," + str(max(notears_dimension_dict_scores["lr_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Naive Bayes (Bernoulli)', 'Linear': str(round(mean(notears_linear_dict_scores["nb"]),2))+" {"+str(min(notears_linear_dict_scores["nb"]))+","+str(max(notears_linear_dict_scores["nb"]))+"}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["nb"]),2))+" {"+str(min(notears_nonlinear_dict_scores["nb"]))+","+str(max(notears_nonlinear_dict_scores["nb"]))+"}", 'Sparsity': str(round(mean(notears_sparse_dict_scores["nb"]),2))+" {"+str(min(notears_sparse_dict_scores["nb"]))+","+str(max(notears_sparse_dict_scores["nb"]))+"}", 'Dimensionality': str(round(mean(notears_dimension_dict_scores["nb"]),2))+" {"+str(min(notears_dimension_dict_scores["nb"]))+","+str(max(notears_dimension_dict_scores["nb"]))+"}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(round(mean(notears_linear_dict_scores["nb_m"]), 2)) + " {" + str(min(notears_linear_dict_scores["nb_m"])) + "," + str(max(notears_linear_dict_scores["nb_m"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["nb_m"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["nb_m"])) + "," + str(max(notears_nonlinear_dict_scores["nb_m"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["nb_m"]), 2)) + " {" + str(min(notears_sparse_dict_scores["nb_m"])) + "," + str(max(notears_sparse_dict_scores["nb_m"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["nb_m"]), 2)) + " {" + str(min(notears_dimension_dict_scores["nb_m"])) + "," + str(max(notears_dimension_dict_scores["nb_m"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(round(mean(notears_linear_dict_scores["nb_g"]), 2)) + " {" + str(min(notears_linear_dict_scores["nb_g"])) + "," + str(max(notears_linear_dict_scores["nb_g"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["nb_g"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["nb_g"])) + "," + str(max(notears_nonlinear_dict_scores["nb_g"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["nb_g"]), 2)) + " {" + str(min(notears_sparse_dict_scores["nb_g"])) + "," + str(max(notears_sparse_dict_scores["nb_g"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["nb_g"]), 2)) + " {" + str(min(notears_dimension_dict_scores["nb_g"])) + "," + str(max(notears_dimension_dict_scores["nb_g"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Naive Bayes (Complement)','Linear': str(round(mean(notears_linear_dict_scores["nb_c"]), 2)) + " {" + str(min(notears_linear_dict_scores["nb_c"])) + "," + str(max(notears_linear_dict_scores["nb_c"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["nb_c"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["nb_c"])) + "," + str(max(notears_nonlinear_dict_scores["nb_c"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["nb_c"]), 2)) + " {" + str(min(notears_sparse_dict_scores["nb_c"])) + "," + str(max(notears_sparse_dict_scores["nb_c"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["nb_c"]), 2)) + " {" + str(min(notears_dimension_dict_scores["nb_c"])) + "," + str(max(notears_dimension_dict_scores["nb_c"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Support Vector Machines (sigmoid)', 'Linear': str(round(mean(notears_linear_dict_scores["svm"]),2))+" {"+str(min(notears_linear_dict_scores["svm"]))+","+str(max(notears_linear_dict_scores["svm"]))+"}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["svm"]),2))+" {"+str(min(notears_nonlinear_dict_scores["svm"]))+","+str(max(notears_nonlinear_dict_scores["svm"]))+"}", 'Sparsity': str(round(mean(notears_sparse_dict_scores["svm"]),2))+" {"+str(min(notears_sparse_dict_scores["svm"]))+","+str(max(notears_sparse_dict_scores["svm"]))+"}", 'Dimensionality': str(round(mean(notears_dimension_dict_scores["svm"]),2))+" {"+str(min(notears_dimension_dict_scores["svm"]))+","+str(max(notears_dimension_dict_scores["svm"]))+"}"})
        #thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(notears_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_linear_dict_scores["svm_l"])) + "," + str(max(notears_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["svm_l"])) + "," + str(max(notears_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_sparse_dict_scores["svm_l"])) + "," + str(max(notears_sparse_dict_scores["svm_l"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_dimension_dict_scores["svm_l"])) + "," + str(max(notears_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Support Vector Machines (poly)','Linear': str(round(mean(notears_linear_dict_scores["svm_po"]), 2)) + " {" + str(min(notears_linear_dict_scores["svm_po"])) + "," + str(max(notears_linear_dict_scores["svm_po"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["svm_po"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["svm_po"])) + "," + str(max(notears_nonlinear_dict_scores["svm_po"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["svm_po"]), 2)) + " {" + str(min(notears_sparse_dict_scores["svm_po"])) + "," + str(max(notears_sparse_dict_scores["svm_po"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["svm_po"]), 2)) + " {" + str(min(notears_dimension_dict_scores["svm_po"])) + "," + str(max(notears_dimension_dict_scores["svm_po"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Support Vector Machines (rbf)','Linear': str(round(mean(notears_linear_dict_scores["svm_r"]), 2)) + " {" + str(min(notears_linear_dict_scores["svm_r"])) + "," + str(max(notears_linear_dict_scores["svm_r"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["svm_r"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["svm_r"])) + "," + str(max(notears_nonlinear_dict_scores["svm_r"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["svm_r"]), 2)) + " {" + str(min(notears_sparse_dict_scores["svm_r"])) + "," + str(max(notears_sparse_dict_scores["svm_r"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["svm_r"]), 2)) + " {" + str(min(notears_dimension_dict_scores["svm_r"])) + "," + str(max(notears_dimension_dict_scores["svm_r"])) + "}"})
        #thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(notears_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_linear_dict_scores["svm_pr"])) + "," + str(max(notears_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["svm_pr"])) + "," + str(max(notears_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_sparse_dict_scores["svm_pr"])) + "," + str(max(notears_sparse_dict_scores["svm_pr"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_dimension_dict_scores["svm_pr"])) + "," + str(max(notears_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'K Nearest Neighbor (uniform)', 'Linear': str(round(mean(notears_linear_dict_scores["knn"]),2))+" {"+str(min(notears_linear_dict_scores["knn"]))+","+str(max(notears_linear_dict_scores["knn"]))+"}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["knn"]),2))+" {"+str(min(notears_nonlinear_dict_scores["knn"]))+","+str(max(notears_nonlinear_dict_scores["knn"]))+"}", 'Sparsity': str(round(mean(notears_sparse_dict_scores["knn"]),2))+" {"+str(min(notears_sparse_dict_scores["knn"]))+","+str(max(notears_sparse_dict_scores["knn"]))+"}", 'Dimensionality': str(round(mean(notears_dimension_dict_scores["knn"]),2))+" {"+str(min(notears_dimension_dict_scores["knn"]))+","+str(max(notears_dimension_dict_scores["knn"]))+"}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(round(mean(notears_linear_dict_scores["knn_d"]), 2)) + " {" + str(min(notears_linear_dict_scores["knn_d"])) + "," + str(max(notears_linear_dict_scores["knn_d"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["knn_d"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["knn_d"])) + "," + str(max(notears_nonlinear_dict_scores["knn_d"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["knn_d"]), 2)) + " {" + str(min(notears_sparse_dict_scores["knn_d"])) + "," + str(max(notears_sparse_dict_scores["knn_d"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["knn_d"]), 2)) + " {" + str(min(notears_dimension_dict_scores["knn_d"])) + "," + str(max(notears_dimension_dict_scores["knn_d"])) + "}"})

        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Decision Tree (gini)','Linear': str(round(mean(notears_poisson_linear_dict_scores["dt"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["dt"])) + "," + str(max(notears_poisson_linear_dict_scores["dt"])) + "}",'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["dt"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["dt"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["dt"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["dt"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["dt"])) + "," + str(max(notears_poisson_sparse_dict_scores["dt"])) + "}",'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["dt"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["dt"])) + "," + str(max(notears_poisson_dimension_dict_scores["dt"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Decision Tree (entropy)','Linear': str(round(mean(notears_poisson_linear_dict_scores["dt_e"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["dt_e"])) + "," + str(max(notears_poisson_linear_dict_scores["dt_e"])) + "}",'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["dt_e"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["dt_e"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["dt_e"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["dt_e"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["dt_e"])) + "," + str(max(notears_poisson_sparse_dict_scores["dt_e"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["dt_e"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["dt_e"])) + "," + str(max(notears_poisson_dimension_dict_scores["dt_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Random Forest (gini)','Linear': str(round(mean(notears_poisson_linear_dict_scores["rf"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["rf"])) + "," + str(max(notears_poisson_linear_dict_scores["rf"])) + "}",'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["rf"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["rf"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["rf"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["rf"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["rf"])) + "," + str(max(notears_poisson_sparse_dict_scores["rf"])) + "}",'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["rf"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["rf"])) + "," + str(max(notears_poisson_dimension_dict_scores["rf"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Random Forest (entropy)','Linear': str(round(mean(notears_poisson_linear_dict_scores["rf_e"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["rf_e"])) + "," + str(max(notears_poisson_linear_dict_scores["rf_e"])) + "}",'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["rf_e"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["rf_e"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["rf_e"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["rf_e"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["rf_e"])) + "," + str(max(notears_poisson_sparse_dict_scores["rf_e"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["rf_e"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["rf_e"])) + "," + str(max(notears_poisson_dimension_dict_scores["rf_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(round(mean(notears_poisson_linear_dict_scores["lr"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["lr"])) + "," + str(max(notears_poisson_linear_dict_scores["lr"])) + "}",'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["lr"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["lr"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["lr"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["lr"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["lr"])) + "," + str(max(notears_poisson_sparse_dict_scores["lr"])) + "}",'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["lr"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["lr"])) + "," + str(max(notears_poisson_dimension_dict_scores["lr"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Logistic Regression (l1)','Linear': str(round(mean(notears_poisson_linear_dict_scores["lr_l1"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["lr_l1"])) + "," + str(max(notears_poisson_linear_dict_scores["lr_l1"])) + "}",'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["lr_l1"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["lr_l1"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["lr_l1"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["lr_l1"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["lr_l1"])) + "," + str(max(notears_poisson_sparse_dict_scores["lr_l1"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["lr_l1"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["lr_l1"])) + "," + str(max(notears_poisson_dimension_dict_scores["lr_l1"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Logistic Regression (l2)','Linear': str(round(mean(notears_poisson_linear_dict_scores["lr_l2"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["lr_l2"])) + "," + str(max(notears_poisson_linear_dict_scores["lr_l2"])) + "}",'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["lr_l2"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["lr_l2"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["lr_l2"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["lr_l2"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["lr_l2"])) + "," + str(max(notears_poisson_sparse_dict_scores["lr_l2"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["lr_l2"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["lr_l2"])) + "," + str(max(notears_poisson_dimension_dict_scores["lr_l2"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(round(mean(notears_poisson_linear_dict_scores["lr_e"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["lr_e"])) + "," + str(max(notears_poisson_linear_dict_scores["lr_e"])) + "}",'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["lr_e"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["lr_e"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["lr_e"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["lr_e"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["lr_e"])) + "," + str(max(notears_poisson_sparse_dict_scores["lr_e"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["lr_e"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["lr_e"])) + "," + str(max(notears_poisson_dimension_dict_scores["lr_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(round(mean(notears_poisson_linear_dict_scores["nb"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["nb"])) + "," + str(max(notears_poisson_linear_dict_scores["nb"])) + "}",'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["nb"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["nb"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["nb"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["nb"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["nb"])) + "," + str(max(notears_poisson_sparse_dict_scores["nb"])) + "}",'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["nb"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["nb"])) + "," + str(max(notears_poisson_dimension_dict_scores["nb"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(round(mean(notears_poisson_linear_dict_scores["nb_m"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["nb_m"])) + "," + str(max(notears_poisson_linear_dict_scores["nb_m"])) + "}",'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["nb_m"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["nb_m"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["nb_m"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["nb_m"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["nb_m"])) + "," + str(max(notears_poisson_sparse_dict_scores["nb_m"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["nb_m"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["nb_m"])) + "," + str(max(notears_poisson_dimension_dict_scores["nb_m"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(round(mean(notears_poisson_linear_dict_scores["nb_g"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["nb_g"])) + "," + str(max(notears_poisson_linear_dict_scores["nb_g"])) + "}",'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["nb_g"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["nb_g"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["nb_g"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["nb_g"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["nb_g"])) + "," + str(max(notears_poisson_sparse_dict_scores["nb_g"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["nb_g"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["nb_g"])) + "," + str(max(notears_poisson_dimension_dict_scores["nb_g"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Naive Bayes (Complement)','Linear': str(round(mean(notears_poisson_linear_dict_scores["nb_c"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["nb_c"])) + "," + str(max(notears_poisson_linear_dict_scores["nb_c"])) + "}",'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["nb_c"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["nb_c"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["nb_c"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["nb_c"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["nb_c"])) + "," + str(max(notears_poisson_sparse_dict_scores["nb_c"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["nb_c"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["nb_c"])) + "," + str(max(notears_poisson_dimension_dict_scores["nb_c"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(round(mean(notears_poisson_linear_dict_scores["svm"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["svm"])) + "," + str(max(notears_poisson_linear_dict_scores["svm"])) + "}",'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["svm"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["svm"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["svm"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["svm"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["svm"])) + "," + str(max(notears_poisson_sparse_dict_scores["svm"])) + "}",'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["svm"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["svm"])) + "," + str(max(notears_poisson_dimension_dict_scores["svm"])) + "}"})
        #thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(notears_poisson_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["svm_l"])) + "," + str(max(notears_poisson_linear_dict_scores["svm_l"])) + "}", 'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["svm_l"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["svm_l"])) + "," + str(max(notears_poisson_sparse_dict_scores["svm_l"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["svm_l"])) + "," + str(max(notears_poisson_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Support Vector Machines (poly)','Linear': str(round(mean(notears_poisson_linear_dict_scores["svm_po"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["svm_po"])) + "," + str(max(notears_poisson_linear_dict_scores["svm_po"])) + "}", 'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["svm_po"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["svm_po"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["svm_po"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["svm_po"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["svm_po"])) + "," + str(max(notears_poisson_sparse_dict_scores["svm_po"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["svm_po"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["svm_po"])) + "," + str(max(notears_poisson_dimension_dict_scores["svm_po"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Support Vector Machines (rbf)','Linear': str(round(mean(notears_poisson_linear_dict_scores["svm_r"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["svm_r"])) + "," + str(max(notears_poisson_linear_dict_scores["svm_r"])) + "}", 'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["svm_r"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["svm_r"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["svm_r"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["svm_r"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["svm_r"])) + "," + str(max(notears_poisson_sparse_dict_scores["svm_r"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["svm_r"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["svm_r"])) + "," + str(max(notears_poisson_dimension_dict_scores["svm_r"])) + "}"})
        #thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(notears_poisson_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["svm_pr"])) + "," + str(max(notears_poisson_linear_dict_scores["svm_pr"])) + "}", 'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["svm_pr"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["svm_pr"])) + "," + str(max(notears_poisson_sparse_dict_scores["svm_pr"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["svm_pr"])) + "," + str(max(notears_poisson_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(round(mean(notears_poisson_linear_dict_scores["knn"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["knn"])) + "," + str(max(notears_poisson_linear_dict_scores["knn"])) + "}",'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["knn"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["knn"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["knn"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["knn"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["knn"])) + "," + str(max(notears_poisson_sparse_dict_scores["knn"])) + "}",'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["knn"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["knn"])) + "," + str(max(notears_poisson_dimension_dict_scores["knn"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(round(mean(notears_poisson_linear_dict_scores["knn_d"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["knn_d"])) + "," + str(max(notears_poisson_linear_dict_scores["knn_d"])) + "}", 'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["knn_d"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["knn_d"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["knn_d"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["knn_d"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["knn_d"])) + "," + str(max(notears_poisson_sparse_dict_scores["knn_d"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["knn_d"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["knn_d"])) + "," + str(max(notears_poisson_dimension_dict_scores["knn_d"])) + "}"})

        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Decision Tree (gini)', 'Linear': str(round(mean(bnlearn_linear_dict_scores["dt"]),2))+" {"+str(min(bnlearn_linear_dict_scores["dt"]))+","+str(max(bnlearn_linear_dict_scores["dt"]))+"}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["dt"]),2))+" {"+str(min(bnlearn_nonlinear_dict_scores["dt"]))+","+str(max(bnlearn_nonlinear_dict_scores["dt"]))+"}", 'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["dt"]),2))+" {"+str(min(bnlearn_sparse_dict_scores["dt"]))+","+str(max(bnlearn_sparse_dict_scores["dt"]))+"}", 'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["dt"]),2))+" {"+str(min(bnlearn_dimension_dict_scores["dt"]))+","+str(max(bnlearn_dimension_dict_scores["dt"]))+"}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Decision Tree (entropy)','Linear': str(round(mean(bnlearn_linear_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["dt_e"])) + "," + str(max(bnlearn_linear_dict_scores["dt_e"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["dt_e"])) + "," + str(max(bnlearn_nonlinear_dict_scores["dt_e"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["dt_e"])) + "," + str(max(bnlearn_sparse_dict_scores["dt_e"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["dt_e"])) + "," + str(max(bnlearn_dimension_dict_scores["dt_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Random Forest (gini)', 'Linear': str(round(mean(bnlearn_linear_dict_scores["rf"]),2))+" {"+str(min(bnlearn_linear_dict_scores["rf"]))+","+str(max(bnlearn_linear_dict_scores["rf"]))+"}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["rf"]),2))+" {"+str(min(bnlearn_nonlinear_dict_scores["rf"]))+","+str(max(bnlearn_nonlinear_dict_scores["rf"]))+"}", 'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["rf"]),2))+" {"+str(min(bnlearn_sparse_dict_scores["rf"]))+","+str(max(bnlearn_sparse_dict_scores["rf"]))+"}", 'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["rf"]),2))+" {"+str(min(bnlearn_dimension_dict_scores["rf"]))+","+str(max(bnlearn_dimension_dict_scores["rf"]))+"}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Random Forest (entropy)','Linear': str(round(mean(bnlearn_linear_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["rf_e"])) + "," + str(max(bnlearn_linear_dict_scores["rf_e"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["rf_e"])) + "," + str(max(bnlearn_nonlinear_dict_scores["rf_e"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["rf_e"])) + "," + str(max(bnlearn_sparse_dict_scores["rf_e"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["rf_e"])) + "," + str(max(bnlearn_dimension_dict_scores["rf_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Logistic Regression (penalty-none)', 'Linear': str(round(mean(bnlearn_linear_dict_scores["lr"]),2))+" {"+str(min(bnlearn_linear_dict_scores["lr"]))+","+str(max(bnlearn_linear_dict_scores["lr"]))+"}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["lr"]),2))+" {"+str(min(bnlearn_nonlinear_dict_scores["lr"]))+","+str(max(bnlearn_nonlinear_dict_scores["lr"]))+"}", 'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["lr"]),2))+" {"+str(min(bnlearn_sparse_dict_scores["lr"]))+","+str(max(bnlearn_sparse_dict_scores["lr"]))+"}", 'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["lr"]),2))+" {"+str(min(bnlearn_dimension_dict_scores["lr"]))+","+str(max(bnlearn_dimension_dict_scores["lr"]))+"}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Logistic Regression (l1)','Linear': str(round(mean(bnlearn_linear_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["lr_l1"])) + "," + str(max(bnlearn_linear_dict_scores["lr_l1"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["lr_l1"])) + "," + str(max(bnlearn_nonlinear_dict_scores["lr_l1"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["lr_l1"])) + "," + str(max(bnlearn_sparse_dict_scores["lr_l1"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["lr_l1"])) + "," + str(max(bnlearn_dimension_dict_scores["lr_l1"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Logistic Regression (l2)','Linear': str(round(mean(bnlearn_linear_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["lr_l2"])) + "," + str(max(bnlearn_linear_dict_scores["lr_l2"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["lr_l2"])) + "," + str(max(bnlearn_nonlinear_dict_scores["lr_l2"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["lr_l2"])) + "," + str(max(bnlearn_sparse_dict_scores["lr_l2"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["lr_l2"])) + "," + str(max(bnlearn_dimension_dict_scores["lr_l2"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(round(mean(bnlearn_linear_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["lr_e"])) + "," + str(max(bnlearn_linear_dict_scores["lr_e"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["lr_e"])) + "," + str(max(bnlearn_nonlinear_dict_scores["lr_e"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["lr_e"])) + "," + str(max(bnlearn_sparse_dict_scores["lr_e"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["lr_e"])) + "," + str(max(bnlearn_dimension_dict_scores["lr_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Naive Bayes (Bernoulli)', 'Linear': str(round(mean(bnlearn_linear_dict_scores["nb"]),2))+" {"+str(min(bnlearn_linear_dict_scores["nb"]))+","+str(max(bnlearn_linear_dict_scores["nb"]))+"}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["nb"]),2))+" {"+str(min(bnlearn_nonlinear_dict_scores["nb"]))+","+str(max(bnlearn_nonlinear_dict_scores["nb"]))+"}", 'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["nb"]),2))+" {"+str(min(bnlearn_sparse_dict_scores["nb"]))+","+str(max(bnlearn_sparse_dict_scores["nb"]))+"}", 'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["nb"]),2))+" {"+str(min(bnlearn_dimension_dict_scores["nb"]))+","+str(max(bnlearn_dimension_dict_scores["nb"]))+"}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(round(mean(bnlearn_linear_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["nb_m"])) + "," + str(max(bnlearn_linear_dict_scores["nb_m"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["nb_m"])) + "," + str(max(bnlearn_nonlinear_dict_scores["nb_m"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["nb_m"])) + "," + str(max(bnlearn_sparse_dict_scores["nb_m"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["nb_m"])) + "," + str(max(bnlearn_dimension_dict_scores["nb_m"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(round(mean(bnlearn_linear_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["nb_g"])) + "," + str(max(bnlearn_linear_dict_scores["nb_g"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["nb_g"])) + "," + str(max(bnlearn_nonlinear_dict_scores["nb_g"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["nb_g"])) + "," + str(max(bnlearn_sparse_dict_scores["nb_g"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["nb_g"])) + "," + str(max(bnlearn_dimension_dict_scores["nb_g"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Naive Bayes (Complement)','Linear': str(round(mean(bnlearn_linear_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["nb_c"])) + "," + str(max(bnlearn_linear_dict_scores["nb_c"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["nb_c"])) + "," + str(max(bnlearn_nonlinear_dict_scores["nb_c"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["nb_c"])) + "," + str(max(bnlearn_sparse_dict_scores["nb_c"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["nb_c"])) + "," + str(max(bnlearn_dimension_dict_scores["nb_c"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Support Vector Machines (sigmoid)', 'Linear': str(round(mean(bnlearn_linear_dict_scores["svm"]),2))+" {"+str(min(bnlearn_linear_dict_scores["svm"]))+","+str(max(bnlearn_linear_dict_scores["svm"]))+"}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["svm"]),2))+" {"+str(min(bnlearn_nonlinear_dict_scores["svm"]))+","+str(max(bnlearn_nonlinear_dict_scores["svm"]))+"}", 'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["svm"]),2))+" {"+str(min(bnlearn_sparse_dict_scores["svm"]))+","+str(max(bnlearn_sparse_dict_scores["svm"]))+"}", 'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["svm"]),2))+" {"+str(min(bnlearn_dimension_dict_scores["svm"]))+","+str(max(bnlearn_dimension_dict_scores["svm"]))+"}"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["svm_l"])) + "," + str(max(bnlearn_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["svm_l"])) + "," + str(max(bnlearn_sparse_dict_scores["svm_l"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["svm_l"])) + "," + str(max(bnlearn_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Support Vector Machines (poly)','Linear': str(round(mean(bnlearn_linear_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["svm_po"])) + "," + str(max(bnlearn_linear_dict_scores["svm_po"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["svm_po"])) + "," + str(max(bnlearn_nonlinear_dict_scores["svm_po"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["svm_po"])) + "," + str(max(bnlearn_sparse_dict_scores["svm_po"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["svm_po"])) + "," + str(max(bnlearn_dimension_dict_scores["svm_po"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Support Vector Machines (rbf)','Linear': str(round(mean(bnlearn_linear_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["svm_r"])) + "," + str(max(bnlearn_linear_dict_scores["svm_r"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["svm_r"])) + "," + str(max(bnlearn_nonlinear_dict_scores["svm_r"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["svm_r"])) + "," + str(max(bnlearn_sparse_dict_scores["svm_r"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["svm_r"])) + "," + str(max(bnlearn_dimension_dict_scores["svm_r"])) + "}"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["svm_pr"])) + "," + str(max(bnlearn_sparse_dict_scores["svm_pr"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["svm_pr"])) + "," + str(max(bnlearn_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'K Nearest Neighbor (uniform)', 'Linear': str(round(mean(bnlearn_linear_dict_scores["knn"]),2))+" {"+str(min(bnlearn_linear_dict_scores["knn"]))+","+str(max(bnlearn_linear_dict_scores["knn"]))+"}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["knn"]),2))+" {"+str(min(bnlearn_nonlinear_dict_scores["knn"]))+","+str(max(bnlearn_nonlinear_dict_scores["knn"]))+"}", 'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["knn"]),2))+" {"+str(min(bnlearn_sparse_dict_scores["knn"]))+","+str(max(bnlearn_sparse_dict_scores["knn"]))+"}", 'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["knn"]),2))+" {"+str(min(bnlearn_dimension_dict_scores["knn"]))+","+str(max(bnlearn_dimension_dict_scores["knn"]))+"}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(round(mean(bnlearn_linear_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["knn_d"])) + "," + str(max(bnlearn_linear_dict_scores["knn_d"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["knn_d"])) + "," + str(max(bnlearn_nonlinear_dict_scores["knn_d"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["knn_d"])) + "," + str(max(bnlearn_sparse_dict_scores["knn_d"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["knn_d"])) + "," + str(max(bnlearn_dimension_dict_scores["knn_d"])) + "}"})

        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Decision Tree (gini)', 'Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["dt"]),2))+" {"+str(min(bnlearn_tabu_linear_dict_scores["dt"]))+","+str(max(bnlearn_tabu_linear_dict_scores["dt"]))+"}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["dt"]),2))+" {"+str(min(bnlearn_tabu_nonlinear_dict_scores["dt"]))+","+str(max(bnlearn_tabu_nonlinear_dict_scores["dt"]))+"}", 'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["dt"]),2))+" {"+str(min(bnlearn_tabu_sparse_dict_scores["dt"]))+","+str(max(bnlearn_tabu_sparse_dict_scores["dt"]))+"}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["dt"]),2))+" {"+str(min(bnlearn_tabu_dimension_dict_scores["dt"]))+","+str(max(bnlearn_tabu_dimension_dict_scores["dt"]))+"}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Decision Tree (entropy)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["dt_e"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["dt_e"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["dt_e"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["dt_e"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["dt_e"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["dt_e"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["dt_e"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["dt_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Random Forest (gini)', 'Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["rf"]),2))+" {"+str(min(bnlearn_tabu_linear_dict_scores["rf"]))+","+str(max(bnlearn_tabu_linear_dict_scores["rf"]))+"}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["rf"]),2))+" {"+str(min(bnlearn_tabu_nonlinear_dict_scores["rf"]))+","+str(max(bnlearn_tabu_nonlinear_dict_scores["rf"]))+"}", 'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["rf"]),2))+" {"+str(min(bnlearn_tabu_sparse_dict_scores["rf"]))+","+str(max(bnlearn_tabu_sparse_dict_scores["rf"]))+"}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["rf"]),2))+" {"+str(min(bnlearn_tabu_dimension_dict_scores["rf"]))+","+str(max(bnlearn_tabu_dimension_dict_scores["rf"]))+"}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Random Forest (entropy)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["rf_e"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["rf_e"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["rf_e"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["rf_e"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["rf_e"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["rf_e"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["rf_e"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["rf_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Logistic Regression (penalty-none)', 'Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["lr"]),2))+" {"+str(min(bnlearn_tabu_linear_dict_scores["lr"]))+","+str(max(bnlearn_tabu_linear_dict_scores["lr"]))+"}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["lr"]),2))+" {"+str(min(bnlearn_tabu_nonlinear_dict_scores["lr"]))+","+str(max(bnlearn_tabu_nonlinear_dict_scores["lr"]))+"}", 'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["lr"]),2))+" {"+str(min(bnlearn_tabu_sparse_dict_scores["lr"]))+","+str(max(bnlearn_tabu_sparse_dict_scores["lr"]))+"}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["lr"]),2))+" {"+str(min(bnlearn_tabu_dimension_dict_scores["lr"]))+","+str(max(bnlearn_tabu_dimension_dict_scores["lr"]))+"}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Logistic Regression (l1)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["lr_l1"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["lr_l1"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["lr_l1"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["lr_l1"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["lr_l1"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["lr_l1"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["lr_l1"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["lr_l1"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Logistic Regression (l2)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["lr_l2"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["lr_l2"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["lr_l2"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["lr_l2"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["lr_l2"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["lr_l2"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["lr_l2"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["lr_l2"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["lr_e"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["lr_e"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["lr_e"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["lr_e"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["lr_e"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["lr_e"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["lr_e"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["lr_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Naive Bayes (Bernoulli)', 'Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["nb"]),2))+" {"+str(min(bnlearn_tabu_linear_dict_scores["nb"]))+","+str(max(bnlearn_tabu_linear_dict_scores["nb"]))+"}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["nb"]),2))+" {"+str(min(bnlearn_tabu_nonlinear_dict_scores["nb"]))+","+str(max(bnlearn_tabu_nonlinear_dict_scores["nb"]))+"}", 'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["nb"]),2))+" {"+str(min(bnlearn_tabu_sparse_dict_scores["nb"]))+","+str(max(bnlearn_tabu_sparse_dict_scores["nb"]))+"}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["nb"]),2))+" {"+str(min(bnlearn_tabu_dimension_dict_scores["nb"]))+","+str(max(bnlearn_tabu_dimension_dict_scores["nb"]))+"}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["nb_m"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["nb_m"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["nb_m"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["nb_m"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["nb_m"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["nb_m"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["nb_m"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["nb_m"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["nb_g"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["nb_g"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["nb_g"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["nb_g"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["nb_g"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["nb_g"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["nb_g"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["nb_g"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Naive Bayes (Complement)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["nb_c"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["nb_c"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["nb_c"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["nb_c"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["nb_c"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["nb_c"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["nb_c"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["nb_c"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Support Vector Machines (sigmoid)', 'Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["svm"]),2))+" {"+str(min(bnlearn_tabu_linear_dict_scores["svm"]))+","+str(max(bnlearn_tabu_linear_dict_scores["svm"]))+"}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["svm"]),2))+" {"+str(min(bnlearn_tabu_nonlinear_dict_scores["svm"]))+","+str(max(bnlearn_tabu_nonlinear_dict_scores["svm"]))+"}", 'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["svm"]),2))+" {"+str(min(bnlearn_tabu_sparse_dict_scores["svm"]))+","+str(max(bnlearn_tabu_sparse_dict_scores["svm"]))+"}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["svm"]),2))+" {"+str(min(bnlearn_tabu_dimension_dict_scores["svm"]))+","+str(max(bnlearn_tabu_dimension_dict_scores["svm"]))+"}"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["svm_l"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["svm_l"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["svm_l"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["svm_l"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Support Vector Machines (poly)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["svm_po"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["svm_po"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["svm_po"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["svm_po"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["svm_po"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["svm_po"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["svm_po"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["svm_po"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Support Vector Machines (rbf)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["svm_r"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["svm_r"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["svm_r"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["svm_r"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["svm_r"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["svm_r"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["svm_r"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["svm_r"])) + "}"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["svm_pr"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["svm_pr"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["svm_pr"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'K Nearest Neighbor (uniform)', 'Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["knn"]),2))+" {"+str(min(bnlearn_tabu_linear_dict_scores["knn"]))+","+str(max(bnlearn_tabu_linear_dict_scores["knn"]))+"}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["knn"]),2))+" {"+str(min(bnlearn_tabu_nonlinear_dict_scores["knn"]))+","+str(max(bnlearn_tabu_nonlinear_dict_scores["knn"]))+"}", 'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["knn"]),2))+" {"+str(min(bnlearn_tabu_sparse_dict_scores["knn"]))+","+str(max(bnlearn_tabu_sparse_dict_scores["knn"]))+"}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["knn"]),2))+" {"+str(min(bnlearn_tabu_dimension_dict_scores["knn"]))+","+str(max(bnlearn_tabu_dimension_dict_scores["knn"]))+"}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["knn_d"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["knn_d"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["knn_d"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["knn_d"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["knn_d"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["knn_d"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["knn_d"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["knn_d"])) + "}"})

        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Decision Tree (gini)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["dt"])) + "," + str(max(bnlearn_pc_linear_dict_scores["dt"])) + "}",'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Decision Tree (entropy)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["dt_e"])) + "," + str(max(bnlearn_pc_linear_dict_scores["dt_e"])) + "}",'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Random Forest (gini)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["rf"])) + "," + str(max(bnlearn_pc_linear_dict_scores["rf"])) + "}",'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Random Forest (entropy)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["rf_e"])) + "," + str(max(bnlearn_pc_linear_dict_scores["rf_e"])) + "}",'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["lr"])) + "," + str(max(bnlearn_pc_linear_dict_scores["lr"])) + "}",'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Logistic Regression (l1)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["lr_l1"])) + "," + str(max(bnlearn_pc_linear_dict_scores["lr_l1"])) + "}",'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Logistic Regression (l2)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["lr_l2"])) + "," + str(max(bnlearn_pc_linear_dict_scores["lr_l2"])) + "}",'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["lr_e"])) + "," + str(max(bnlearn_pc_linear_dict_scores["lr_e"])) + "}",'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["nb"])) + "," + str(max(bnlearn_pc_linear_dict_scores["nb"])) + "}",'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["nb_m"])) + "," + str(max(bnlearn_pc_linear_dict_scores["nb_m"])) + "}",'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["nb_g"])) + "," + str(max(bnlearn_pc_linear_dict_scores["nb_g"])) + "}",'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Naive Bayes (Complement)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["nb_c"])) + "," + str(max(bnlearn_pc_linear_dict_scores["nb_c"])) + "}",'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["svm"])) + "," + str(max(bnlearn_pc_linear_dict_scores["svm"])) + "}",'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_pc_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(bnlearn_pc_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_pc_nonlinear_dict_scores["svm_l"])) + "," + str(max(bnlearn_pc_nonlinear_dict_scores["svm_l"])) + "}", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Support Vector Machines (poly)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["svm_po"])) + "," + str(max(bnlearn_pc_linear_dict_scores["svm_po"])) + "}",'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Support Vector Machines (rbf)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["svm_r"])) + "," + str(max(bnlearn_pc_linear_dict_scores["svm_r"])) + "}",'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_pc_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(bnlearn_pc_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_pc_nonlinear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_pc_nonlinear_dict_scores["svm_pr"])) + "}", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["knn"])) + "," + str(max(bnlearn_pc_linear_dict_scores["knn"])) + "}",'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["knn_d"])) + "," + str(max(bnlearn_pc_linear_dict_scores["knn_d"])) + "}",'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})

        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Decision Tree (gini)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["dt"])) + "," + str(max(bnlearn_gs_linear_dict_scores["dt"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Decision Tree (entropy)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["dt_e"])) + "," + str(max(bnlearn_gs_linear_dict_scores["dt_e"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Random Forest (gini)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["rf"])) + "," + str(max(bnlearn_gs_linear_dict_scores["rf"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Random Forest (entropy)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["rf_e"])) + "," + str(max(bnlearn_gs_linear_dict_scores["rf_e"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["lr"])) + "," + str(max(bnlearn_gs_linear_dict_scores["lr"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Logistic Regression (l1)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["lr_l1"])) + "," + str(max(bnlearn_gs_linear_dict_scores["lr_l1"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Logistic Regression (l2)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["lr_l2"])) + "," + str(max(bnlearn_gs_linear_dict_scores["lr_l2"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["lr_e"])) + "," + str(max(bnlearn_gs_linear_dict_scores["lr_e"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["nb"])) + "," + str(max(bnlearn_gs_linear_dict_scores["nb"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["nb_m"])) + "," + str(max(bnlearn_gs_linear_dict_scores["nb_m"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["nb_g"])) + "," + str(max(bnlearn_gs_linear_dict_scores["nb_g"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Naive Bayes (Complement)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["nb_c"])) + "," + str(max(bnlearn_gs_linear_dict_scores["nb_c"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["svm"])) + "," + str(max(bnlearn_gs_linear_dict_scores["svm"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_gs_linear_dict_scores["svm_l"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Support Vector Machines (poly)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["svm_po"])) + "," + str(max(bnlearn_gs_linear_dict_scores["svm_po"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Support Vector Machines (rbf)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["svm_r"])) + "," + str(max(bnlearn_gs_linear_dict_scores["svm_r"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_gs_linear_dict_scores["svm_pr"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["knn"])) + "," + str(max(bnlearn_gs_linear_dict_scores["knn"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["knn_d"])) + "," + str(max(bnlearn_gs_linear_dict_scores["knn_d"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})

        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Decision Tree (gini)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["dt"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["dt"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Decision Tree (entropy)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["dt_e"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["dt_e"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Random Forest (gini)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["rf"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["rf"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Random Forest (entropy)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["rf_e"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["rf_e"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["lr"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["lr"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Logistic Regression (l1)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["lr_l1"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["lr_l1"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Logistic Regression (l2)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["lr_l2"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["lr_l2"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["lr_e"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["lr_e"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["nb"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["nb"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["nb_m"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["nb_m"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["nb_g"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["nb_g"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Naive Bayes (Complement)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["nb_c"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["nb_c"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["svm"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["svm"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["svm_l"])) + "}", 'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Support Vector Machines (poly)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["svm_po"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["svm_po"])) + "}", 'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Support Vector Machines (rbf)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["svm_r"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["svm_r"])) + "}", 'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["svm_pr"])) + "}", 'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["knn"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["knn"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (IAMB)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(round(mean(bnlearn_iamb_linear_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_iamb_linear_dict_scores["knn_d"])) + "," + str(max(bnlearn_iamb_linear_dict_scores["knn_d"])) + "}", 'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})

        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Decision Tree (gini)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["dt"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["dt"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["dt"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["dt"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["dt"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["dt"])) + "}",'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["dt"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["dt"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Decision Tree (entropy)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["dt_e"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["dt_e"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["dt_e"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["dt_e"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["dt_e"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["dt_e"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["dt_e"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["dt_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Random Forest (gini)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["rf"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["rf"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["rf"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["rf"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["rf"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["rf"])) + "}",'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["rf"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["rf"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Random Forest (entropy)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["rf_e"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["rf_e"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["rf_e"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["rf_e"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["rf_e"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["rf_e"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["rf_e"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["rf_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["lr"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["lr"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["lr"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["lr"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["lr"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["lr"])) + "}",'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["lr"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["lr"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Logistic Regression (l1)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["lr_l1"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["lr_l1"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["lr_l1"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["lr_l1"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["lr_l1"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["lr_l1"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["lr_l1"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["lr_l1"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Logistic Regression (l2)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["lr_l2"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["lr_l2"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["lr_l2"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["lr_l2"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["lr_l2"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["lr_l2"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["lr_l2"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["lr_l2"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["lr_e"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["lr_e"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["lr_e"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["lr_e"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["lr_e"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["lr_e"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["lr_e"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["lr_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["nb"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["nb"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["nb"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["nb"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["nb"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["nb"])) + "}",'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["nb"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["nb"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["nb_m"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["nb_m"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["nb_m"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["nb_m"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["nb_m"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["nb_m"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["nb_m"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["nb_m"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["nb_g"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["nb_g"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["nb_g"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["nb_g"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["nb_g"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["nb_g"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["nb_g"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["nb_g"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Naive Bayes (Complement)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["nb_c"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["nb_c"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["nb_c"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["nb_c"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["nb_c"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["nb_c"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["nb_c"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["nb_c"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["svm"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["svm"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["svm"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["svm"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["svm"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["svm"])) + "}",'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["svm"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["svm"])) + "}"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["svm_l"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["svm_l"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["svm_l"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["svm_l"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Support Vector Machines (poly)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["svm_po"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["svm_po"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["svm_po"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["svm_po"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["svm_po"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["svm_po"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["svm_po"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["svm_po"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Support Vector Machines (rbf)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["svm_r"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["svm_r"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["svm_r"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["svm_r"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["svm_r"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["svm_r"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["svm_r"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["svm_r"])) + "}"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["svm_pr"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["svm_pr"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["svm_pr"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["knn"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["knn"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["knn"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["knn"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["knn"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["knn"])) + "}",'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["knn"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["knn"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["knn_d"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["knn_d"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["knn_d"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["knn_d"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["knn_d"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["knn_d"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["knn_d"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["knn_d"])) + "}"})

        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Decision Tree (gini)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["dt"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["dt"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["dt"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["dt"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["dt"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["dt"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["dt"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["dt"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Decision Tree (entropy)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["dt_e"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["dt_e"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["dt_e"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["dt_e"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["dt_e"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["dt_e"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["dt_e"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["dt_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Random Forest (gini)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["rf"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["rf"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["rf"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["rf"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["rf"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["rf"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["rf"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["rf"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Random Forest (entropy)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["rf_e"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["rf_e"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["rf_e"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["rf_e"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["rf_e"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["rf_e"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["rf_e"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["rf_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["lr"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["lr"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["lr"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["lr"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["lr"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["lr"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["lr"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["lr"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Logistic Regression (l1)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["lr_l1"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["lr_l1"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["lr_l1"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["lr_l1"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["lr_l1"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["lr_l1"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["lr_l1"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["lr_l1"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Logistic Regression (l2)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["lr_l2"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["lr_l2"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["lr_l2"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["lr_l2"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["lr_l2"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["lr_l2"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["lr_l2"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["lr_l2"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["lr_e"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["lr_e"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["lr_e"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["lr_e"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["lr_e"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["lr_e"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["lr_e"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["lr_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["nb"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["nb"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["nb"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["nb"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["nb"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["nb"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["nb"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["nb"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["nb_m"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["nb_m"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["nb_m"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["nb_m"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["nb_m"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["nb_m"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["nb_m"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["nb_m"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["nb_g"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["nb_g"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["nb_g"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["nb_g"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["nb_g"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["nb_g"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["nb_g"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["nb_g"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Naive Bayes (Complement)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["nb_c"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["nb_c"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["nb_c"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["nb_c"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["nb_c"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["nb_c"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["nb_c"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["nb_c"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["svm"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["svm"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["svm"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["svm"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["svm"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["svm"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["svm"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["svm"])) + "}"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["svm_l"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["svm_l"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["svm_l"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["svm_l"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Support Vector Machines (poly)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["svm_po"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["svm_po"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["svm_po"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["svm_po"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["svm_po"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["svm_po"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["svm_po"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["svm_po"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Support Vector Machines (rbf)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["svm_r"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["svm_r"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["svm_r"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["svm_r"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["svm_r"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["svm_r"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["svm_r"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["svm_r"])) + "}"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["svm_pr"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["svm_pr"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["svm_pr"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["knn"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["knn"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["knn"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["knn"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["knn"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["knn"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["knn"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["knn"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["knn_d"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["knn_d"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["knn_d"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["knn_d"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["knn_d"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["knn_d"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["knn_d"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["knn_d"])) + "}"})

        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Decision Tree (gini)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["dt"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["dt"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["dt"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["dt"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["dt"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["dt"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["dt"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["dt"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Decision Tree (entropy)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["dt_e"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["dt_e"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["dt_e"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["dt_e"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["dt_e"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["dt_e"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["dt_e"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["dt_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Random Forest (gini)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["rf"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["rf"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["rf"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["rf"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["rf"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["rf"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["rf"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["rf"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Random Forest (entropy)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["rf_e"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["rf_e"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["rf_e"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["rf_e"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["rf_e"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["rf_e"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["rf_e"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["rf_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["lr"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["lr"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["lr"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["lr"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["lr"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["lr"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["lr"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["lr"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Logistic Regression (l1)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["lr_l1"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["lr_l1"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["lr_l1"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["lr_l1"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["lr_l1"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["lr_l1"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["lr_l1"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["lr_l1"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Logistic Regression (l2)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["lr_l2"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["lr_l2"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["lr_l2"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["lr_l2"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["lr_l2"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["lr_l2"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["lr_l2"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["lr_l2"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["lr_e"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["lr_e"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["lr_e"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["lr_e"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["lr_e"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["lr_e"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["lr_e"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["lr_e"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["nb"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["nb"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["nb"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["nb"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["nb"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["nb"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["nb"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["nb"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["nb_m"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["nb_m"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["nb_m"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["nb_m"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["nb_m"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["nb_m"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["nb_m"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["nb_m"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["nb_g"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["nb_g"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["nb_g"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["nb_g"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["nb_g"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["nb_g"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["nb_g"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["nb_g"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Naive Bayes (Complement)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["nb_c"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["nb_c"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["nb_c"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["nb_c"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["nb_c"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["nb_c"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["nb_c"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["nb_c"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["svm"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["svm"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["svm"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["svm"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["svm"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["svm"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["svm"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["svm"])) + "}"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["svm_l"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["svm_l"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["svm_l"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["svm_l"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Support Vector Machines (poly)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["svm_po"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["svm_po"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["svm_po"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["svm_po"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["svm_po"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["svm_po"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["svm_po"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["svm_po"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Support Vector Machines (rbf)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["svm_r"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["svm_r"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["svm_r"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["svm_r"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["svm_r"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["svm_r"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["svm_r"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["svm_r"])) + "}"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["svm_pr"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["svm_pr"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["svm_pr"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["knn"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["knn"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["knn"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["knn"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["knn"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["knn"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["knn"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["knn"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["knn_d"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["knn_d"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["knn_d"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["knn_d"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["knn_d"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["knn_d"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["knn_d"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["knn_d"])) + "}"})


write_learned_to_csv()

def write_real_to_csv():
    experiments = ['Model', 'Linear', 'Non-linear', 'Sparsity', 'Dimensionality']
    with open('real_experiments_summary.csv', 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Linear', 'Non-linear', 'Sparsity', 'Dimensionality']
        thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        thewriter.writeheader()
        thewriter.writerow({'Model': 'Decision Tree (gini)','Linear': str(mean(real_linear_dt_scores))+" {"+str(min(real_linear_dt_scores))+","+str(max(real_linear_dt_scores))+"}", 'Non-linear': str(mean(real_nonlinear_dt_scores))+" {"+str(min(real_nonlinear_dt_scores))+","+str(max(real_nonlinear_dt_scores))+"}", 'Sparsity': str(mean(real_sparse_dt_scores))+" {"+str(min(real_sparse_dt_scores))+","+str(max(real_sparse_dt_scores))+"}", 'Dimensionality': str(mean(real_dimension_dt_scores))+" {"+str(min(real_dimension_dt_scores))+","+str(max(real_dimension_dt_scores))+"}"})
        thewriter.writerow({'Model': 'Decision Tree (entropy)', 'Linear': str(mean(real_linear_dt_entropy_scores)) + " {" + str(min(real_linear_dt_entropy_scores)) + "," + str(max(real_linear_dt_entropy_scores)) + "}",'Non-linear': str(mean(real_nonlinear_dt_entropy_scores)) + " {" + str(min(real_nonlinear_dt_entropy_scores)) + "," + str(max(real_nonlinear_dt_entropy_scores)) + "}",'Sparsity': str(mean(real_sparse_dt_entropy_scores)) + " {" + str(min(real_sparse_dt_entropy_scores)) + "," + str(max(real_sparse_dt_entropy_scores)) + "}",'Dimensionality': str(mean(real_dimension_dt_entropy_scores)) + " {" + str(min(real_dimension_dt_entropy_scores)) + "," + str(max(real_dimension_dt_entropy_scores)) + "}"})
        thewriter.writerow({'Model': 'Random Forest (gini)', 'Linear': str(mean(real_linear_rf_scores))+" {"+str(min(real_linear_rf_scores))+","+str(max(real_linear_rf_scores))+"}", 'Non-linear': str(mean(real_nonlinear_rf_scores))+" {"+str(min(real_nonlinear_rf_scores))+","+str(max(real_nonlinear_rf_scores))+"}", 'Sparsity': str(mean(real_sparse_rf_scores))+" {"+str(min(real_sparse_rf_scores))+","+str(max(real_sparse_rf_scores))+"}", 'Dimensionality': str(mean(real_dimension_rf_scores))+" {"+str(min(real_dimension_rf_scores))+","+str(max(real_dimension_rf_scores))+"}"})
        thewriter.writerow({'Model': 'Random Forest (entropy)', 'Linear': str(mean(real_linear_rf_entropy_scores)) + " {" + str(min(real_linear_rf_entropy_scores)) + "," + str(max(real_linear_rf_entropy_scores)) + "}",'Non-linear': str(mean(real_nonlinear_rf_entropy_scores)) + " {" + str(min(real_nonlinear_rf_entropy_scores)) + "," + str(max(real_nonlinear_rf_entropy_scores)) + "}",'Sparsity': str(mean(real_sparse_rf_entropy_scores)) + " {" + str(min(real_sparse_rf_entropy_scores)) + "," + str(max(real_sparse_rf_entropy_scores)) + "}",'Dimensionality': str(mean(real_dimension_rf_entropy_scores)) + " {" + str(min(real_dimension_rf_entropy_scores)) + "," + str(max(real_dimension_rf_entropy_scores)) + "}"})
        thewriter.writerow({'Model': 'Logistic Regression (penalty-none)', 'Linear': str(mean(real_linear_lr_scores))+" {"+str(min(real_linear_lr_scores))+","+str(max(real_linear_lr_scores))+"}", 'Non-linear': str(mean(real_nonlinear_lr_scores))+" {"+str(min(real_nonlinear_lr_scores))+","+str(max(real_nonlinear_lr_scores))+"}", 'Sparsity': str(mean(real_sparse_lr_scores))+" {"+str(min(real_sparse_lr_scores))+","+str(max(real_sparse_lr_scores))+"}", 'Dimensionality': str(mean(real_dimension_lr_scores))+" {"+str(min(real_dimension_lr_scores))+","+str(max(real_dimension_lr_scores))+"}"})
        thewriter.writerow({'Model': 'Logistic Regression (l1)', 'Linear': str(mean(real_linear_lr_l1_scores)) + " {" + str(min(real_linear_lr_l1_scores)) + "," + str(max(real_linear_lr_l1_scores)) + "}",'Non-linear': str(mean(real_nonlinear_lr_l1_scores)) + " {" + str(min(real_nonlinear_lr_l1_scores)) + "," + str(max(real_nonlinear_lr_l1_scores)) + "}",'Sparsity': str(mean(real_sparse_lr_l1_scores)) + " {" + str(min(real_sparse_lr_l1_scores)) + "," + str(max(real_sparse_lr_l1_scores)) + "}",'Dimensionality': str(mean(real_dimension_lr_l1_scores)) + " {" + str(min(real_dimension_lr_l1_scores)) + "," + str(max(real_dimension_lr_l1_scores)) + "}"})
        thewriter.writerow({'Model': 'Logistic Regression (l2)', 'Linear': str(mean(real_linear_lr_l2_scores)) + " {" + str(min(real_linear_lr_l2_scores)) + "," + str(max(real_linear_lr_l2_scores)) + "}",'Non-linear': str(mean(real_nonlinear_lr_l2_scores)) + " {" + str(min(real_nonlinear_lr_l2_scores)) + "," + str(max(real_nonlinear_lr_l2_scores)) + "}",'Sparsity': str(mean(real_sparse_lr_l2_scores)) + " {" + str(min(real_sparse_lr_l2_scores)) + "," + str(max(real_sparse_lr_l2_scores)) + "}",'Dimensionality': str(mean(real_dimension_lr_l2_scores)) + " {" + str(min(real_dimension_lr_l2_scores)) + "," + str(max(real_dimension_lr_l2_scores)) + "}"})
        thewriter.writerow({'Model': 'Logistic Regression (elasticnet)', 'Linear': str(mean(real_linear_lr_elastic_scores)) + " {" + str(min(real_linear_lr_elastic_scores)) + "," + str(max(real_linear_lr_elastic_scores)) + "}",'Non-linear': str(mean(real_nonlinear_lr_elastic_scores)) + " {" + str(min(real_nonlinear_lr_elastic_scores)) + "," + str(max(real_nonlinear_lr_elastic_scores)) + "}",'Sparsity': str(mean(real_sparse_lr_elastic_scores)) + " {" + str(min(real_sparse_lr_elastic_scores)) + "," + str(max(real_sparse_lr_elastic_scores)) + "}",'Dimensionality': str(mean(real_dimension_lr_elastic_scores)) + " {" + str(min(real_dimension_lr_elastic_scores)) + "," + str(max(real_dimension_lr_elastic_scores)) + "}"})
        thewriter.writerow({'Model': 'Naive Bayes (Bernoulli)', 'Linear': str(mean(real_linear_gb_scores))+" {"+str(min(real_linear_gb_scores))+","+str(max(real_linear_gb_scores))+"}",'Non-linear': str(mean(real_nonlinear_gb_scores))+" {"+str(min(real_nonlinear_gb_scores))+","+str(max(real_nonlinear_gb_scores))+"}", 'Sparsity': str(mean(real_sparse_gb_scores))+" {"+str(min(real_sparse_gb_scores))+","+str(max(real_sparse_gb_scores))+"}", 'Dimensionality': str(mean(real_dimension_gb_scores))+" {"+str(min(real_dimension_gb_scores))+","+str(max(real_dimension_gb_scores))+"}"})
        thewriter.writerow({'Model': 'Naive Bayes (Multinomial)', 'Linear': str(mean(real_linear_gb_multi_scores)) + " {" + str(min(real_linear_gb_multi_scores)) + "," + str(max(real_linear_gb_multi_scores)) + "}",'Non-linear': str(mean(real_nonlinear_gb_multi_scores)) + " {" + str(min(real_nonlinear_gb_multi_scores)) + "," + str(max(real_nonlinear_gb_multi_scores)) + "}",'Sparsity': str(mean(real_sparse_gb_multi_scores)) + " {" + str(min(real_sparse_gb_multi_scores)) + "," + str(max(real_sparse_gb_multi_scores)) + "}",'Dimensionality': str(mean(real_dimension_gb_multi_scores)) + " {" + str(min(real_dimension_gb_multi_scores)) + "," + str(max(real_dimension_gb_multi_scores)) + "}"})
        thewriter.writerow({'Model': 'Naive Bayes (Gaussian)','Linear': str(mean(real_linear_gb_gaussian_scores)) + " {" + str(min(real_linear_gb_gaussian_scores)) + "," + str(max(real_linear_gb_gaussian_scores)) + "}",'Non-linear': str(mean(real_nonlinear_gb_gaussian_scores)) + " {" + str(min(real_nonlinear_gb_gaussian_scores)) + "," + str(max(real_nonlinear_gb_gaussian_scores)) + "}",'Sparsity': str(mean(real_sparse_gb_gaussian_scores)) + " {" + str(min(real_sparse_gb_gaussian_scores)) + "," + str(max(real_sparse_gb_gaussian_scores)) + "}",'Dimensionality': str(mean(real_dimension_gb_gaussian_scores)) + " {" + str(min(real_dimension_gb_gaussian_scores)) + "," + str(max(real_dimension_gb_gaussian_scores)) + "}"})
        thewriter.writerow({'Model': 'Naive Bayes (Complement)','Linear': str(mean(real_linear_gb_complement_scores)) + " {" + str(min(real_linear_gb_complement_scores)) + "," + str(max(real_linear_gb_complement_scores)) + "}",'Non-linear': str(mean(real_nonlinear_gb_complement_scores)) + " {" + str(min(real_nonlinear_gb_complement_scores)) + "," + str(max(real_nonlinear_gb_complement_scores)) + "}",'Sparsity': str(mean(real_sparse_gb_complement_scores)) + " {" + str(min(real_sparse_gb_complement_scores)) + "," + str(max(real_sparse_gb_complement_scores)) + "}",'Dimensionality': str(mean(real_dimension_gb_complement_scores)) + " {" + str(min(real_dimension_gb_complement_scores)) + "," + str(max(real_dimension_gb_complement_scores)) + "}"})
        thewriter.writerow({'Model': 'Support Vector Machines (sigmoid)', 'Linear': str(mean(real_linear_svm_scores))+" {"+str(min(real_linear_svm_scores))+","+str(max(real_linear_svm_scores))+"}",'Non-linear': str(mean(real_nonlinear_svm_scores))+" {"+str(min(real_nonlinear_svm_scores))+","+str(max(real_nonlinear_svm_scores))+"}", 'Sparsity': str(mean(real_sparse_svm_scores))+" {"+str(min(real_sparse_svm_scores))+","+str(max(real_sparse_svm_scores))+"}", 'Dimensionality': str(mean(real_dimension_svm_scores))+" {"+str(min(real_dimension_svm_scores))+","+str(max(real_dimension_svm_scores))+"}"})
        #thewriter.writerow({'Model': 'Support Vector Machines (linear)','Linear': str(mean(real_linear_svm_linear_scores)) + " {" + str(min(real_linear_svm_linear_scores)) + "," + str(max(real_linear_svm_linear_scores)) + "}",'Non-linear': str(mean(real_nonlinear_svm_linear_scores)) + " {" + str(min(real_nonlinear_svm_linear_scores)) + "," + str(max(real_nonlinear_svm_linear_scores)) + "}",'Sparsity': str(mean(real_sparse_svm_linear_scores)) + " {" + str(min(real_sparse_svm_linear_scores)) + "," + str(max(real_sparse_svm_linear_scores)) + "}",'Dimensionality': str(mean(real_dimension_svm_linear_scores)) + " {" + str(min(real_dimension_svm_linear_scores)) + "," + str(max(real_dimension_svm_linear_scores)) + "}"})
        thewriter.writerow({'Model': 'Support Vector Machines (poly)','Linear': str(mean(real_linear_svm_poly_scores)) + " {" + str(min(real_linear_svm_poly_scores)) + "," + str(max(real_linear_svm_poly_scores)) + "}",'Non-linear': str(mean(real_nonlinear_svm_poly_scores)) + " {" + str(min(real_nonlinear_svm_poly_scores)) + "," + str(max(real_nonlinear_svm_poly_scores)) + "}",'Sparsity': str(mean(real_sparse_svm_poly_scores)) + " {" + str(min(real_sparse_svm_poly_scores)) + "," + str(max(real_sparse_svm_poly_scores)) + "}",'Dimensionality': str(mean(real_dimension_svm_poly_scores)) + " {" + str(min(real_dimension_svm_poly_scores)) + "," + str(max(real_dimension_svm_poly_scores)) + "}"})
        thewriter.writerow({'Model': 'Support Vector Machines (rbf)','Linear': str(mean(real_linear_svm_rbf_scores)) + " {" + str(min(real_linear_svm_rbf_scores)) + "," + str(max(real_linear_svm_rbf_scores)) + "}",'Non-linear': str(mean(real_nonlinear_svm_rbf_scores)) + " {" + str(min(real_nonlinear_svm_rbf_scores)) + "," + str(max(real_nonlinear_svm_rbf_scores)) + "}",'Sparsity': str(mean(real_sparse_svm_rbf_scores)) + " {" + str(min(real_sparse_svm_rbf_scores)) + "," + str(max(real_sparse_svm_rbf_scores)) + "}",'Dimensionality': str(mean(real_dimension_svm_rbf_scores)) + " {" + str(min(real_dimension_svm_rbf_scores)) + "," + str(max(real_dimension_svm_rbf_scores)) + "}"})
        #thewriter.writerow({'Model': 'Support Vector Machines (precomputed)','Linear': str(mean(real_linear_svm_precomputed_scores)) + " {" + str(min(real_linear_svm_precomputed_scores)) + "," + str(max(real_linear_svm_precomputed_scores)) + "}",'Non-linear': str(mean(real_nonlinear_svm_precomputed_scores)) + " {" + str(min(real_nonlinear_svm_precomputed_scores)) + "," + str(max(real_nonlinear_svm_precomputed_scores)) + "}",'Sparsity': str(mean(real_sparse_svm_precomputed_scores)) + " {" + str(min(real_sparse_svm_precomputed_scores)) + "," + str(max(real_sparse_svm_precomputed_scores)) + "}",'Dimensionality': str(mean(real_dimension_svm_precomputed_scores)) + " {" + str(min(real_dimension_svm_precomputed_scores)) + "," + str(max(real_dimension_svm_precomputed_scores)) + "}"})
        thewriter.writerow({'Model': 'K Nearest Neighbor (uniform)', 'Linear': str(mean(real_linear_knn_scores))+" {"+str(min(real_linear_knn_scores))+","+str(max(real_linear_knn_scores))+"}",'Non-linear': str(mean(real_nonlinear_knn_scores))+" {"+str(min(real_nonlinear_knn_scores))+","+str(max(real_nonlinear_knn_scores))+"}", 'Sparsity': str(mean(real_sparse_knn_scores))+" {"+str(min(real_sparse_knn_scores))+","+str(max(real_sparse_knn_scores))+"}", 'Dimensionality': str(mean(real_dimension_knn_scores))+" {"+str(min(real_dimension_knn_scores))+","+str(max(real_dimension_knn_scores))+"}"})
        thewriter.writerow({'Model': 'K Nearest Neighbor (distance)', 'Linear': str(mean(real_linear_knn_distance_scores))+" {"+str(min(real_linear_knn_distance_scores))+","+str(max(real_linear_knn_distance_scores))+"}",'Non-linear': str(mean(real_nonlinear_knn_distance_scores))+" {"+str(min(real_nonlinear_knn_distance_scores))+","+str(max(real_nonlinear_knn_distance_scores))+"}", 'Sparsity': str(mean(real_sparse_knn_distance_scores))+" {"+str(min(real_sparse_knn_distance_scores))+","+str(max(real_sparse_knn_distance_scores))+"}", 'Dimensionality': str(mean(real_dimension_knn_distance_scores))+" {"+str(min(real_dimension_knn_distance_scores))+","+str(max(real_dimension_knn_distance_scores))+"}"})

write_real_to_csv()

def write_real_to_figures():
    plt.style.use('_mpl-gallery')
    #plt.rcParams["figure.autolayout"] = True
    #labels = list('ABCDEFGHIJK')#'NT_L2', 'NT_Log','NT_Poi', 'BN_HC', 'BN_Tabu', 'BN_PC', 'BN_GS', 'BN_IAMB', 'BN_MMHC', 'BN_RSMAX2', 'BN_H2PC')
    fs = 10  # fontsize
    # make data:
    # plot

    fig, ax = plt.subplots(nrows=17, ncols=4,figsize=(80, 80), sharey=True)
    fig.tight_layout(h_pad=2)
    ax[0, 0].boxplot([notears_l2_linear_dict_scores["dt"], notears_linear_dict_scores["dt"], notears_poisson_linear_dict_scores["dt"], bnlearn_linear_dict_scores["dt"], bnlearn_tabu_linear_dict_scores["dt"], bnlearn_pc_linear_dict_scores["dt"], bnlearn_gs_linear_dict_scores["dt"], [0], bnlearn_mmhc_linear_dict_scores["dt"], bnlearn_rsmax2_linear_dict_scores["dt"], bnlearn_h2pc_linear_dict_scores["dt"]],showmeans=True, meanline=True)
    ax[0, 0].set_title('DT-gini-Linear', fontsize=fs)
    ax[0, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[0, 1].boxplot([notears_l2_nonlinear_dict_scores["dt"], notears_nonlinear_dict_scores["dt"], notears_poisson_nonlinear_dict_scores["dt"], bnlearn_nonlinear_dict_scores["dt"], bnlearn_tabu_nonlinear_dict_scores["dt"], [0], [0], [0], bnlearn_mmhc_nonlinear_dict_scores["dt"], bnlearn_rsmax2_nonlinear_dict_scores["dt"], bnlearn_h2pc_nonlinear_dict_scores["dt"]], showmeans=True, meanline=True)
    ax[0, 1].set_title('DT-gini-Nonlinear', fontsize=fs)
    ax[0, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[0, 2].boxplot([notears_l2_sparse_dict_scores["dt"], notears_sparse_dict_scores["dt"], notears_poisson_sparse_dict_scores["dt"], bnlearn_sparse_dict_scores["dt"], bnlearn_tabu_sparse_dict_scores["dt"], [0], [0], [0], bnlearn_mmhc_sparse_dict_scores["dt"], bnlearn_rsmax2_sparse_dict_scores["dt"], bnlearn_h2pc_sparse_dict_scores["dt"]],showmeans=True, meanline=True)
    ax[0, 2].set_title('DT-gini-Sparse', fontsize=fs)
    ax[0, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[0, 3].boxplot([notears_l2_dimension_dict_scores["dt"], notears_dimension_dict_scores["dt"], notears_poisson_dimension_dict_scores["dt"], bnlearn_dimension_dict_scores["dt"], bnlearn_tabu_dimension_dict_scores["dt"], [0], [0], [0], bnlearn_mmhc_dimension_dict_scores["dt"], bnlearn_rsmax2_dimension_dict_scores["dt"], bnlearn_h2pc_dimension_dict_scores["dt"]], showmeans=True, meanline=True)
    ax[0, 3].set_title('DT-gini-Dimensional', fontsize=fs)
    ax[0, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[1, 0].boxplot([notears_l2_linear_dict_scores["dt"], notears_linear_dict_scores["dt"],
                      notears_poisson_linear_dict_scores["dt"], bnlearn_linear_dict_scores["dt"],
                      bnlearn_tabu_linear_dict_scores["dt"], bnlearn_pc_linear_dict_scores["dt"],
                      bnlearn_gs_linear_dict_scores["dt"], [0],
                      bnlearn_mmhc_linear_dict_scores["dt"], bnlearn_rsmax2_linear_dict_scores["dt"],
                      bnlearn_h2pc_linear_dict_scores["dt"]], showmeans=True, meanline=True)
    ax[1, 0].set_title('DT-entropy-Linear', fontsize=fs)
    ax[1, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[1, 1].boxplot([notears_l2_nonlinear_dict_scores["dt_e"], notears_nonlinear_dict_scores["dt_e"],
                      notears_poisson_nonlinear_dict_scores["dt_e"], bnlearn_nonlinear_dict_scores["dt_e"],
                      bnlearn_tabu_nonlinear_dict_scores["dt_e"], [0], [0], [0],
                      bnlearn_mmhc_nonlinear_dict_scores["dt_e"], bnlearn_rsmax2_nonlinear_dict_scores["dt_e"],
                      bnlearn_h2pc_nonlinear_dict_scores["dt_e"]], showmeans=True, meanline=True)
    ax[1, 1].set_title('DT-entropy-Nonlinear', fontsize=fs)
    ax[1, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[1, 2].boxplot([notears_l2_sparse_dict_scores["dt_e"], notears_sparse_dict_scores["dt_e"],
                      notears_poisson_sparse_dict_scores["dt_e"], bnlearn_sparse_dict_scores["dt_e"],
                      bnlearn_tabu_sparse_dict_scores["dt_e"], [0], [0], [0], bnlearn_mmhc_sparse_dict_scores["dt_e"],
                      bnlearn_rsmax2_sparse_dict_scores["dt_e"], bnlearn_h2pc_sparse_dict_scores["dt_e"]], showmeans=True,
                     meanline=True)
    ax[1, 2].set_title('DT-entropy-Sparse', fontsize=fs)
    ax[1, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[1, 3].boxplot([notears_l2_dimension_dict_scores["dt_e"], notears_dimension_dict_scores["dt_e"],
                      notears_poisson_dimension_dict_scores["dt_e"], bnlearn_dimension_dict_scores["dt_e"],
                      bnlearn_tabu_dimension_dict_scores["dt_e"], [0], [0], [0], bnlearn_mmhc_dimension_dict_scores["dt_e"],
                      bnlearn_rsmax2_dimension_dict_scores["dt_e"], bnlearn_h2pc_dimension_dict_scores["dt_e"]],
                     showmeans=True, meanline=True)
    ax[1, 3].set_title('DT-entropy-Dimensional', fontsize=fs)
    ax[1, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)

    ax[2, 0].boxplot([notears_l2_linear_dict_scores["rf"], notears_linear_dict_scores["rf"], notears_poisson_linear_dict_scores["rf"], bnlearn_linear_dict_scores["rf"], bnlearn_tabu_linear_dict_scores["rf"], bnlearn_pc_linear_dict_scores["rf"], bnlearn_gs_linear_dict_scores["rf"], [0], bnlearn_mmhc_linear_dict_scores["rf"], bnlearn_rsmax2_linear_dict_scores["rf"], bnlearn_h2pc_linear_dict_scores["rf"]], showmeans=True, meanline=True)
    ax[2, 0].set_title('RF-gini-Linear', fontsize=fs)
    ax[2, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[2, 1].boxplot([notears_l2_nonlinear_dict_scores["rf"], notears_nonlinear_dict_scores["rf"], notears_poisson_nonlinear_dict_scores["rf"], bnlearn_nonlinear_dict_scores["rf"], bnlearn_tabu_nonlinear_dict_scores["rf"], [0], [0], [0], bnlearn_mmhc_nonlinear_dict_scores["rf"], bnlearn_rsmax2_nonlinear_dict_scores["rf"], bnlearn_h2pc_nonlinear_dict_scores["rf"]], showmeans=True, meanline=True)
    ax[2, 1].set_title('RF-gini-Nonlinear', fontsize=fs)
    ax[2, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[2, 2].boxplot([notears_l2_sparse_dict_scores["rf"], notears_sparse_dict_scores["rf"],notears_poisson_sparse_dict_scores["rf"], bnlearn_sparse_dict_scores["rf"],bnlearn_tabu_sparse_dict_scores["rf"], [0],[0], [0],bnlearn_mmhc_sparse_dict_scores["rf"], bnlearn_rsmax2_sparse_dict_scores["rf"],bnlearn_h2pc_sparse_dict_scores["rf"]], showmeans=True, meanline=True)
    ax[2, 2].set_title('RF-gini-Sparse', fontsize=fs)
    ax[2, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[2, 3].boxplot([notears_l2_dimension_dict_scores["rf"], notears_dimension_dict_scores["rf"],notears_poisson_dimension_dict_scores["rf"], bnlearn_dimension_dict_scores["rf"],bnlearn_tabu_dimension_dict_scores["rf"], [0],[0], [0],bnlearn_mmhc_dimension_dict_scores["rf"], bnlearn_rsmax2_dimension_dict_scores["rf"],bnlearn_h2pc_dimension_dict_scores["rf"]], showmeans=True, meanline=True)
    ax[2, 3].set_title('RF-gini-Dimensional', fontsize=fs)
    ax[2, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)

    ax[3, 0].boxplot([notears_l2_linear_dict_scores["rf_e"], notears_linear_dict_scores["rf_e"],
                      notears_poisson_linear_dict_scores["rf_e"], bnlearn_linear_dict_scores["rf_e"],
                      bnlearn_tabu_linear_dict_scores["rf_e"], bnlearn_pc_linear_dict_scores["rf_e"],
                      bnlearn_gs_linear_dict_scores["rf_e"], [0],
                      bnlearn_mmhc_linear_dict_scores["rf_e"], bnlearn_rsmax2_linear_dict_scores["rf_e"],
                      bnlearn_h2pc_linear_dict_scores["rf_e"]], showmeans=True, meanline=True)
    ax[3, 0].set_title('RF-entropy-Linear', fontsize=fs)
    ax[3, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[3, 1].boxplot([notears_l2_nonlinear_dict_scores["rf_e"], notears_nonlinear_dict_scores["rf_e"],
                      notears_poisson_nonlinear_dict_scores["rf_e"], bnlearn_nonlinear_dict_scores["rf_e"],
                      bnlearn_tabu_nonlinear_dict_scores["rf_e"], [0], [0], [0],
                      bnlearn_mmhc_nonlinear_dict_scores["rf_e"], bnlearn_rsmax2_nonlinear_dict_scores["rf_e"],
                      bnlearn_h2pc_nonlinear_dict_scores["rf_e"]], showmeans=True, meanline=True)
    ax[3, 1].set_title('RF-entropy-Nonlinear', fontsize=fs)
    ax[3, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[3, 2].boxplot([notears_l2_sparse_dict_scores["rf_e"], notears_sparse_dict_scores["rf_e"],
                      notears_poisson_sparse_dict_scores["rf_e"], bnlearn_sparse_dict_scores["rf_e"],
                      bnlearn_tabu_sparse_dict_scores["rf_e"], [0], [0], [0], bnlearn_mmhc_sparse_dict_scores["rf_e"],
                      bnlearn_rsmax2_sparse_dict_scores["rf_e"], bnlearn_h2pc_sparse_dict_scores["rf_e"]], showmeans=True,
                     meanline=True)
    ax[3, 2].set_title('RF-entropy-Sparse', fontsize=fs)
    ax[3, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[3, 3].boxplot([notears_l2_dimension_dict_scores["rf_e"], notears_dimension_dict_scores["rf_e"],
                      notears_poisson_dimension_dict_scores["rf_e"], bnlearn_dimension_dict_scores["rf_e"],
                      bnlearn_tabu_dimension_dict_scores["rf_e"], [0], [0], [0], bnlearn_mmhc_dimension_dict_scores["rf_e"],
                      bnlearn_rsmax2_dimension_dict_scores["rf_e"], bnlearn_h2pc_dimension_dict_scores["rf_e"]],
                     showmeans=True, meanline=True)
    ax[3, 3].set_title('RF-entropy-Dimensional', fontsize=fs)
    ax[3, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)

    ax[4, 0].boxplot([notears_l2_linear_dict_scores["lr"], notears_linear_dict_scores["lr"],notears_poisson_linear_dict_scores["lr"], bnlearn_linear_dict_scores["lr"],bnlearn_tabu_linear_dict_scores["lr"], bnlearn_pc_linear_dict_scores["lr"],bnlearn_gs_linear_dict_scores["lr"], [0],bnlearn_mmhc_linear_dict_scores["lr"], bnlearn_rsmax2_linear_dict_scores["lr"],bnlearn_h2pc_linear_dict_scores["lr"]], showmeans=True, meanline=True)
    ax[4, 0].set_title('LR-none-Linear', fontsize=fs)
    ax[4, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[4, 1].boxplot([notears_l2_nonlinear_dict_scores["lr"], notears_nonlinear_dict_scores["lr"],notears_poisson_nonlinear_dict_scores["lr"], bnlearn_nonlinear_dict_scores["lr"],bnlearn_tabu_nonlinear_dict_scores["lr"], [0],[0], [0],bnlearn_mmhc_nonlinear_dict_scores["lr"], bnlearn_rsmax2_nonlinear_dict_scores["lr"],bnlearn_h2pc_nonlinear_dict_scores["lr"]], showmeans=True, meanline=True)
    ax[4, 1].set_title('LR-none-Nonlinear', fontsize=fs)
    ax[4, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[4, 2].boxplot([notears_l2_sparse_dict_scores["lr"], notears_sparse_dict_scores["lr"],notears_poisson_sparse_dict_scores["lr"], bnlearn_sparse_dict_scores["lr"],bnlearn_tabu_sparse_dict_scores["lr"], [0],[0], [0],bnlearn_mmhc_sparse_dict_scores["lr"], bnlearn_rsmax2_sparse_dict_scores["lr"],bnlearn_h2pc_sparse_dict_scores["lr"]],  showmeans=True, meanline=True)
    ax[4, 2].set_title('LR-none-Sparse', fontsize=fs)
    ax[4, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[4, 3].boxplot([notears_l2_dimension_dict_scores["lr"], notears_dimension_dict_scores["lr"],notears_poisson_dimension_dict_scores["lr"], bnlearn_dimension_dict_scores["lr"],bnlearn_tabu_dimension_dict_scores["lr"], [0],[0], [0],bnlearn_mmhc_dimension_dict_scores["lr"], bnlearn_rsmax2_dimension_dict_scores["lr"],bnlearn_h2pc_dimension_dict_scores["lr"]], showmeans=True, meanline=True)
    ax[4, 3].set_title('LR-none-Dimensional', fontsize=fs)
    ax[4, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)

    ax[5, 0].boxplot([notears_l2_linear_dict_scores["lr_l1"], notears_linear_dict_scores["lr_l1"],
                      notears_poisson_linear_dict_scores["lr_l1"], bnlearn_linear_dict_scores["lr_l1"],
                      bnlearn_tabu_linear_dict_scores["lr_l1"], bnlearn_pc_linear_dict_scores["lr_l1"],
                      bnlearn_gs_linear_dict_scores["lr_l1"], [0],
                      bnlearn_mmhc_linear_dict_scores["lr_l1"], bnlearn_rsmax2_linear_dict_scores["lr_l1"],
                      bnlearn_h2pc_linear_dict_scores["lr_l1"]], showmeans=True, meanline=True)
    ax[5, 0].set_title('LR-L1-Linear', fontsize=fs)
    ax[5, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[5, 1].boxplot([notears_l2_nonlinear_dict_scores["lr_l1"], notears_nonlinear_dict_scores["lr_l1"],
                      notears_poisson_nonlinear_dict_scores["lr_l1"], bnlearn_nonlinear_dict_scores["lr_l1"],
                      bnlearn_tabu_nonlinear_dict_scores["lr_l1"], [0], [0], [0],
                      bnlearn_mmhc_nonlinear_dict_scores["lr_l1"], bnlearn_rsmax2_nonlinear_dict_scores["lr_l1"],
                      bnlearn_h2pc_nonlinear_dict_scores["lr_l1"]], showmeans=True, meanline=True)
    ax[5, 1].set_title('LR-L1-Nonlinear', fontsize=fs)
    ax[5, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[5, 2].boxplot([notears_l2_sparse_dict_scores["lr_l1"], notears_sparse_dict_scores["lr_l1"],
                      notears_poisson_sparse_dict_scores["lr_l1"], bnlearn_sparse_dict_scores["lr_l1"],
                      bnlearn_tabu_sparse_dict_scores["lr_l1"], [0], [0], [0], bnlearn_mmhc_sparse_dict_scores["lr_l1"],
                      bnlearn_rsmax2_sparse_dict_scores["lr_l1"], bnlearn_h2pc_sparse_dict_scores["lr_l1"]], showmeans=True,
                     meanline=True)
    ax[5, 2].set_title('LR-L1-Sparse', fontsize=fs)
    ax[5, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[5, 3].boxplot([notears_l2_dimension_dict_scores["lr_l1"], notears_dimension_dict_scores["lr_l1"],
                      notears_poisson_dimension_dict_scores["lr_l1"], bnlearn_dimension_dict_scores["lr_l1"],
                      bnlearn_tabu_dimension_dict_scores["lr_l1"], [0], [0], [0], bnlearn_mmhc_dimension_dict_scores["lr_l1"],
                      bnlearn_rsmax2_dimension_dict_scores["lr_l1"], bnlearn_h2pc_dimension_dict_scores["lr_l1"]],
                     showmeans=True, meanline=True)
    ax[5, 3].set_title('LR-L1-Dimensional', fontsize=fs)
    ax[5, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)

    ax[6, 0].boxplot([notears_l2_linear_dict_scores["lr_l2"], notears_linear_dict_scores["lr_l2"],
                      notears_poisson_linear_dict_scores["lr_l2"], bnlearn_linear_dict_scores["lr_l2"],
                      bnlearn_tabu_linear_dict_scores["lr_l2"], bnlearn_pc_linear_dict_scores["lr_l2"],
                      bnlearn_gs_linear_dict_scores["lr_l2"], [0],
                      bnlearn_mmhc_linear_dict_scores["lr_l2"], bnlearn_rsmax2_linear_dict_scores["lr_l2"],
                      bnlearn_h2pc_linear_dict_scores["lr_l2"]], showmeans=True, meanline=True)
    ax[6, 0].set_title('LR-L2-Linear', fontsize=fs)
    ax[6, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[6, 1].boxplot([notears_l2_nonlinear_dict_scores["lr_l2"], notears_nonlinear_dict_scores["lr_l2"],
                      notears_poisson_nonlinear_dict_scores["lr_l2"], bnlearn_nonlinear_dict_scores["lr_l2"],
                      bnlearn_tabu_nonlinear_dict_scores["lr_l2"], [0], [0], [0],
                      bnlearn_mmhc_nonlinear_dict_scores["lr_l2"], bnlearn_rsmax2_nonlinear_dict_scores["lr_l2"],
                      bnlearn_h2pc_nonlinear_dict_scores["lr_l2"]], showmeans=True, meanline=True)
    ax[6, 1].set_title('LR-L2-Nonlinear', fontsize=fs)
    ax[6, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[6, 2].boxplot([notears_l2_sparse_dict_scores["lr_l2"], notears_sparse_dict_scores["lr_l2"],
                      notears_poisson_sparse_dict_scores["lr_l2"], bnlearn_sparse_dict_scores["lr_l2"],
                      bnlearn_tabu_sparse_dict_scores["lr_l2"], [0], [0], [0], bnlearn_mmhc_sparse_dict_scores["lr_l2"],
                      bnlearn_rsmax2_sparse_dict_scores["lr_l2"], bnlearn_h2pc_sparse_dict_scores["lr_l2"]], showmeans=True,
                     meanline=True)
    ax[6, 2].set_title('LR-L2-Sparse', fontsize=fs)
    ax[6, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[6, 3].boxplot([notears_l2_dimension_dict_scores["lr_l2"], notears_dimension_dict_scores["lr_l2"],
                      notears_poisson_dimension_dict_scores["lr_l2"], bnlearn_dimension_dict_scores["lr_l2"],
                      bnlearn_tabu_dimension_dict_scores["lr_l2"], [0], [0], [0], bnlearn_mmhc_dimension_dict_scores["lr_l2"],
                      bnlearn_rsmax2_dimension_dict_scores["lr_l2"], bnlearn_h2pc_dimension_dict_scores["lr_l2"]],
                     showmeans=True, meanline=True)
    ax[6, 3].set_title('LR-L2-Dimensional', fontsize=fs)
    ax[6, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)

    ax[7, 0].boxplot([notears_l2_linear_dict_scores["lr_e"], notears_linear_dict_scores["lr_e"],
                      notears_poisson_linear_dict_scores["lr_e"], bnlearn_linear_dict_scores["lr_e"],
                      bnlearn_tabu_linear_dict_scores["lr_e"], bnlearn_pc_linear_dict_scores["lr_e"],
                      bnlearn_gs_linear_dict_scores["lr_e"], [0],
                      bnlearn_mmhc_linear_dict_scores["lr_e"], bnlearn_rsmax2_linear_dict_scores["lr_e"],
                      bnlearn_h2pc_linear_dict_scores["lr_e"]], showmeans=True, meanline=True)
    ax[7, 0].set_title('LR-elasticnet-Linear', fontsize=fs)
    ax[7, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[7, 1].boxplot([notears_l2_nonlinear_dict_scores["lr_e"], notears_nonlinear_dict_scores["lr_e"],
                      notears_poisson_nonlinear_dict_scores["lr_e"], bnlearn_nonlinear_dict_scores["lr_e"],
                      bnlearn_tabu_nonlinear_dict_scores["lr_e"], [0], [0], [0],
                      bnlearn_mmhc_nonlinear_dict_scores["lr_e"], bnlearn_rsmax2_nonlinear_dict_scores["lr_e"],
                      bnlearn_h2pc_nonlinear_dict_scores["lr_e"]], showmeans=True, meanline=True)
    ax[7, 1].set_title('LR-elasticnet-Nonlinear', fontsize=fs)
    ax[7, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[7, 2].boxplot([notears_l2_sparse_dict_scores["lr_e"], notears_sparse_dict_scores["lr_e"],
                      notears_poisson_sparse_dict_scores["lr_e"], bnlearn_sparse_dict_scores["lr_e"],
                      bnlearn_tabu_sparse_dict_scores["lr_e"], [0], [0], [0], bnlearn_mmhc_sparse_dict_scores["lr_e"],
                      bnlearn_rsmax2_sparse_dict_scores["lr_e"], bnlearn_h2pc_sparse_dict_scores["lr_e"]], showmeans=True,
                     meanline=True)
    ax[7, 2].set_title('LR-elasticnet-Sparse', fontsize=fs)
    ax[7, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[7, 3].boxplot([notears_l2_dimension_dict_scores["lr_e"], notears_dimension_dict_scores["lr_e"],
                      notears_poisson_dimension_dict_scores["lr_e"], bnlearn_dimension_dict_scores["lr_e"],
                      bnlearn_tabu_dimension_dict_scores["lr_e"], [0], [0], [0], bnlearn_mmhc_dimension_dict_scores["lr_e"],
                      bnlearn_rsmax2_dimension_dict_scores["lr_e"], bnlearn_h2pc_dimension_dict_scores["lr_e"]],
                     showmeans=True, meanline=True)
    ax[7, 3].set_title('LR-elasticnet-Dimensional', fontsize=fs)
    ax[7, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)


    ax[8, 0].boxplot([notears_l2_linear_dict_scores["nb"], notears_linear_dict_scores["nb"],notears_poisson_linear_dict_scores["nb"], bnlearn_linear_dict_scores["nb"],bnlearn_tabu_linear_dict_scores["nb"], bnlearn_pc_linear_dict_scores["nb"],bnlearn_gs_linear_dict_scores["nb"], [0],bnlearn_mmhc_linear_dict_scores["nb"], bnlearn_rsmax2_linear_dict_scores["nb"],bnlearn_h2pc_linear_dict_scores["nb"]], showmeans=True, meanline=True)
    ax[8, 0].set_title('NB-bernoulli-Linear', fontsize=fs)
    ax[8, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[8, 1].boxplot([notears_l2_nonlinear_dict_scores["nb"], notears_nonlinear_dict_scores["nb"],notears_poisson_nonlinear_dict_scores["nb"], bnlearn_nonlinear_dict_scores["nb"],bnlearn_tabu_nonlinear_dict_scores["nb"], [0],[0], [0],bnlearn_mmhc_nonlinear_dict_scores["nb"], bnlearn_rsmax2_nonlinear_dict_scores["nb"],bnlearn_h2pc_nonlinear_dict_scores["nb"]], showmeans=True, meanline=True)
    ax[8, 1].set_title('NB-bernoulli-Nonlinear', fontsize=fs)
    ax[8, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[8, 2].boxplot([notears_l2_sparse_dict_scores["nb"], notears_sparse_dict_scores["nb"],notears_poisson_sparse_dict_scores["nb"], bnlearn_sparse_dict_scores["nb"],bnlearn_tabu_sparse_dict_scores["nb"], [0],[0], [0],bnlearn_mmhc_sparse_dict_scores["nb"], bnlearn_rsmax2_sparse_dict_scores["nb"],bnlearn_h2pc_sparse_dict_scores["nb"]], showmeans=True, meanline=True)
    ax[8, 2].set_title('NB-bernoulli-Sparse', fontsize=fs)
    ax[8, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[8, 3].boxplot([notears_l2_dimension_dict_scores["nb"], notears_dimension_dict_scores["nb"],notears_poisson_dimension_dict_scores["nb"], bnlearn_dimension_dict_scores["nb"],bnlearn_tabu_dimension_dict_scores["nb"], [0],[0], [0],bnlearn_mmhc_dimension_dict_scores["nb"], bnlearn_rsmax2_dimension_dict_scores["nb"],bnlearn_h2pc_dimension_dict_scores["nb"]], showmeans=True, meanline=True)
    ax[8, 3].set_title('NB-bernoulli-Dimensional', fontsize=fs)
    ax[8, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)

    ax[9, 0].boxplot([notears_l2_linear_dict_scores["nb_g"], notears_linear_dict_scores["nb_g"],
                      notears_poisson_linear_dict_scores["nb_g"], bnlearn_linear_dict_scores["nb_g"],
                      bnlearn_tabu_linear_dict_scores["nb_g"], bnlearn_pc_linear_dict_scores["nb_g"],
                      bnlearn_gs_linear_dict_scores["nb_g"], [0],
                      bnlearn_mmhc_linear_dict_scores["nb_g"], bnlearn_rsmax2_linear_dict_scores["nb_g"],
                      bnlearn_h2pc_linear_dict_scores["nb_g"]], showmeans=True, meanline=True)
    ax[9, 0].set_title('NB-gaussian-Linear', fontsize=fs)
    ax[9, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[9, 1].boxplot([notears_l2_nonlinear_dict_scores["nb_g"], notears_nonlinear_dict_scores["nb_g"],
                      notears_poisson_nonlinear_dict_scores["nb_g"], bnlearn_nonlinear_dict_scores["nb_g"],
                      bnlearn_tabu_nonlinear_dict_scores["nb_g"], [0], [0], [0],
                      bnlearn_mmhc_nonlinear_dict_scores["nb_g"], bnlearn_rsmax2_nonlinear_dict_scores["nb_g"],
                      bnlearn_h2pc_nonlinear_dict_scores["nb_g"]], showmeans=True, meanline=True)
    ax[9, 1].set_title('NB-gaussian-Nonlinear', fontsize=fs)
    ax[9, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[9, 2].boxplot([notears_l2_sparse_dict_scores["nb_g"], notears_sparse_dict_scores["nb_g"],
                      notears_poisson_sparse_dict_scores["nb_g"], bnlearn_sparse_dict_scores["nb_g"],
                      bnlearn_tabu_sparse_dict_scores["nb_g"], [0], [0], [0], bnlearn_mmhc_sparse_dict_scores["nb_g"],
                      bnlearn_rsmax2_sparse_dict_scores["nb_g"], bnlearn_h2pc_sparse_dict_scores["nb_g"]], showmeans=True,
                     meanline=True)
    ax[9, 2].set_title('NB-gaussian-Sparse', fontsize=fs)
    ax[9, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[9, 3].boxplot([notears_l2_dimension_dict_scores["nb_g"], notears_dimension_dict_scores["nb_g"],
                      notears_poisson_dimension_dict_scores["nb_g"], bnlearn_dimension_dict_scores["nb_g"],
                      bnlearn_tabu_dimension_dict_scores["nb_g"], [0], [0], [0], bnlearn_mmhc_dimension_dict_scores["nb_g"],
                      bnlearn_rsmax2_dimension_dict_scores["nb_g"], bnlearn_h2pc_dimension_dict_scores["nb_g"]],
                     showmeans=True, meanline=True)
    ax[9, 3].set_title('NB-gaussian-Dimensional', fontsize=fs)
    ax[9, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)

    ax[10, 0].boxplot([notears_l2_linear_dict_scores["nb_m"], notears_linear_dict_scores["nb_m"],
                      notears_poisson_linear_dict_scores["nb_m"], bnlearn_linear_dict_scores["nb_m"],
                      bnlearn_tabu_linear_dict_scores["nb_m"], bnlearn_pc_linear_dict_scores["nb_m"],
                      bnlearn_gs_linear_dict_scores["nb_m"], [0],
                      bnlearn_mmhc_linear_dict_scores["nb_m"], bnlearn_rsmax2_linear_dict_scores["nb_m"],
                      bnlearn_h2pc_linear_dict_scores["nb_m"]], showmeans=True, meanline=True)
    ax[10, 0].set_title('NB-multinomial-Linear', fontsize=fs)
    ax[10, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[10, 1].boxplot([notears_l2_nonlinear_dict_scores["nb_m"], notears_nonlinear_dict_scores["nb_m"],
                      notears_poisson_nonlinear_dict_scores["nb_m"], bnlearn_nonlinear_dict_scores["nb_m"],
                      bnlearn_tabu_nonlinear_dict_scores["nb_m"], [0], [0], [0],
                      bnlearn_mmhc_nonlinear_dict_scores["nb_m"], bnlearn_rsmax2_nonlinear_dict_scores["nb_m"],
                      bnlearn_h2pc_nonlinear_dict_scores["nb_m"]], showmeans=True, meanline=True)
    ax[10, 1].set_title('NB-multinomial-Nonlinear', fontsize=fs)
    ax[10, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[10, 2].boxplot([notears_l2_sparse_dict_scores["nb_m"], notears_sparse_dict_scores["nb_m"],
                      notears_poisson_sparse_dict_scores["nb_m"], bnlearn_sparse_dict_scores["nb_m"],
                      bnlearn_tabu_sparse_dict_scores["nb_m"], [0], [0], [0], bnlearn_mmhc_sparse_dict_scores["nb_m"],
                      bnlearn_rsmax2_sparse_dict_scores["nb_m"], bnlearn_h2pc_sparse_dict_scores["nb_m"]],
                     showmeans=True,
                     meanline=True)
    ax[10, 2].set_title('NB-multinomial-Sparse', fontsize=fs)
    ax[10, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[10, 3].boxplot([notears_l2_dimension_dict_scores["nb_m"], notears_dimension_dict_scores["nb_m"],
                      notears_poisson_dimension_dict_scores["nb_m"], bnlearn_dimension_dict_scores["nb_m"],
                      bnlearn_tabu_dimension_dict_scores["nb_m"], [0], [0], [0],
                      bnlearn_mmhc_dimension_dict_scores["nb_m"],
                      bnlearn_rsmax2_dimension_dict_scores["nb_m"], bnlearn_h2pc_dimension_dict_scores["nb_m"]],
                     showmeans=True, meanline=True)
    ax[10, 3].set_title('NB-multinomial-Dimensional', fontsize=fs)
    ax[10, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)

    ax[11, 0].boxplot([notears_l2_linear_dict_scores["nb_c"], notears_linear_dict_scores["nb_c"],
                      notears_poisson_linear_dict_scores["nb_c"], bnlearn_linear_dict_scores["nb_c"],
                      bnlearn_tabu_linear_dict_scores["nb_c"], bnlearn_pc_linear_dict_scores["nb_c"],
                      bnlearn_gs_linear_dict_scores["nb_c"], [0],
                      bnlearn_mmhc_linear_dict_scores["nb_c"], bnlearn_rsmax2_linear_dict_scores["nb_c"],
                      bnlearn_h2pc_linear_dict_scores["nb_c"]], showmeans=True, meanline=True)
    ax[11, 0].set_title('NB-complement-Linear', fontsize=fs)
    ax[11, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[11, 1].boxplot([notears_l2_nonlinear_dict_scores["nb_c"], notears_nonlinear_dict_scores["nb_c"],
                      notears_poisson_nonlinear_dict_scores["nb_c"], bnlearn_nonlinear_dict_scores["nb_c"],
                      bnlearn_tabu_nonlinear_dict_scores["nb_c"], [0], [0], [0],
                      bnlearn_mmhc_nonlinear_dict_scores["nb_c"], bnlearn_rsmax2_nonlinear_dict_scores["nb_c"],
                      bnlearn_h2pc_nonlinear_dict_scores["nb_c"]], showmeans=True, meanline=True)
    ax[11, 1].set_title('NB-complement-Nonlinear', fontsize=fs)
    ax[11, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[11, 2].boxplot([notears_l2_sparse_dict_scores["nb_c"], notears_sparse_dict_scores["nb_c"],
                      notears_poisson_sparse_dict_scores["nb_c"], bnlearn_sparse_dict_scores["nb_c"],
                      bnlearn_tabu_sparse_dict_scores["nb_c"], [0], [0], [0], bnlearn_mmhc_sparse_dict_scores["nb_c"],
                      bnlearn_rsmax2_sparse_dict_scores["nb_c"], bnlearn_h2pc_sparse_dict_scores["nb_c"]],
                     showmeans=True,
                     meanline=True)
    ax[11, 2].set_title('NB-complement-Sparse', fontsize=fs)
    ax[11, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[11, 3].boxplot([notears_l2_dimension_dict_scores["nb_c"], notears_dimension_dict_scores["nb_c"],
                      notears_poisson_dimension_dict_scores["nb_c"], bnlearn_dimension_dict_scores["nb_c"],
                      bnlearn_tabu_dimension_dict_scores["nb_c"], [0], [0], [0],
                      bnlearn_mmhc_dimension_dict_scores["nb_c"],
                      bnlearn_rsmax2_dimension_dict_scores["nb_c"], bnlearn_h2pc_dimension_dict_scores["nb_c"]],
                     showmeans=True, meanline=True)
    ax[11, 3].set_title('NB-complement-Dimensional', fontsize=fs)
    ax[11, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                         "BN_RSMAX2", "BN_H2PC"], rotation=90)

    ax[12, 0].boxplot([notears_l2_linear_dict_scores["svm"], notears_linear_dict_scores["svm"],notears_poisson_linear_dict_scores["svm"], bnlearn_linear_dict_scores["svm"],bnlearn_tabu_linear_dict_scores["svm"], bnlearn_pc_linear_dict_scores["svm"],bnlearn_gs_linear_dict_scores["svm"], [0],bnlearn_mmhc_linear_dict_scores["svm"], bnlearn_rsmax2_linear_dict_scores["svm"],bnlearn_h2pc_linear_dict_scores["svm"]], showmeans=True, meanline=True)
    ax[12, 0].set_title('SVM-sigmoid-Linear', fontsize=fs)
    ax[12, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[12, 1].boxplot([notears_l2_nonlinear_dict_scores["svm"], notears_nonlinear_dict_scores["svm"],notears_poisson_nonlinear_dict_scores["svm"], bnlearn_nonlinear_dict_scores["svm"],bnlearn_tabu_nonlinear_dict_scores["svm"], [0],[0], [0],bnlearn_mmhc_nonlinear_dict_scores["svm"], bnlearn_rsmax2_nonlinear_dict_scores["svm"],bnlearn_h2pc_nonlinear_dict_scores["svm"]], showmeans=True, meanline=True)
    ax[12, 1].set_title('SVM-sigmoid-Nonlinear', fontsize=fs)
    ax[12, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[12, 2].boxplot([notears_l2_sparse_dict_scores["svm"], notears_sparse_dict_scores["svm"],notears_poisson_sparse_dict_scores["svm"], bnlearn_sparse_dict_scores["svm"],bnlearn_tabu_sparse_dict_scores["svm"], [0],[0], [0],bnlearn_mmhc_sparse_dict_scores["svm"], bnlearn_rsmax2_sparse_dict_scores["svm"],bnlearn_h2pc_sparse_dict_scores["svm"]], showmeans=True, meanline=True)
    ax[12, 2].set_title('SVM-sigmoid-Sparse', fontsize=fs)
    ax[12, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[12, 3].boxplot([notears_l2_dimension_dict_scores["svm"], notears_dimension_dict_scores["svm"],notears_poisson_dimension_dict_scores["svm"], bnlearn_dimension_dict_scores["svm"],bnlearn_tabu_dimension_dict_scores["svm"], [0],[0], [0],bnlearn_mmhc_dimension_dict_scores["svm"], bnlearn_rsmax2_dimension_dict_scores["svm"],bnlearn_h2pc_dimension_dict_scores["svm"]],  showmeans=True, meanline=True)
    ax[12, 3].set_title('SVM-sigmoid-Dimensional', fontsize=fs)
    ax[12, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)

    ax[13, 0].boxplot([notears_l2_linear_dict_scores["svm_po"], notears_linear_dict_scores["svm_po"],
                       notears_poisson_linear_dict_scores["svm_po"], bnlearn_linear_dict_scores["svm_po"],
                       bnlearn_tabu_linear_dict_scores["svm_po"], bnlearn_pc_linear_dict_scores["svm_po"],
                       bnlearn_gs_linear_dict_scores["svm_po"], [0],
                       bnlearn_mmhc_linear_dict_scores["svm_po"], bnlearn_rsmax2_linear_dict_scores["svm_po"],
                       bnlearn_h2pc_linear_dict_scores["svm_po"]], showmeans=True, meanline=True)
    ax[13, 0].set_title('SVM-polynomial-Linear', fontsize=fs)
    ax[13, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                          "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[13, 1].boxplot([notears_l2_nonlinear_dict_scores["svm_po"], notears_nonlinear_dict_scores["svm_po"],
                       notears_poisson_nonlinear_dict_scores["svm_po"], bnlearn_nonlinear_dict_scores["svm_po"],
                       bnlearn_tabu_nonlinear_dict_scores["svm_po"], [0], [0], [0],
                       bnlearn_mmhc_nonlinear_dict_scores["svm_po"], bnlearn_rsmax2_nonlinear_dict_scores["svm_po"],
                       bnlearn_h2pc_nonlinear_dict_scores["svm_po"]], showmeans=True, meanline=True)
    ax[13, 1].set_title('SVM-polynomial-Nonlinear', fontsize=fs)
    ax[13, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                          "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[13, 2].boxplot([notears_l2_sparse_dict_scores["svm_po"], notears_sparse_dict_scores["svm_po"],
                       notears_poisson_sparse_dict_scores["svm_po"], bnlearn_sparse_dict_scores["svm_po"],
                       bnlearn_tabu_sparse_dict_scores["svm_po"], [0], [0], [0], bnlearn_mmhc_sparse_dict_scores["svm_po"],
                       bnlearn_rsmax2_sparse_dict_scores["svm_po"], bnlearn_h2pc_sparse_dict_scores["svm_po"]],
                      showmeans=True, meanline=True)
    ax[13, 2].set_title('SVM-polynomial-Sparse', fontsize=fs)
    ax[13, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                          "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[13, 3].boxplot([notears_l2_dimension_dict_scores["svm_po"], notears_dimension_dict_scores["svm_po"],
                       notears_poisson_dimension_dict_scores["svm_po"], bnlearn_dimension_dict_scores["svm_po"],
                       bnlearn_tabu_dimension_dict_scores["svm_po"], [0], [0], [0],
                       bnlearn_mmhc_dimension_dict_scores["svm_po"], bnlearn_rsmax2_dimension_dict_scores["svm_po"],
                       bnlearn_h2pc_dimension_dict_scores["svm_po"]], showmeans=True, meanline=True)
    ax[13, 3].set_title('SVM-polynomial-Dimensional', fontsize=fs)
    ax[13, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                          "BN_RSMAX2", "BN_H2PC"], rotation=90)

    ax[14, 0].boxplot([notears_l2_linear_dict_scores["svm_r"], notears_linear_dict_scores["svm_r"],
                       notears_poisson_linear_dict_scores["svm_r"], bnlearn_linear_dict_scores["svm_r"],
                       bnlearn_tabu_linear_dict_scores["svm_r"], bnlearn_pc_linear_dict_scores["svm_r"],
                       bnlearn_gs_linear_dict_scores["svm_r"], [0],
                       bnlearn_mmhc_linear_dict_scores["svm_r"], bnlearn_rsmax2_linear_dict_scores["svm_r"],
                       bnlearn_h2pc_linear_dict_scores["svm_r"]], showmeans=True, meanline=True)
    ax[14, 0].set_title('SVM-rbf-Linear', fontsize=fs)
    ax[14, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                          "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[14, 1].boxplot([notears_l2_nonlinear_dict_scores["svm_r"], notears_nonlinear_dict_scores["svm_r"],
                       notears_poisson_nonlinear_dict_scores["svm_r"], bnlearn_nonlinear_dict_scores["svm_r"],
                       bnlearn_tabu_nonlinear_dict_scores["svm_r"], [0], [0], [0],
                       bnlearn_mmhc_nonlinear_dict_scores["svm_r"], bnlearn_rsmax2_nonlinear_dict_scores["svm_r"],
                       bnlearn_h2pc_nonlinear_dict_scores["svm_r"]], showmeans=True, meanline=True)
    ax[14, 1].set_title('SVM-rbf-Nonlinear', fontsize=fs)
    ax[14, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                          "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[14, 2].boxplot([notears_l2_sparse_dict_scores["svm_r"], notears_sparse_dict_scores["svm_r"],
                       notears_poisson_sparse_dict_scores["svm_r"], bnlearn_sparse_dict_scores["svm_r"],
                       bnlearn_tabu_sparse_dict_scores["svm_r"], [0], [0], [0], bnlearn_mmhc_sparse_dict_scores["svm_r"],
                       bnlearn_rsmax2_sparse_dict_scores["svm_r"], bnlearn_h2pc_sparse_dict_scores["svm_r"]],
                      showmeans=True, meanline=True)
    ax[14, 2].set_title('SVM-rbf-Sparse', fontsize=fs)
    ax[14, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                          "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[14, 3].boxplot([notears_l2_dimension_dict_scores["svm_r"], notears_dimension_dict_scores["svm_r"],
                       notears_poisson_dimension_dict_scores["svm_r"], bnlearn_dimension_dict_scores["svm_r"],
                       bnlearn_tabu_dimension_dict_scores["svm_r"], [0], [0], [0],
                       bnlearn_mmhc_dimension_dict_scores["svm_r"], bnlearn_rsmax2_dimension_dict_scores["svm_r"],
                       bnlearn_h2pc_dimension_dict_scores["svm_r"]], showmeans=True, meanline=True)
    ax[14, 3].set_title('SVM-rbf-Dimensional', fontsize=fs)
    ax[14, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                          "BN_RSMAX2", "BN_H2PC"], rotation=90)

    ax[15, 0].boxplot([notears_l2_linear_dict_scores["knn"], notears_linear_dict_scores["knn"],notears_poisson_linear_dict_scores["knn"], bnlearn_linear_dict_scores["knn"],bnlearn_tabu_linear_dict_scores["knn"], bnlearn_pc_linear_dict_scores["knn"],bnlearn_gs_linear_dict_scores["knn"], [0],bnlearn_mmhc_linear_dict_scores["knn"], bnlearn_rsmax2_linear_dict_scores["knn"],bnlearn_h2pc_linear_dict_scores["knn"]], showmeans=True, meanline=True)
    ax[15, 0].set_title('KNN-uniform-Linear', fontsize=fs)
    ax[15, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[15, 1].boxplot([notears_l2_nonlinear_dict_scores["knn"], notears_nonlinear_dict_scores["knn"],notears_poisson_nonlinear_dict_scores["knn"], bnlearn_nonlinear_dict_scores["knn"],bnlearn_tabu_nonlinear_dict_scores["knn"], [0],[0], [0],bnlearn_mmhc_nonlinear_dict_scores["knn"], bnlearn_rsmax2_nonlinear_dict_scores["knn"],bnlearn_h2pc_nonlinear_dict_scores["knn"]],  showmeans=True, meanline=True)
    ax[15, 1].set_title('KNN-uniform-Nonlinear', fontsize=fs)
    ax[15, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[15, 2].boxplot([notears_l2_sparse_dict_scores["knn"], notears_sparse_dict_scores["knn"],notears_poisson_sparse_dict_scores["knn"], bnlearn_sparse_dict_scores["knn"],bnlearn_tabu_sparse_dict_scores["knn"], [0],[0], [0],bnlearn_mmhc_sparse_dict_scores["knn"], bnlearn_rsmax2_sparse_dict_scores["knn"],bnlearn_h2pc_sparse_dict_scores["knn"]],  showmeans=True, meanline=True)
    ax[15, 2].set_title('KNN-uniform-Sparse', fontsize=fs)
    ax[15, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[15, 3].boxplot([notears_l2_dimension_dict_scores["knn"], notears_dimension_dict_scores["knn"],notears_poisson_dimension_dict_scores["knn"], bnlearn_dimension_dict_scores["knn"],bnlearn_tabu_dimension_dict_scores["knn"], [0],[0], [0],bnlearn_mmhc_dimension_dict_scores["knn"], bnlearn_rsmax2_dimension_dict_scores["knn"],bnlearn_h2pc_dimension_dict_scores["knn"]], showmeans=True, meanline=True)
    ax[15, 3].set_title('KNN-uniform-Dimensional', fontsize=fs)
    ax[15, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ["NT_L2", "NT_Log","NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC", "BN_RSMAX2", "BN_H2PC"], rotation=90)

    ax[16, 0].boxplot([notears_l2_linear_dict_scores["knn_d"], notears_linear_dict_scores["knn_d"],
                       notears_poisson_linear_dict_scores["knn_d"], bnlearn_linear_dict_scores["knn_d"],
                       bnlearn_tabu_linear_dict_scores["knn_d"], bnlearn_pc_linear_dict_scores["knn_d"],
                       bnlearn_gs_linear_dict_scores["knn_d"], [0],
                       bnlearn_mmhc_linear_dict_scores["knn_d"], bnlearn_rsmax2_linear_dict_scores["knn_d"],
                       bnlearn_h2pc_linear_dict_scores["knn_d"]], showmeans=True, meanline=True)
    ax[16, 0].set_title('KNN-distance-Linear', fontsize=fs)
    ax[16, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                          "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[16, 1].boxplot([notears_l2_nonlinear_dict_scores["knn_d"], notears_nonlinear_dict_scores["knn_d"],
                       notears_poisson_nonlinear_dict_scores["knn_d"], bnlearn_nonlinear_dict_scores["knn_d"],
                       bnlearn_tabu_nonlinear_dict_scores["knn_d"], [0], [0], [0],
                       bnlearn_mmhc_nonlinear_dict_scores["knn_d"], bnlearn_rsmax2_nonlinear_dict_scores["knn_d"],
                       bnlearn_h2pc_nonlinear_dict_scores["knn_d"]], showmeans=True, meanline=True)
    ax[16, 1].set_title('KNN-distance-Nonlinear', fontsize=fs)
    ax[16, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                          "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[16, 2].boxplot([notears_l2_sparse_dict_scores["knn_d"], notears_sparse_dict_scores["knn_d"],
                       notears_poisson_sparse_dict_scores["knn_d"], bnlearn_sparse_dict_scores["knn_d"],
                       bnlearn_tabu_sparse_dict_scores["knn_d"], [0], [0], [0], bnlearn_mmhc_sparse_dict_scores["knn_d"],
                       bnlearn_rsmax2_sparse_dict_scores["knn_d"], bnlearn_h2pc_sparse_dict_scores["knn_d"]],
                      showmeans=True, meanline=True)
    ax[16, 2].set_title('KNN-distance-Sparse', fontsize=fs)
    ax[16, 2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                          "BN_RSMAX2", "BN_H2PC"], rotation=90)
    ax[16, 3].boxplot([notears_l2_dimension_dict_scores["knn_d"], notears_dimension_dict_scores["knn_d"],
                       notears_poisson_dimension_dict_scores["knn_d"], bnlearn_dimension_dict_scores["knn_d"],
                       bnlearn_tabu_dimension_dict_scores["knn_d"], [0], [0], [0],
                       bnlearn_mmhc_dimension_dict_scores["knn_d"], bnlearn_rsmax2_dimension_dict_scores["knn_d"],
                       bnlearn_h2pc_dimension_dict_scores["knn_d"]], showmeans=True, meanline=True)
    ax[16, 3].set_title('KNN-distance-Dimensional', fontsize=fs)
    ax[16, 3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         ["NT_L2", "NT_Log", "NT_Poi", "BN_HC", "BN_Tabu", "BN_PC", "BN_GS", "BN_IAMB", "BN_MMHC",
                          "BN_RSMAX2", "BN_H2PC"], rotation=90)

    #ax.set(ylim=(0,1), yticks=np.arange(0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
    fig.tight_layout()
    plt.savefig('pipeline_summary_benchmark_boxplots.png')
    plt.show()


    # Group by figure
    labels = ['DT_G', 'DT_E', 'RF_G', 'RF_E', 'LR', 'LR_L1', 'LR_L2', 'LR_E', 'NB_B', 'NB_G', 'NB_M', 'NB_C', 'SVM_S',
              'SVM_P', 'SVM_R', 'KNN_W', 'KNN_D']
    bn_means = [round(mean(bnlearn_linear_dict_scores["dt"]),2), round(mean(bnlearn_linear_dict_scores["dt_e"]),2), round(mean(bnlearn_linear_dict_scores["rf"]),2), round(mean(bnlearn_linear_dict_scores["rf_e"]),2), round(mean(bnlearn_linear_dict_scores["lr"]),2), round(mean(bnlearn_linear_dict_scores["lr_l1"]),2), round(mean(bnlearn_linear_dict_scores["lr_l2"]),2), round(mean(bnlearn_linear_dict_scores["lr_e"]),2), round(mean(bnlearn_linear_dict_scores["nb"]),2), round(mean(bnlearn_linear_dict_scores["nb_g"]),2), round(mean(bnlearn_linear_dict_scores["nb_m"]),2), round(mean(bnlearn_linear_dict_scores["nb_c"]),2), round(mean(bnlearn_linear_dict_scores["svm"]),2), round(mean(bnlearn_linear_dict_scores["svm_po"]),2), round(mean(bnlearn_linear_dict_scores["svm_r"]),2), round(mean(bnlearn_linear_dict_scores["knn"]),2), round(mean(bnlearn_linear_dict_scores["knn_d"]),2)]
    nt_means = [round(mean(notears_linear_dict_scores["dt"]),2), round(mean(notears_linear_dict_scores["dt_e"]),2), round(mean(notears_linear_dict_scores["rf"]),2), round(mean(notears_linear_dict_scores["rf_e"]),2), round(mean(notears_linear_dict_scores["lr"]),2), round(mean(notears_linear_dict_scores["lr_l1"]),2), round(mean(notears_linear_dict_scores["lr_l2"]),2), round(mean(notears_linear_dict_scores["lr_e"]),2), round(mean(notears_linear_dict_scores["nb"]),2), round(mean(notears_linear_dict_scores["nb_g"]),2), round(mean(notears_linear_dict_scores["nb_m"]),2), round(mean(notears_linear_dict_scores["nb_c"]),2), round(mean(notears_linear_dict_scores["svm"]),2), round(mean(notears_linear_dict_scores["svm_po"]),2), round(mean(notears_linear_dict_scores["svm_r"]),2), round(mean(notears_linear_dict_scores["knn"]),2), round(mean(notears_linear_dict_scores["knn_d"]),2)]

    x = np.arange(len(labels))  # the label locations
    width = 2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, bn_means, width, label='NO')
    rects2 = ax.bar(x + width / 2, nt_means, width, label='NO_TEARS')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Linear Problem - Performance by library on ML technique')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.set_size_inches(15, 15)
    fig.tight_layout()
    plt.savefig('pipeline_summary_benchmark_by_library_bargraph.png')
    plt.show()

write_real_to_figures()

def prediction_real_learned():
    print("#### SimCal Real/Learned-world Predictions ####")

    print("-- Exact (1-1) max(rank) output")
    real_linear_workflows = {'Decision Tree (gini)': max(real_linear_dt_scores), 'Decision Tree (entropy)': max(real_linear_dt_entropy_scores), 'Random Forest (gini)': max(real_linear_rf_scores), 'Random Forest (entropy)': max(real_linear_rf_entropy_scores),'Logistic Regression (none)': max(real_linear_lr_scores), 'Logistic Regression (l1)': max(real_linear_lr_l1_scores), 'Logistic Regression (l2)': max(real_linear_lr_l2_scores), 'Logistic Regression (elasticnet)': max(real_linear_lr_elastic_scores), 'Naive Bayes (bernoulli)': max(real_linear_gb_scores), 'Naive Bayes (multinomial)': max(real_linear_gb_multi_scores), 'Naive Bayes (gaussian)': max(real_linear_gb_gaussian_scores), 'Naive Bayes (complement)': max(real_linear_gb_complement_scores), 'Support Vector Machine (sigmoid)': max(real_linear_svm_scores), 'Support Vector Machine (polynomial)': max(real_linear_svm_poly_scores), 'Support Vector Machine (rbf)': max(real_linear_svm_rbf_scores), 'K Nearest Neighbor (uniform)': max(real_linear_knn_scores), 'K Nearest Neighbor (distance)': max(real_linear_knn_distance_scores)}
    top_real_linear = max(real_linear_workflows, key=real_linear_workflows.get)
    print("Real world - Linear problem, Prediction: ", top_real_linear)
    sim_linear_workflows = {'BN Decision Tree (gini)': max(bnlearn_linear_dict_scores["dt"]), 'BN Decision Tree (entropy)': max(bnlearn_linear_dict_scores["dt_e"]),'NT Decision Tree (gini)': max(notears_linear_dict_scores["dt"]),'NT Decision Tree (entropy)': max(notears_linear_dict_scores["dt_e"]), 'BN Random Forest (gini)': max(bnlearn_linear_dict_scores["rf"]), 'BN Random Forest (entropy)': max(bnlearn_linear_dict_scores["rf_e"]),'NT Random Forest (gini)': max(notears_linear_dict_scores["rf"]),'NT Random Forest (entropy)': max(notears_linear_dict_scores["rf_e"]),'BN Logistic Regression (none)': max(bnlearn_linear_dict_scores["lr"]),'BN Logistic Regression (l1)': max(bnlearn_linear_dict_scores["lr_l1"]),'BN Logistic Regression (l2)': max(bnlearn_linear_dict_scores["lr_l2"]),'BN Logistic Regression (elastic)': max(bnlearn_linear_dict_scores["lr_e"]), 'NT Logistic Regression (none)': max(notears_linear_dict_scores["lr"]),  'NT Logistic Regression (l1)': max(notears_linear_dict_scores["lr_l1"]), 'NT Logistic Regression (l2)': max(notears_linear_dict_scores["lr_l2"]), 'NT Logistic Regression (elastic)': max(notears_linear_dict_scores["lr_e"]),'BN Naive Bayes (bernoulli)': max(bnlearn_linear_dict_scores["nb"]),'BN Naive Bayes (gaussian)': max(bnlearn_linear_dict_scores["nb_g"]),'BN Naive Bayes (multinomial)': max(bnlearn_linear_dict_scores["nb_m"]),'BN Naive Bayes (complement)': max(bnlearn_linear_dict_scores["nb_c"]), 'NT Naive Bayes (bernoulli)': max(notears_linear_dict_scores["nb"]),'NT Naive Bayes (gaussian)': max(notears_linear_dict_scores["nb_g"]),'NT Naive Bayes (multinomial)': max(notears_linear_dict_scores["nb_m"]),'NT Naive Bayes (complement)': max(notears_linear_dict_scores["nb_c"]), 'BN Support Vector Machine (sigmoid)': max(bnlearn_linear_dict_scores["svm"]), 'BN Support Vector Machine (polynomial)': max(bnlearn_linear_dict_scores["svm_po"]), 'BN Support Vector Machine (rbf)': max(bnlearn_linear_dict_scores["svm_r"]), 'NT Support Vector Machine (sigmoid)': max(notears_linear_dict_scores["svm"]),'NT Support Vector Machine (polynomial)': max(notears_linear_dict_scores["svm_po"]),'NT Support Vector Machine (rbf)': max(notears_linear_dict_scores["svm_r"]), 'BN K Nearest Neighbor (weight)': max(bnlearn_linear_dict_scores["knn"]),'BN K Nearest Neighbor (distance)': max(bnlearn_linear_dict_scores["knn_d"]),'NT K Nearest Neighbor (weight)': max(notears_linear_dict_scores["knn"]), 'NT K Nearest Neighbor (distance)': max(notears_linear_dict_scores["knn_d"])}
    top_learned_linear = max(sim_linear_workflows, key=sim_linear_workflows.get)
    print("Learned world - Linear problem, Prediction: ", top_learned_linear)

    real_nonlinear_workflows = {'Decision Tree (gini)': max(real_nonlinear_dt_scores),
                             'Decision Tree (entropy)': max(real_nonlinear_dt_entropy_scores),
                             'Random Forest (gini)': max(real_nonlinear_rf_scores),
                             'Random Forest (entropy)': max(real_nonlinear_rf_entropy_scores),
                             'Logistic Regression (none)': max(real_nonlinear_lr_scores),
                             'Logistic Regression (l1)': max(real_nonlinear_lr_l1_scores),
                             'Logistic Regression (l2)': max(real_nonlinear_lr_l2_scores),
                             'Logistic Regression (elasticnet)': max(real_nonlinear_lr_elastic_scores),
                             'Naive Bayes (bernoulli)': max(real_nonlinear_gb_scores),
                             'Naive Bayes (multinomial)': max(real_nonlinear_gb_multi_scores),
                             'Naive Bayes (gaussian)': max(real_nonlinear_gb_gaussian_scores),
                             'Naive Bayes (complement)': max(real_nonlinear_gb_complement_scores),
                             'Support Vector Machine (sigmoid)': max(real_nonlinear_svm_scores),
                             'Support Vector Machine (polynomial)': max(real_nonlinear_svm_poly_scores),
                             'Support Vector Machine (rbf)': max(real_nonlinear_svm_rbf_scores),
                             'K Nearest Neighbor (uniform)': max(real_nonlinear_knn_scores),
                             'K Nearest Neighbor (distance)': max(real_nonlinear_knn_distance_scores)}
    top_real_nonlinear = max(real_nonlinear_workflows, key=real_nonlinear_workflows.get)
    print("Real world - Nonlinear problem, Prediction: ", top_real_nonlinear)
    sim_nonlinear_workflows = {'BN Decision Tree (gini)': max(bnlearn_nonlinear_dict_scores["dt"]),
                            'BN Decision Tree (entropy)': max(bnlearn_nonlinear_dict_scores["dt_e"]),
                            'NT Decision Tree (gini)': max(notears_nonlinear_dict_scores["dt"]),
                            'NT Decision Tree (entropy)': max(notears_nonlinear_dict_scores["dt_e"]),
                            'BN Random Forest (gini)': max(bnlearn_nonlinear_dict_scores["rf"]),
                            'BN Random Forest (entropy)': max(bnlearn_nonlinear_dict_scores["rf_e"]),
                            'NT Random Forest (gini)': max(notears_nonlinear_dict_scores["rf"]),
                            'NT Random Forest (entropy)': max(notears_nonlinear_dict_scores["rf_e"]),
                            'BN Logistic Regression (none)': max(bnlearn_nonlinear_dict_scores["lr"]),
                            'BN Logistic Regression (l1)': max(bnlearn_nonlinear_dict_scores["lr_l1"]),
                            'BN Logistic Regression (l2)': max(bnlearn_nonlinear_dict_scores["lr_l2"]),
                            'BN Logistic Regression (elastic)': max(bnlearn_nonlinear_dict_scores["lr_e"]),
                            'NT Logistic Regression (none)': max(notears_nonlinear_dict_scores["lr"]),
                            'NT Logistic Regression (l1)': max(notears_nonlinear_dict_scores["lr_l1"]),
                            'NT Logistic Regression (l2)': max(notears_nonlinear_dict_scores["lr_l2"]),
                            'NT Logistic Regression (elastic)': max(notears_nonlinear_dict_scores["lr_e"]),
                            'BN Naive Bayes (bernoulli)': max(bnlearn_nonlinear_dict_scores["nb"]),
                            'BN Naive Bayes (gaussian)': max(bnlearn_nonlinear_dict_scores["nb_g"]),
                            'BN Naive Bayes (multinomial)': max(bnlearn_nonlinear_dict_scores["nb_m"]),
                            'BN Naive Bayes (complement)': max(bnlearn_nonlinear_dict_scores["nb_c"]),
                            'NT Naive Bayes (bernoulli)': max(notears_nonlinear_dict_scores["nb"]),
                            'NT Naive Bayes (gaussian)': max(notears_nonlinear_dict_scores["nb_g"]),
                            'NT Naive Bayes (multinomial)': max(notears_nonlinear_dict_scores["nb_m"]),
                            'NT Naive Bayes (complement)': max(notears_nonlinear_dict_scores["nb_c"]),
                            'BN Support Vector Machine (sigmoid)': max(bnlearn_nonlinear_dict_scores["svm"]),
                            'BN Support Vector Machine (polynomial)': max(bnlearn_nonlinear_dict_scores["svm_po"]),
                            'BN Support Vector Machine (rbf)': max(bnlearn_nonlinear_dict_scores["svm_r"]),
                            'NT Support Vector Machine (sigmoid)': max(notears_nonlinear_dict_scores["svm"]),
                            'NT Support Vector Machine (polynomial)': max(notears_nonlinear_dict_scores["svm_po"]),
                            'NT Support Vector Machine (rbf)': max(notears_nonlinear_dict_scores["svm_r"]),
                            'BN K Nearest Neighbor (weight)': max(bnlearn_nonlinear_dict_scores["knn"]),
                            'BN K Nearest Neighbor (distance)': max(bnlearn_nonlinear_dict_scores["knn_d"]),
                            'NT K Nearest Neighbor (weight)': max(notears_nonlinear_dict_scores["knn"]),
                            'NT K Nearest Neighbor (distance)': max(notears_nonlinear_dict_scores["knn_d"])}
    top_learned_nonlinear = max(sim_nonlinear_workflows, key=sim_nonlinear_workflows.get)
    print("Learned world - Nonlinear problem, Prediction: ", top_learned_nonlinear)

    real_sparse_workflows = {'Decision Tree (gini)': max(real_sparse_dt_scores),
                             'Decision Tree (entropy)': max(real_sparse_dt_entropy_scores),
                             'Random Forest (gini)': max(real_sparse_rf_scores),
                             'Random Forest (entropy)': max(real_sparse_rf_entropy_scores),
                             'Logistic Regression (none)': max(real_sparse_lr_scores),
                             'Logistic Regression (l1)': max(real_sparse_lr_l1_scores),
                             'Logistic Regression (l2)': max(real_sparse_lr_l2_scores),
                             'Logistic Regression (elasticnet)': max(real_sparse_lr_elastic_scores),
                             'Naive Bayes (bernoulli)': max(real_sparse_gb_scores),
                             'Naive Bayes (multinomial)': max(real_sparse_gb_multi_scores),
                             'Naive Bayes (gaussian)': max(real_sparse_gb_gaussian_scores),
                             'Naive Bayes (complement)': max(real_sparse_gb_complement_scores),
                             'Support Vector Machine (sigmoid)': max(real_sparse_svm_scores),
                             'Support Vector Machine (polynomial)': max(real_sparse_svm_poly_scores),
                             'Support Vector Machine (rbf)': max(real_sparse_svm_rbf_scores),
                             'K Nearest Neighbor (uniform)': max(real_sparse_knn_scores),
                             'K Nearest Neighbor (distance)': max(real_sparse_knn_distance_scores)}
    top_real_sparse = max(real_sparse_workflows, key=real_sparse_workflows.get)
    print("Real world - Sparse problem, Prediction: ", top_real_sparse)
    sim_sparse_workflows = {'BN Decision Tree (gini)': max(bnlearn_sparse_dict_scores["dt"]),
                            'BN Decision Tree (entropy)': max(bnlearn_sparse_dict_scores["dt_e"]),
                            'NT Decision Tree (gini)': max(notears_sparse_dict_scores["dt"]),
                            'NT Decision Tree (entropy)': max(notears_sparse_dict_scores["dt_e"]),
                            'BN Random Forest (gini)': max(bnlearn_sparse_dict_scores["rf"]),
                            'BN Random Forest (entropy)': max(bnlearn_sparse_dict_scores["rf_e"]),
                            'NT Random Forest (gini)': max(notears_sparse_dict_scores["rf"]),
                            'NT Random Forest (entropy)': max(notears_sparse_dict_scores["rf_e"]),
                            'BN Logistic Regression (none)': max(bnlearn_sparse_dict_scores["lr"]),
                            'BN Logistic Regression (l1)': max(bnlearn_sparse_dict_scores["lr_l1"]),
                            'BN Logistic Regression (l2)': max(bnlearn_sparse_dict_scores["lr_l2"]),
                            'BN Logistic Regression (elastic)': max(bnlearn_sparse_dict_scores["lr_e"]),
                            'NT Logistic Regression (none)': max(notears_sparse_dict_scores["lr"]),
                            'NT Logistic Regression (l1)': max(notears_sparse_dict_scores["lr_l1"]),
                            'NT Logistic Regression (l2)': max(notears_sparse_dict_scores["lr_l2"]),
                            'NT Logistic Regression (elastic)': max(notears_sparse_dict_scores["lr_e"]),
                            'BN Naive Bayes (bernoulli)': max(bnlearn_sparse_dict_scores["nb"]),
                            'BN Naive Bayes (gaussian)': max(bnlearn_sparse_dict_scores["nb_g"]),
                            'BN Naive Bayes (multinomial)': max(bnlearn_sparse_dict_scores["nb_m"]),
                            'BN Naive Bayes (complement)': max(bnlearn_sparse_dict_scores["nb_c"]),
                            'NT Naive Bayes (bernoulli)': max(notears_sparse_dict_scores["nb"]),
                            'NT Naive Bayes (gaussian)': max(notears_sparse_dict_scores["nb_g"]),
                            'NT Naive Bayes (multinomial)': max(notears_sparse_dict_scores["nb_m"]),
                            'NT Naive Bayes (complement)': max(notears_sparse_dict_scores["nb_c"]),
                            'BN Support Vector Machine (sigmoid)': max(bnlearn_sparse_dict_scores["svm"]),
                            'BN Support Vector Machine (polynomial)': max(bnlearn_sparse_dict_scores["svm_po"]),
                            'BN Support Vector Machine (rbf)': max(bnlearn_sparse_dict_scores["svm_r"]),
                            'NT Support Vector Machine (sigmoid)': max(notears_sparse_dict_scores["svm"]),
                            'NT Support Vector Machine (polynomial)': max(notears_sparse_dict_scores["svm_po"]),
                            'NT Support Vector Machine (rbf)': max(notears_sparse_dict_scores["svm_r"]),
                            'BN K Nearest Neighbor (weight)': max(bnlearn_sparse_dict_scores["knn"]),
                            'BN K Nearest Neighbor (distance)': max(bnlearn_sparse_dict_scores["knn_d"]),
                            'NT K Nearest Neighbor (weight)': max(notears_sparse_dict_scores["knn"]),
                            'NT K Nearest Neighbor (distance)': max(notears_sparse_dict_scores["knn_d"])}
    top_learned_sparse = max(sim_sparse_workflows, key=sim_sparse_workflows.get)
    print("Learned world - Sparse problem, Prediction: ", top_learned_sparse)

    real_dimension_workflows = {'Decision Tree (gini)': max(real_dimension_dt_scores),
                             'Decision Tree (entropy)': max(real_dimension_dt_entropy_scores),
                             'Random Forest (gini)': max(real_dimension_rf_scores),
                             'Random Forest (entropy)': max(real_dimension_rf_entropy_scores),
                             'Logistic Regression (none)': max(real_dimension_lr_scores),
                             'Logistic Regression (l1)': max(real_dimension_lr_l1_scores),
                             'Logistic Regression (l2)': max(real_dimension_lr_l2_scores),
                             'Logistic Regression (elasticnet)': max(real_dimension_lr_elastic_scores),
                             'Naive Bayes (bernoulli)': max(real_dimension_gb_scores),
                             'Naive Bayes (multinomial)': max(real_dimension_gb_multi_scores),
                             'Naive Bayes (gaussian)': max(real_dimension_gb_gaussian_scores),
                             'Naive Bayes (complement)': max(real_dimension_gb_complement_scores),
                             'Support Vector Machine (sigmoid)': max(real_dimension_svm_scores),
                             'Support Vector Machine (polynomial)': max(real_dimension_svm_poly_scores),
                             'Support Vector Machine (rbf)': max(real_dimension_svm_rbf_scores),
                             'K Nearest Neighbor (uniform)': max(real_dimension_knn_scores),
                             'K Nearest Neighbor (distance)': max(real_dimension_knn_distance_scores)}
    top_real_dimension = max(real_dimension_workflows, key=real_dimension_workflows.get)
    print("Real world - Dimensional problem, Prediction: ", top_real_dimension)
    sim_dimension_workflows = {'BN Decision Tree (gini)': max(bnlearn_dimension_dict_scores["dt"]),
                            'BN Decision Tree (entropy)': max(bnlearn_dimension_dict_scores["dt_e"]),
                            'NT Decision Tree (gini)': max(notears_dimension_dict_scores["dt"]),
                            'NT Decision Tree (entropy)': max(notears_dimension_dict_scores["dt_e"]),
                            'BN Random Forest (gini)': max(bnlearn_dimension_dict_scores["rf"]),
                            'BN Random Forest (entropy)': max(bnlearn_dimension_dict_scores["rf_e"]),
                            'NT Random Forest (gini)': max(notears_dimension_dict_scores["rf"]),
                            'NT Random Forest (entropy)': max(notears_dimension_dict_scores["rf_e"]),
                            'BN Logistic Regression (none)': max(bnlearn_dimension_dict_scores["lr"]),
                            'BN Logistic Regression (l1)': max(bnlearn_dimension_dict_scores["lr_l1"]),
                            'BN Logistic Regression (l2)': max(bnlearn_dimension_dict_scores["lr_l2"]),
                            'BN Logistic Regression (elastic)': max(bnlearn_dimension_dict_scores["lr_e"]),
                            'NT Logistic Regression (none)': max(notears_dimension_dict_scores["lr"]),
                            'NT Logistic Regression (l1)': max(notears_dimension_dict_scores["lr_l1"]),
                            'NT Logistic Regression (l2)': max(notears_dimension_dict_scores["lr_l2"]),
                            'NT Logistic Regression (elastic)': max(notears_dimension_dict_scores["lr_e"]),
                            'BN Naive Bayes (bernoulli)': max(bnlearn_dimension_dict_scores["nb"]),
                            'BN Naive Bayes (gaussian)': max(bnlearn_dimension_dict_scores["nb_g"]),
                            'BN Naive Bayes (multinomial)': max(bnlearn_dimension_dict_scores["nb_m"]),
                            'BN Naive Bayes (complement)': max(bnlearn_dimension_dict_scores["nb_c"]),
                            'NT Naive Bayes (bernoulli)': max(notears_dimension_dict_scores["nb"]),
                            'NT Naive Bayes (gaussian)': max(notears_dimension_dict_scores["nb_g"]),
                            'NT Naive Bayes (multinomial)': max(notears_dimension_dict_scores["nb_m"]),
                            'NT Naive Bayes (complement)': max(notears_dimension_dict_scores["nb_c"]),
                            'BN Support Vector Machine (sigmoid)': max(bnlearn_dimension_dict_scores["svm"]),
                            'BN Support Vector Machine (polynomial)': max(bnlearn_dimension_dict_scores["svm_po"]),
                            'BN Support Vector Machine (rbf)': max(bnlearn_dimension_dict_scores["svm_r"]),
                            'NT Support Vector Machine (sigmoid)': max(notears_dimension_dict_scores["svm"]),
                            'NT Support Vector Machine (polynomial)': max(notears_dimension_dict_scores["svm_po"]),
                            'NT Support Vector Machine (rbf)': max(notears_dimension_dict_scores["svm_r"]),
                            'BN K Nearest Neighbor (weight)': max(bnlearn_dimension_dict_scores["knn"]),
                            'BN K Nearest Neighbor (distance)': max(bnlearn_dimension_dict_scores["knn_d"]),
                            'NT K Nearest Neighbor (weight)': max(notears_dimension_dict_scores["knn"]),
                            'NT K Nearest Neighbor (distance)': max(notears_dimension_dict_scores["knn_d"])}
    top_learned_dimension = max(sim_dimension_workflows, key=sim_dimension_workflows.get)
    print("Learned world - Dimensional problem, Prediction: ", top_learned_dimension)

    #print("Relative (point-based) rank output")

    #workflows = {'Decision Tree': real_nonlinear_dt, 'Random Forest': real_nonlinear_rf,
    #             'Logistic Regression': real_nonlinear_lr, 'Naive Bayes': real_nonlinear_gb,
    #             'Support Vector Machine': real_nonlinear_svm, 'K Nearest Neighbor': real_nonlinear_knn}
    #top_real = max(workflows, key=workflows.get)
    #print("Real world - Non-Linear ground truth, Prediction ", top_real)
    #workflows = {'BN Decision Tree': bnlearn_nonlinear_dt, 'NT Decision Tree': notears_nonlinear_dt,
    #             'BN Random Forest': bnlearn_nonlinear_rf, 'NT Random Forest': notears_nonlinear_rf,
    #             'BN Logistic Regression': bnlearn_nonlinear_lr, 'NT Logistic Regression': notears_nonlinear_lr,
    #             'BN Naive Bayes': bnlearn_nonlinear_nb, 'NT Naive Bayes': notears_nonlinear_nb,
    #             'BN Support Vector Machine': bnlearn_nonlinear_svm, 'NT Support Vector Machine': notears_nonlinear_svm,
    #             'BN K Nearest Neighbor': bnlearn_nonlinear_knn, 'NT K Nearest Neighbor': notears_nonlinear_knn}
    #top_learned = max(workflows, key=workflows.get)
    #print("Learned world - Non-Linear, Prediction ", top_learned)
    #workflows = {'Decision Tree': real_sparse_dt, 'Random Forest': real_sparse_rf,
    #             'Logistic Regression': real_sparse_lr, 'Naive Bayes': real_sparse_gb,
    #             'Support Vector Machine': real_sparse_svm, 'K Nearest Neighbor': real_sparse_knn}
    #top_real = max(workflows, key=workflows.get)
    #print("Real world - Sparse ground truth, Prediction ", top_real)
    #workflows = {'BN Decision Tree': bnlearn_sparse_dt, 'NT Decision Tree': notears_sparse_dt,
    #             'BN Random Forest': bnlearn_sparse_rf, 'NT Random Forest': notears_sparse_rf,
    #             'BN Logistic Regression': bnlearn_sparse_lr, 'NT Logistic Regression': notears_sparse_lr,
    #             'BN Naive Bayes': bnlearn_sparse_nb, 'NT Naive Bayes': notears_sparse_nb,
    #             'BN Support Vector Machine': bnlearn_sparse_svm, 'NT Support Vector Machine': notears_sparse_svm,
    #             'BN K Nearest Neighbor': bnlearn_sparse_knn, 'NT K Nearest Neighbor': notears_sparse_knn}
    #top_learned = max(workflows, key=workflows.get)
    #print("Learned world - Sparse, Prediction ", top_learned)

real_experiment_summary = pd.read_csv("real_experiments_summary.csv")
real_experiment_summary

learned_experiment_summary = pd.read_csv("simulation_experiments_summary.csv")
learned_experiment_summary

prediction_real_learned()
