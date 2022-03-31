import random

#1 - compact dictionary into a dict or dict 414-436 homer simpson
#3 - Compact a single execution of a pipeline into a class 445-764

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
from sklearn.metrics import accuracy_score
from dagsim.baseDS import Graph, Generic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import simulation_notears
import simulation_bnlearn
import simulation_dagsim
import simulation_models
import simulation_pgmpy
import simulation_pomegranate
from sklearn import metrics
from sklearn import svm

#Save linear, nonlinear, sparse, dimensional training set of the real-world for reproducablity
global pipeline_type
global linear_training
global nonlinear_training
global sparse_training
global dimensional_training

# Attampt at globalising the training set of all pipelines from real world
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#pipeline_type = 2
#nonlinear_training = simulation_dagsim.setup_realworld(pipeline_type, 10000, 5000)
#pipeline_type = 3
#sparse_training = simulation_dagsim.setup_realworld(pipeline_type, 10000, 5000)
#pipeline_type = 4
#dimensional_training = simulation_dagsim.setup_realworld(pipeline_type, 10000, 5000)

# import the saved training and test data from DagSim's real world
def import_real_world_csv(pipeline_type):
    global train_data
    train_data = pd.read_csv("train.csv")
    global train_data_numpy
    train_data_numpy = train_data.to_numpy()
    global x_train
    global y_train
    if(pipeline_type==4):
        x_train = train_data.iloc[:, 0:10].to_numpy().reshape([-1, 10])  # num predictors
        y_train = train_data.iloc[:, 10].to_numpy().reshape([-1]).ravel()  # outcome
    elif(pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        x_train = train_data.iloc[:, 0:4].to_numpy().reshape([-1, 4])  # num predictors
        y_train = train_data.iloc[:, 4].to_numpy().reshape([-1]).ravel()  # outcome

    global test_data
    global x_test
    global y_test
    test_data = pd.read_csv("test.csv")
    if(pipeline_type==4):
        x_test = test_data.iloc[:, 0:10].to_numpy().reshape([-1, 10])
        y_test = test_data.iloc[:, 10].to_numpy().reshape([-1]).ravel()
    elif(pipeline_type==1 or pipeline_type==2 or pipeline_type==3 ):
        x_test = test_data.iloc[:, 0:4].to_numpy().reshape([-1, 4])
        y_test = test_data.iloc[:, 4].to_numpy().reshape([-1]).ravel()

# Evaluate function for all ML techniques in the real-world
def realworld_evaluate(pipeline_type):
    import_real_world_csv(pipeline_type)
    #Decision Tree
    clf = DecisionTreeClassifier(criterion='gini')
    clf = clf.fit(x_train, y_train)
    if(pipeline_type==1):
        global real_linear_dt_scores
        y_pred = clf.predict(x_test)
        real_linear_dt_scores = accuracy_score(y_test, y_pred)
    elif(pipeline_type==2):
        global real_nonlinear_dt_scores
        y_pred = clf.predict(x_test)
        real_nonlinear_dt_scores = accuracy_score(y_test, y_pred)
    elif(pipeline_type==3):
        global real_sparse_dt_scores
        y_pred = clf.predict(x_test)
        real_sparse_dt_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_dt_scores
        y_pred = clf.predict(x_test)
        real_dimension_dt_scores = accuracy_score(y_test, y_pred)
    clf = DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(x_train, y_train)
    if (pipeline_type == 1):
        global real_linear_dt_entropy_scores
        y_pred = clf.predict(x_test)
        real_linear_dt_entropy_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 2):
        global real_nonlinear_dt_entropy_scores
        y_pred = clf.predict(x_test)
        real_nonlinear_dt_entropy_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 3):
        global real_sparse_dt_entropy_scores
        y_pred = clf.predict(x_test)
        real_sparse_dt_entropy_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_dt_entropy_scores
        y_pred = clf.predict(x_test)
        real_dimension_dt_entropy_scores = accuracy_score(y_test, y_pred)
    rf = RandomForestClassifier(criterion='gini')
    rf = rf.fit(x_train, y_train)
    if (pipeline_type == 1):
        global real_linear_rf_scores
        y_pred = rf.predict(x_test)
        real_linear_rf_scores = accuracy_score(y_test, y_pred)
    elif(pipeline_type==2):
        global real_nonlinear_rf_scores
        y_pred = rf.predict(x_test)
        real_nonlinear_rf_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 3):
        global real_sparse_rf_scores
        y_pred = rf.predict(x_test)
        real_sparse_rf_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_rf_scores
        y_pred = rf.predict(x_test)
        real_dimension_rf_scores = accuracy_score(y_test, y_pred)
    rf = RandomForestClassifier(criterion='entropy')
    rf = rf.fit(x_train, y_train)
    if (pipeline_type == 1):
        global real_linear_rf_entropy_scores
        y_pred = rf.predict(x_test)
        real_linear_rf_entropy_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 2):
        global real_nonlinear_rf_entropy_scores
        y_pred = rf.predict(x_test)
        real_nonlinear_rf_entropy_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 3):
        global real_sparse_rf_entropy_scores
        y_pred = rf.predict(x_test)
        real_sparse_rf_entropy_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_rf_entropy_scores
        y_pred = rf.predict(x_test)
        real_dimension_rf_entropy_scores = accuracy_score(y_test, y_pred)
    lr = LogisticRegression(penalty='none')
    lr = lr.fit(x_train, y_train)
    if (pipeline_type == 1):
        global real_linear_lr_scores
        y_pred = lr.predict(x_test)
        real_linear_lr_scores = accuracy_score(y_test, y_pred)
    elif(pipeline_type==2):
        global real_nonlinear_lr_scores
        y_pred = lr.predict(x_test)
        real_nonlinear_lr_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 3):
        global real_sparse_lr_scores
        y_pred = lr.predict(x_test)
        real_sparse_lr_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_lr_scores
        y_pred = lr.predict(x_test)
        real_dimension_lr_scores = accuracy_score(y_test, y_pred)
    lr = LogisticRegression(penalty='l1', solver='liblinear', l1_ratio=1)
    lr = lr.fit(x_train, y_train)
    if (pipeline_type == 1):
        global real_linear_lr_l1_scores
        y_pred = lr.predict(x_test)
        real_linear_lr_l1_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 2):
        global real_nonlinear_lr_l1_scores
        y_pred = lr.predict(x_test)
        real_nonlinear_lr_l1_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 3):
        global real_sparse_lr_l1_scores
        y_pred = lr.predict(x_test)
        real_sparse_lr_l1_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_lr_l1_scores
        y_pred = lr.predict(x_test)
        real_dimension_lr_l1_scores = accuracy_score(y_test, y_pred)
    lr = LogisticRegression(penalty='l2')
    lr = lr.fit(x_train, y_train)
    coef = lr.coef_
    print("This is the coeff ", coef)
    if (pipeline_type == 1):
        global real_linear_lr_l2_scores
        y_pred = lr.predict(x_test)
        real_linear_lr_l2_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 2):
        global real_nonlinear_lr_l2_scores
        y_pred = lr.predict(x_test)
        real_nonlinear_lr_l2_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 3):
        global real_sparse_lr_l2_scores
        y_pred = lr.predict(x_test)
        real_sparse_lr_l2_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_lr_l2_scores
        y_pred = lr.predict(x_test)
        real_dimension_lr_l2_scores = accuracy_score(y_test, y_pred)
    lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
    lr = lr.fit(x_train, y_train)
    if (pipeline_type == 1):
        global real_linear_lr_elastic_scores
        y_pred = lr.predict(x_test)
        real_linear_lr_elastic_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 2):
        global real_nonlinear_lr_elastic_scores
        y_pred = lr.predict(x_test)
        real_nonlinear_lr_elastic_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 3):
        global real_sparse_lr_elastic_scores
        y_pred = lr.predict(x_test)
        real_sparse_lr_elastic_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_lr_elastic_scores
        y_pred = lr.predict(x_test)
        real_dimension_lr_elastic_scores = accuracy_score(y_test, y_pred)
    gnb = BernoulliNB()
    gnb = gnb.fit(x_train, y_train)
    if (pipeline_type == 1):
        global real_linear_gb_scores
        y_pred = gnb.predict(x_test)
        real_linear_gb_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 2):
        global real_nonlinear_gb_scores
        y_pred = gnb.predict(x_test)
        real_nonlinear_gb_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 3):
        global real_sparse_gb_scores
        y_pred = gnb.predict(x_test)
        real_sparse_gb_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_gb_scores
        y_pred = gnb.predict(x_test)
        real_dimension_gb_scores = accuracy_score(y_test, y_pred)
    gnb = GaussianNB()
    gnb = gnb.fit(x_train, y_train)
    if (pipeline_type == 1):
        global real_linear_gb_gaussian_scores
        y_pred = gnb.predict(x_test)
        real_linear_gb_gaussian_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 2):
        global real_nonlinear_gb_gaussian_scores
        y_pred = gnb.predict(x_test)
        real_nonlinear_gb_gaussian_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 3):
        global real_sparse_gb_gaussian_scores
        y_pred = gnb.predict(x_test)
        real_sparse_gb_gaussian_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_gb_gaussian_scores
        y_pred = gnb.predict(x_test)
        real_dimension_gb_gaussian_scores = accuracy_score(y_test, y_pred)
    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(x_train)
    X_test_minmax = min_max_scaler.transform(x_test)
    gnb = MultinomialNB()
    gnb = gnb.fit(X_train_minmax, y_train)
    if (pipeline_type == 1):
        global real_linear_gb_multi_scores
        y_pred = gnb.predict(X_test_minmax)
        real_linear_gb_multi_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 2):
        global real_nonlinear_gb_multi_scores
        y_pred = gnb.predict(X_test_minmax)
        real_nonlinear_gb_multi_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 3):
        global real_sparse_gb_multi_scores
        y_pred = gnb.predict(X_test_minmax)
        real_sparse_gb_multi_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_gb_multi_scores
        y_pred = gnb.predict(X_test_minmax)
        real_dimension_gb_multi_scores = accuracy_score(y_test, y_pred)
    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(x_train)
    X_test_minmax = min_max_scaler.transform(x_test)
    gnb = ComplementNB()
    gnb = gnb.fit(X_train_minmax, y_train)
    if (pipeline_type == 1):
        global real_linear_gb_complement_scores
        y_pred = gnb.predict(X_test_minmax)
        real_linear_gb_complement_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 2):
        global real_nonlinear_gb_complement_scores
        y_pred = gnb.predict(X_test_minmax)
        real_nonlinear_gb_complement_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 3):
        global real_sparse_gb_complement_scores
        y_pred = gnb.predict(X_test_minmax)
        real_sparse_gb_complement_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_gb_complement_scores
        y_pred = gnb.predict(X_test_minmax)
        real_dimension_gb_complement_scores = accuracy_score(y_test, y_pred)
    clf = svm.SVC(kernel="sigmoid")
    clf = clf.fit(x_train, y_train)
    if (pipeline_type == 1):
        global real_linear_svm_scores
        y_pred = clf.predict(x_test)
        real_linear_svm_scores = accuracy_score(y_test, y_pred)
    elif(pipeline_type==2):
        global real_nonlinear_svm_scores
        y_pred = clf.predict(x_test)
        real_nonlinear_svm_scores = accuracy_score(y_test, y_pred)
    elif(pipeline_type==3):
        global real_sparse_svm_scores
        y_pred = clf.predict(x_test)
        real_sparse_svm_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_svm_scores
        y_pred = clf.predict(x_test)
        real_dimension_svm_scores = accuracy_score(y_test, y_pred)
    #clf = svm.SVC(kernel="linear")
    #clf = clf.fit(x_train, y_train)
    #y_pred = clf.predict(x_test)
    #if (pipeline_type == 1):
    #    global real_linear_svm_linear_scores
    #    real_linear_svm_linear_scores = cross_val_score(clf, x_train, y_train, cv=10)
    #elif(pipeline_type==2):
    #    global real_nonlinear_svm_linear_scores
    #    real_nonlinear_svm_linear_scores = cross_val_score(clf, x_train, y_train, cv=10)
    #elif(pipeline_type==3):
    #    global real_sparse_svm_linear_scores
    #    real_sparse_svm_linear_scores = cross_val_score(clf, x_train, y_train, cv=10)
    #elif (pipeline_type == 4):
    #    global real_dimension_svm_linear_scores
    #    real_dimension_svm_linear_scores = cross_val_score(clf, x_train, y_train, cv=10)
    clf = svm.SVC(kernel="poly")
    clf = clf.fit(x_train, y_train)
    if (pipeline_type == 1):
        global real_linear_svm_poly_scores
        y_pred = clf.predict(x_test)
        real_linear_svm_poly_scores = accuracy_score(y_test, y_pred)
    elif(pipeline_type==2):
        global real_nonlinear_svm_poly_scores
        clf.predict(x_test)
        real_nonlinear_svm_poly_scores = accuracy_score(y_test, y_pred)
    elif(pipeline_type==3):
        global real_sparse_svm_poly_scores
        clf.predict(x_test)
        real_sparse_svm_poly_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_svm_poly_scores
        clf.predict(x_test)
        real_dimension_svm_poly_scores = accuracy_score(y_test, y_pred)
    clf = svm.SVC(kernel="rbf")
    clf = clf.fit(x_train, y_train)
    if (pipeline_type == 1):
        global real_linear_svm_rbf_scores
        y_pred = clf.predict(x_test)
        real_linear_svm_rbf_scores = accuracy_score(y_test, y_pred)
    elif(pipeline_type==2):
        global real_nonlinear_svm_rbf_scores
        y_pred = clf.predict(x_test)
        real_nonlinear_svm_rbf_scores = accuracy_score(y_test, y_pred)
    elif(pipeline_type==3):
        global real_sparse_svm_rbf_scores
        y_pred = clf.predict(x_test)
        real_sparse_svm_rbf_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_svm_rbf_scores
        y_pred = clf.predict(x_test)
        real_dimension_svm_rbf_scores = accuracy_score(y_test, y_pred)
#    clf = svm.SVC(kernel="precomputed")
#    clf = clf.fit(x_train, y_train)
#    y_pred = clf.predict(x_test)
#    if (pipeline_type == 1):
#        global real_linear_svm_precomputed_scores
#        real_linear_svm_precomputed_scores = cross_val_score(clf, x_train, y_train, cv=10)
#    elif(pipeline_type==2):
#        global real_nonlinear_svm_precomputed_scores
#        real_nonlinear_svm_precomputed_scores = cross_val_score(clf, x_train, y_train, cv=10)
#    elif(pipeline_type==3):
#        global real_sparse_svm_precomputed_scores
#        real_sparse_svm_precomputed_scores = cross_val_score(clf, x_train, y_train, cv=10)
#    elif (pipeline_type == 4):
#        global real_dimension_svm_precomputed_scores
#        real_dimension_svm_precomputed_scores = cross_val_score(clf, x_train, y_train, cv=10)
    clf = KNeighborsClassifier(weights='uniform')
    clf = clf.fit(x_train, y_train)
    if (pipeline_type == 1):
        global real_linear_knn_scores
        y_pred = clf.predict(x_test)
        real_linear_knn_scores = accuracy_score(y_test, y_pred)
    elif(pipeline_type==2):
        global real_nonlinear_knn_scores
        y_pred = clf.predict(x_test)
        real_nonlinear_knn_scores = accuracy_score(y_test, y_pred)
    elif(pipeline_type==3):
        global real_sparse_knn_scores
        y_pred = clf.predict(x_test)
        real_sparse_knn_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_knn_scores
        y_pred = clf.predict(x_test)
        real_dimension_knn_scores = accuracy_score(y_test, y_pred)
    clf = KNeighborsClassifier(weights='distance')
    clf = clf.fit(x_train, y_train)
    if (pipeline_type == 1):
        global real_linear_knn_distance_scores
        y_pred = clf.predict(x_test)
        real_linear_knn_distance_scores = accuracy_score(y_test, y_pred)
    elif(pipeline_type==2):
        global real_nonlinear_knn_distance_scores
        y_pred = clf.predict(x_test)
        real_nonlinear_knn_distance_scores = accuracy_score(y_test, y_pred)
    elif(pipeline_type==3):
        global real_sparse_knn_distance_scores
        y_pred = clf.predict(x_test)
        real_sparse_knn_distance_scores = accuracy_score(y_test, y_pred)
    elif (pipeline_type == 4):
        global real_dimension_knn_distance_scores
        y_pred = clf.predict(x_test)
        real_dimension_knn_distance_scores = accuracy_score(y_test, y_pred)

print("This is the first occurance of the real-world benchmarks")
realworld_evaluate(pipeline_type)

pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)

realworld_evaluate(pipeline_type)

pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)

realworld_evaluate(pipeline_type)

pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)

realworld_evaluate(pipeline_type)

pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)

# Simulation library structure learning section

print("This is the first occurance of the simulated benchmarks")
simulated_data_train = simulation_notears.notears_setup(train_data_numpy[0:100], 1000, 1000)[0]
simulated_data_test = simulation_notears.notears_setup(train_data_numpy[0:100], 1000, 1000)[1]

#simulation_notears.notears_nonlinear_setup(train_data_numpy[0:100], 10000, 5000)

# import the saved training and test data from the simulation framework's learned world
#def import_simulated_csv():
#    global no_tears_sample_train
#    no_tears_sample_train= pd.read_csv('W_est_train.csv')
#    #global no_tears_sample_test
#    #no_tears_sample_test = pd.read_csv('W_est_test.csv')
#    #global no_tears_nonlinear_sample_train
#    #no_tears_nonlinear_sample_train = pd.read_csv('K_est_train.csv')
#    #global no_tears_nonlinear_sample_test
#    #no_tears_nonlinear_sample_test = pd.read_csv('K_est_test.csv')
#    global bn_learn_sample_train
#    bn_learn_sample_train = pd.read_csv('Z_est_train.csv')
#    #global bn_learn_sample_test
#    #bn_learn_sample_test = pd.read_csv('Z_est_test.csv')
#    global pomegranate_sample_train
#    pomegranate_sample_train = pd.read_csv('X_est_train.csv')
#    global pgmpy_sample_train
#    pgmpy_sample_train = pd.read_csv('V_est_train.csv')

#import_simulated_csv()

def run_learned_workflows(x_train, y_train, x_test, y_test, pipeline_type, alg):
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
notears_linear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "NO TEARS (Logistic)")
notears_linear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "NO TEARS (Logistic)")

pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_notears.notears_setup(train_data_numpy[0:100], 1000, 1000)[0]
simulated_data_test = simulation_notears.notears_setup(train_data_numpy[0:100], 1000, 1000)[1]

notears_nonlinear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "NO TEARS (Logistic)")
notears_nonlinear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4],pipeline_type, "NO TEARS (Logistic)")

pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_notears.notears_setup(train_data_numpy[0:100], 1000, 1000)[0]
simulated_data_test = simulation_notears.notears_setup(train_data_numpy[0:100], 1000, 1000)[1]

notears_sparse_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "NO TEARS (Logistic)")
notears_sparse_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "NO TEARS (Logistic)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_notears.notears_setup(train_data_numpy[0:100], 1000, 1000)[0]
simulated_data_test = simulation_notears.notears_setup(train_data_numpy[0:100], 1000, 1000)[1]

notears_dimension_dict_scores = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], x_test, y_test, pipeline_type, "NO TEARS (Logistic)")
notears_dimension_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], simulated_data_test[:,0:10], simulated_data_test[:,10], pipeline_type, "NO TEARS (Logistic)")

#notears hyperparameter loss function l2
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_notears.notears_setup_b(train_data_numpy[0:100], 1000, 1000)[0]
simulated_data_test = simulation_notears.notears_setup_b(train_data_numpy[0:100], 1000, 1000)[1]
notears_l2_linear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "NO TEARS (L2)")
notears_l2_linear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4],pipeline_type, "NO TEARS (L2)")

pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_notears.notears_setup_b(train_data_numpy[0:100], 1000, 1000)[0]
simulated_data_test = simulation_notears.notears_setup_b(train_data_numpy[0:100], 1000, 1000)[1]

notears_l2_nonlinear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "NO TEARS (L2)")
notears_l2_nonlinear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4],pipeline_type, "NO TEARS (L2)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_notears.notears_setup_b(train_data_numpy[0:100], 1000, 1000)[0]
simulated_data_test = simulation_notears.notears_setup_b(train_data_numpy[0:100], 1000, 1000)[1]

notears_l2_sparse_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "NO TEARS (L2)")
notears_l2_sparse_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "NO TEARS (L2)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_notears.notears_setup_b(train_data_numpy[0:100], 1000, 1000)[0]
simulated_data_test = simulation_notears.notears_setup_b(train_data_numpy[0:100], 1000, 1000)[1]

notears_l2_dimension_dict_scores = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], x_test, y_test, pipeline_type, "NO TEARS (L2)")
notears_l2_dimension_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], simulated_data_test[:,0:10], simulated_data_test[:,10],pipeline_type, "NO TEARS (L2)")

#notears hyperparameter loss function poisson
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_notears.notears_setup_c(train_data_numpy[0:100], 1000, 1000)[0]
simulated_data_test = simulation_notears.notears_setup_c(train_data_numpy[0:100], 1000, 1000)[1]
notears_poisson_linear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "NO TEARS (Poisson)")
notears_poisson_linear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "NO TEARS (Poisson)")

pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_notears.notears_setup_c(train_data_numpy[0:100], 1000, 1000)[0]
simulated_data_test = simulation_notears.notears_setup_c(train_data_numpy[0:100], 1000, 1000)[1]

notears_poisson_nonlinear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "NO TEARS (Poisson)")
notears_poisson_nonlinear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "NO TEARS (Poisson)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_notears.notears_setup_c(train_data_numpy[0:100], 1000, 1000)[0]
simulated_data_test = simulation_notears.notears_setup_c(train_data_numpy[0:100], 1000, 1000)[1]

notears_poisson_sparse_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "NO TEARS (Poisson)")
notears_poisson_sparse_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4],pipeline_type, "NO TEARS (Poisson)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_notears.notears_setup_c(train_data_numpy[0:100], 1000, 1000)[0]
simulated_data_test = simulation_notears.notears_setup_c(train_data_numpy[0:100], 1000, 1000)[1]

notears_poisson_dimension_dict_scores = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], x_test, y_test, pipeline_type, "NO TEARS (Poisson)")
notears_poisson_dimension_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], simulated_data_test[:,0:10], simulated_data_test[:,10],pipeline_type, "NO TEARS (Poisson)")

#bnlearn simulation scoring
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_hc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_bnlearn.bnlearn_setup_hc(train_data[0:100], pipeline_type)[1]

bnlearn_linear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (HC)")
bnlearn_linear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "BN LEARN (HC)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_hc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_bnlearn.bnlearn_setup_hc(train_data[0:100], pipeline_type)[1]

bnlearn_nonlinear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (HC)")
bnlearn_nonlinear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "BN LEARN (HC)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_hc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_bnlearn.bnlearn_setup_hc(train_data[0:100], pipeline_type)[1]

bnlearn_sparse_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (HC)")
bnlearn_sparse_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "BN LEARN (HC)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_hc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_bnlearn.bnlearn_setup_hc(train_data[0:100], pipeline_type)[1]
bnlearn_dimension_dict_scores = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], x_test, y_test, pipeline_type, "BN LEARN (HC)")
bnlearn_dimension_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], simulated_data_test[:,0:10], simulated_data_test[:,10], pipeline_type, "BN LEARN (HC)")

#Run hyperparameter of bnlearn - tabu
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_tabu(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_bnlearn.bnlearn_setup_tabu(train_data[0:100], pipeline_type)[1]

bnlearn_tabu_linear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (TABU)")
bnlearn_tabu_linear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "BN LEARN (TABU)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_tabu(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_bnlearn.bnlearn_setup_tabu(train_data[0:100], pipeline_type)[1]

bnlearn_tabu_nonlinear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (TABU)")
bnlearn_tabu_nonlinear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "BN LEARN (TABU)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_tabu(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_bnlearn.bnlearn_setup_tabu(train_data[0:100], pipeline_type)[1]

bnlearn_tabu_sparse_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (TABU)")
bnlearn_tabu_sparse_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "BN LEARN (TABU)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_tabu(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_bnlearn.bnlearn_setup_tabu(train_data[0:100], pipeline_type)[1]
bnlearn_tabu_dimension_dict_scores = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], x_test, y_test, pipeline_type, "BN LEARN (TABU)")
bnlearn_tabu_dimension_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], simulated_data_test[:,0:10], simulated_data_test[:,10], pipeline_type, "BN LEARN (TABU)")
#end of tabu workflows

#Run hyperparameter of bnlearn - pc
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_pc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_bnlearn.bnlearn_setup_pc(train_data[0:100], pipeline_type)[1]

bnlearn_pc_linear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (PC)")
bnlearn_pc_linear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "BN LEARN (PC)")
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
#pipeline_type = 1
#simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_bnlearn.bnlearn_setup_gs(train_data[0:100], pipeline_type)
#import_simulated_csv()

#bnlearn_gs_linear_dict_scores = run_learned_workflows(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4], pipeline_type, "BN LEARN (GS)")
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
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_mmhc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_bnlearn.bnlearn_setup_mmhc(train_data[0:100], pipeline_type)[1]

bnlearn_mmhc_linear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (MMHC)")
bnlearn_mmhc_linear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "BN LEARN (MMHC)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_mmhc(train_data[0:100], pipeline_type)[0] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
simulated_data_test = simulation_bnlearn.bnlearn_setup_mmhc(train_data[0:100], pipeline_type)[1] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed

bnlearn_mmhc_nonlinear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (MMHC)")
bnlearn_mmhc_nonlinear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "BN LEARN (MMHC)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_mmhc(train_data[0:100], pipeline_type)[0] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
simulated_data_test = simulation_bnlearn.bnlearn_setup_mmhc(train_data[0:100], pipeline_type)[1] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed

bnlearn_mmhc_sparse_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (MMHC)")
bnlearn_mmhc_sparse_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4],pipeline_type, "BN LEARN (MMHC)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_mmhc(train_data[0:100], pipeline_type)[0] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
simulated_data_test = simulation_bnlearn.bnlearn_setup_mmhc(train_data[0:100], pipeline_type)[1] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
bnlearn_mmhc_dimension_dict_scores = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], x_test, y_test, pipeline_type, "BN LEARN (MMHC)")
bnlearn_mmhc_dimension_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], simulated_data_test[:,0:10], simulated_data_test[:,10], pipeline_type, "BN LEARN (MMHC)")
#end of mmhc workflows

#Run hyperparameter of bnlearn - rsmax2
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_rsmax2(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_bnlearn.bnlearn_setup_rsmax2(train_data[0:100], pipeline_type)[1]

bnlearn_rsmax2_linear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (RSMAX2)")
bnlearn_rsmax2_linear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "BN LEARN (RSMAX2)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_rsmax2(train_data[0:100], pipeline_type)[0] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
simulated_data_test = simulation_bnlearn.bnlearn_setup_rsmax2(train_data[0:100], pipeline_type)[1] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed

bnlearn_rsmax2_nonlinear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (RSMAX2)")
bnlearn_rsmax2_nonlinear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "BN LEARN (RSMAX2)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_rsmax2(train_data[0:100], pipeline_type)[0] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
simulated_data_test = simulation_bnlearn.bnlearn_setup_rsmax2(train_data[0:100], pipeline_type)[1] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed

bnlearn_rsmax2_sparse_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (RSMAX2)")
bnlearn_rsmax2_sparse_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "BN LEARN (RSMAX2)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_rsmax2(train_data[0:100], pipeline_type)[0] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
simulated_data_test = simulation_bnlearn.bnlearn_setup_rsmax2(train_data[0:100], pipeline_type)[1] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
bnlearn_rsmax2_dimension_dict_scores = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], x_test, y_test, pipeline_type, "BN LEARN (RSMAX2)")
bnlearn_rsmax2_dimension_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], simulated_data_test[:,0:10], simulated_data_test[:,10], pipeline_type, "BN LEARN (RSMAX2)")
#end of rsmax2 workflows

#Run hyperparameter of bnlearn - h2pc
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_h2pc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_bnlearn.bnlearn_setup_h2pc(train_data[0:100], pipeline_type)[1]

bnlearn_h2pc_linear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (H2PC)")
bnlearn_h2pc_linear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "BN LEARN (H2PC)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_h2pc(train_data[0:100], pipeline_type)[0] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
simulated_data_test = simulation_bnlearn.bnlearn_setup_h2pc(train_data[0:100], pipeline_type)[1] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed

bnlearn_h2pc_nonlinear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (H2PC)")
bnlearn_h2pc_nonlinear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4],pipeline_type, "BN LEARN (H2PC)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_h2pc(train_data[0:100], pipeline_type)[0] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
simulated_data_test = simulation_bnlearn.bnlearn_setup_h2pc(train_data[0:100], pipeline_type)[1] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed

bnlearn_h2pc_sparse_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "BN LEARN (H2PC)")
bnlearn_h2pc_sparse_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "BN LEARN (H2PC)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_bnlearn.bnlearn_setup_h2pc(train_data[0:100], pipeline_type)[0] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
simulated_data_test = simulation_bnlearn.bnlearn_setup_h2pc(train_data[0:100], pipeline_type)[1] #rpy2.rinterface_lib.embedded.RRuntimeError: Error in bn.fit(my_bn, databn) : the graph is only partially directed
bnlearn_h2pc_dimension_dict_scores = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], x_test, y_test, pipeline_type, "BN LEARN (H2PC)")
bnlearn_h2pc_dimension_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], simulated_data_test[:,0:10], simulated_data_test[:,10], pipeline_type, "BN LEARN (H2PC)")
#end of h2pc workflows

#pomegranate simulation scoring
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pomegranate.pomegranate_setup(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pomegranate.pomegranate_setup(train_data[0:100], pipeline_type)[1]

pomegranate_exact_linear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "POMEGRANATE (EXACT)")
pomegranate_exact_linear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "POMEGRANATE (EXACT)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pomegranate.pomegranate_setup(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pomegranate.pomegranate_setup(train_data[0:100], pipeline_type)[1]

pomegranate_exact_nonlinear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "POMEGRANATE (EXACT)")
pomegranate_exact_nonlinear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "POMEGRANATE (EXACT)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pomegranate.pomegranate_setup(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pomegranate.pomegranate_setup(train_data[0:100], pipeline_type)[1]

pomegranate_exact_sparse_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "POMEGRANATE (EXACT)")
pomegranate_exact_sparse_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "POMEGRANATE (EXACT)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pomegranate.pomegranate_setup(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pomegranate.pomegranate_setup(train_data[0:100], pipeline_type)[1]
pomegranate_exact_dimension_dict_scores = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], x_test, y_test, pipeline_type, "POMEGRANATE (EXACT)")
pomegranate_exact_dimension_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], simulated_data_test[:,0:10], simulated_data_test[:,10], pipeline_type, "POMEGRANATE (EXACT)")

#pomegranate hyperparameter simulation scoring - greedy
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pomegranate.pomegranate_setup_b(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pomegranate.pomegranate_setup_b(train_data[0:100], pipeline_type)[1]

pomegranate_greedy_linear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "POMEGRANATE (GREEDY)")
pomegranate_greedy_linear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4],pipeline_type, "POMEGRANATE (GREEDY)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pomegranate.pomegranate_setup_b(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pomegranate.pomegranate_setup_b(train_data[0:100], pipeline_type)[1]

pomegranate_greedy_nonlinear_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "POMEGRANATE (GREEDY)")
pomegranate_greedy_nonlinear_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "POMEGRANATE (GREEDY)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pomegranate.pomegranate_setup_b(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pomegranate.pomegranate_setup_b(train_data[0:100], pipeline_type)[1]

pomegranate_greedy_sparse_dict_scores = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], x_test, y_test, pipeline_type, "POMEGRANATE (GREEDY)")
pomegranate_greedy_sparse_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:4], simulated_data_train[:,4], simulated_data_test[:,0:4], simulated_data_test[:,4], pipeline_type, "POMEGRANATE (GREEDY)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pomegranate.pomegranate_setup_b(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pomegranate.pomegranate_setup_b(train_data[0:100], pipeline_type)[1]
pomegranate_greedy_dimension_dict_scores = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], x_test, y_test, pipeline_type, "POMEGRANATE (GREEDY)")
pomegranate_greedy_dimension_dict_scores_simtest = run_learned_workflows(simulated_data_train[:,0:10], simulated_data_train[:,10], simulated_data_test[:,0:10], simulated_data_test[:,10], pipeline_type, "POMEGRANATE (GREEDY)")

#pomegranate hyperparameter simulation scoring - Chow-liu
#pipeline_type = 1
#simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_pomegranate.pomegranate_setup_c(train_data[0:100], pipeline_type)
#import_simulated_csv()

#pomegranate_chow_linear_dict_scores = run_learned_workflows(pomegranate_sample_train.iloc[:,0:4], pomegranate_sample_train.iloc[:,4], pipeline_type, "POMEGRANATE (CHOW-LIU)")
#pipeline_type = 2
#simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_pomegranate.pomegranate_setup_c(train_data[0:100], pipeline_type)
#import_simulated_csv()

#pomegranate_chow_nonlinear_dict_scores = run_learned_workflows(pomegranate_sample_train.iloc[:,0:4], pomegranate_sample_train.iloc[:,4], pipeline_type, "POMEGRANATE (CHOW-LIU)")
#pipeline_type = 3
#simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_pomegranate.pomegranate_setup_c(train_data[0:100], pipeline_type)
#import_simulated_csv()

#pomegranate_chow_sparse_dict_scores = run_learned_workflows(pomegranate_sample_train.iloc[:,0:4], pomegranate_sample_train.iloc[:,4], pipeline_type, "POMEGRANATE (CHOW-LIU)")
#pipeline_type = 4
#simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_pomegranate.pomegranate_setup_c(train_data[0:100], pipeline_type)
#import_simulated_csv()
#pomegranate_chow_dimension_dict_scores = run_learned_workflows(pomegranate_sample_train.iloc[:,0:10], pomegranate_sample_train.iloc[:,10], pipeline_type, "POMEGRANATE (CHOW-LIU)")

#pgmpy simulation scoring -Hill-climbing
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pgmpy.pgmpy_setup_hc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pgmpy.pgmpy_setup_hc(train_data[0:100], pipeline_type)[1]

pgmpy_hc_linear_dict_scores = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], x_test, y_test, pipeline_type, "PGMPY (HC)")
pgmpy_hc_linear_dict_scores_simtest = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], simulated_data_test.iloc[:,0:4], simulated_data_test.iloc[:,4], pipeline_type, "PGMPY (HC)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pgmpy.pgmpy_setup_hc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pgmpy.pgmpy_setup_hc(train_data[0:100], pipeline_type)[1]

pgmpy_hc_nonlinear_dict_scores = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], x_test, y_test, pipeline_type, "PGMPY (HC)")
pgmpy_hc_nonlinear_dict_scores_simtest = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], simulated_data_test.iloc[:,0:4], simulated_data_test.iloc[:,4], pipeline_type, "PGMPY (HC)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pgmpy.pgmpy_setup_hc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pgmpy.pgmpy_setup_hc(train_data[0:100], pipeline_type)[1]

pgmpy_hc_sparse_dict_scores = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], x_test, y_test, pipeline_type, "PGMPY (HC)")
pgmpy_hc_sparse_dict_scores_simtest = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], simulated_data_test.iloc[:,0:4], simulated_data_test.iloc[:,4], pipeline_type, "PGMPY (HC)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pgmpy.pgmpy_setup_hc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pgmpy.pgmpy_setup_hc(train_data[0:100], pipeline_type)[1]
pgmpy_hc_dimension_dict_scores = run_learned_workflows(simulated_data_train.iloc[:,0:10], simulated_data_train.iloc[:,10], x_test, y_test, pipeline_type, "PGMPY (HC)")
pgmpy_hc_dimension_dict_scores_simtest = run_learned_workflows(simulated_data_train.iloc[:,0:10], simulated_data_train.iloc[:,10], simulated_data_test.iloc[:,0:10], simulated_data_test.iloc[:,10], pipeline_type, "PGMPY (HC)")

#pgmpy simulation scoring - Tree search
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pgmpy.pgmpy_setup_tree(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pgmpy.pgmpy_setup_tree(train_data[0:100], pipeline_type)[1]

pgmpy_tree_linear_dict_scores = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], x_test, y_test, pipeline_type, "PGMPY (Tree)")
pgmpy_tree_linear_dict_scores_simtest = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], simulated_data_test.iloc[:,0:4], simulated_data_test.iloc[:,4], pipeline_type, "PGMPY (Tree)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pgmpy.pgmpy_setup_tree(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pgmpy.pgmpy_setup_tree(train_data[0:100], pipeline_type)[1]

pgmpy_tree_nonlinear_dict_scores = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], x_test, y_test, pipeline_type, "PGMPY (Tree)")
pgmpy_tree_nonlinear_dict_scores_simtest = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], simulated_data_test.iloc[:,0:4], simulated_data_test.iloc[:,4], pipeline_type, "PGMPY (Tree)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pgmpy.pgmpy_setup_tree(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pgmpy.pgmpy_setup_tree(train_data[0:100], pipeline_type)[1]

pgmpy_tree_sparse_dict_scores = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], x_test, y_test, pipeline_type, "PGMPY (TREE)")
pgmpy_tree_sparse_dict_scores_simtest = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], simulated_data_test.iloc[:,0:4], simulated_data_test.iloc[:,4], pipeline_type, "PGMPY (TREE)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pgmpy.pgmpy_setup_tree(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pgmpy.pgmpy_setup_tree(train_data[0:100], pipeline_type)[1]
pgmpy_tree_dimension_dict_scores = run_learned_workflows(simulated_data_train.iloc[:,0:10], simulated_data_train.iloc[:,10], x_test, y_test, pipeline_type, "PGMPY (TREE)")
pgmpy_tree_dimension_dict_scores_simtest = run_learned_workflows(simulated_data_train.iloc[:,0:10], simulated_data_train.iloc[:,10], simulated_data_test.iloc[:,0:10], simulated_data_test.iloc[:,10], pipeline_type, "PGMPY (TREE)")

#pgmpy simulation scoring - MMHC
pipeline_type = 1
simulation_dagsim.setup_realworld(pipeline_type, 1000, 1000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pgmpy.pgmpy_setup_mmhc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pgmpy.pgmpy_setup_mmhc(train_data[0:100], pipeline_type)[1]

pgmpy_mmhc_linear_dict_scores = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], x_test, y_test, pipeline_type, "PGMPY (MMHC)")
pgmpy_mmhc_linear_dict_scores_simtest = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], simulated_data_test.iloc[:,0:4], simulated_data_test.iloc[:,4], pipeline_type, "PGMPY (MMHC)")
pipeline_type = 2
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pgmpy.pgmpy_setup_mmhc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pgmpy.pgmpy_setup_mmhc(train_data[0:100], pipeline_type)[1]

pgmpy_mmhc_nonlinear_dict_scores = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], x_test, y_test, pipeline_type, "PGMPY (MMHC)")
pgmpy_mmhc_nonlinear_dict_scores_simtest = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], simulated_data_test.iloc[:,0:4], simulated_data_test.iloc[:,4], pipeline_type, "PGMPY (MMHC)")
pipeline_type = 3
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pgmpy.pgmpy_setup_mmhc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pgmpy.pgmpy_setup_mmhc(train_data[0:100], pipeline_type)[1]

pgmpy_mmhc_sparse_dict_scores = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], x_test, y_test, pipeline_type, "PGMPY (MMHC)")
pgmpy_mmhc_sparse_dict_scores_simtest = run_learned_workflows(simulated_data_train.iloc[:,0:4], simulated_data_train.iloc[:,4], simulated_data_test.iloc[:,0:4], simulated_data_test.iloc[:,4], pipeline_type, "PGMPY (MMHC)")
pipeline_type = 4
simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
import_real_world_csv(pipeline_type)
simulated_data_train = simulation_pgmpy.pgmpy_setup_mmhc(train_data[0:100], pipeline_type)[0]
simulated_data_test = simulation_pgmpy.pgmpy_setup_mmhc(train_data[0:100], pipeline_type)[1]
pgmpy_mmhc_dimension_dict_scores = run_learned_workflows(simulated_data_train.iloc[:,0:10], simulated_data_train.iloc[:,10], x_test, y_test, pipeline_type, "PGMPY (MMHC)")
pgmpy_mmhc_dimension_dict_scores_simtest = run_learned_workflows(simulated_data_train.iloc[:,0:10], simulated_data_train.iloc[:,10], simulated_data_test.iloc[:,0:10], simulated_data_test.iloc[:,10], pipeline_type, "PGMPY (MMHC)")

#pgmpy simulation scoring - PC - - single positional indexer is out-of-bounds doesnt output same shape as given
#pipeline_type = 1
#simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_pgmpy.pgmpy_setup_pc(train_data[0:100], pipeline_type)
#import_simulated_csv()

#pgmpy_pc_linear_dict_scores = run_learned_workflows(pgmpy_sample_train.iloc[:,0:4], pgmpy_sample_train.iloc[:,4], pipeline_type, "PGMPY (PC)")
#pipeline_type = 2
#simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_pgmpy.pgmpy_setup_pc(train_data[0:100], pipeline_type)
#import_simulated_csv()

#pgmpy_pc_nonlinear_dict_scores = run_learned_workflows(pgmpy_sample_train.iloc[:,0:4], pgmpy_sample_train.iloc[:,4], pipeline_type, "PGMPY (PC)")
#pipeline_type = 3
#simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_pgmpy.pgmpy_setup_pc(train_data[0:100], pipeline_type)
#import_simulated_csv()

#pgmpy_pc_sparse_dict_scores = run_learned_workflows(pgmpy_sample_train.iloc[:,0:4], pgmpy_sample_train.iloc[:,4], pipeline_type, "PGMPY (PC)")
#pipeline_type = 4
#simulation_dagsim.setup_realworld(pipeline_type, 1000, 5000)
#import_real_world_csv(pipeline_type)
#simulation_pgmpy.pgmpy_setup_pc(train_data[0:100], pipeline_type)
#import_simulated_csv()
#pgmpy_pc_dimension_dict_scores = run_learned_workflows(pgmpy_sample_train.iloc[:,0:10], pgmpy_sample_train.iloc[:,10], pipeline_type, "PGMPY (PC)")

def write_learned_to_csv():
    experiments = ['Algorithm', 'Model', 'Linear', 'Non-linear', 'Sparsity', 'Dimensionality']
    with open('simulation_experiments_summary.csv', 'w', newline='') as csvfile:
        fieldnames = ['Algorithm', 'Model', 'Linear', 'Non-linear', 'Sparsity', 'Dimensionality']
        thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        thewriter.writeheader()

        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Decision Tree (gini)','Linear': str(notears_l2_linear_dict_scores["dt"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["dt"]),'Sparsity': str(notears_l2_sparse_dict_scores["dt"]) ,'Dimensionality': str(notears_l2_dimension_dict_scores["dt"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Decision Tree (entropy)','Linear': str(notears_l2_linear_dict_scores["dt_e"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["dt_e"]),'Sparsity': str(notears_l2_sparse_dict_scores["dt_e"]),'Dimensionality': str(notears_l2_dimension_dict_scores["dt_e"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Random Forest (gini)','Linear': str(notears_l2_linear_dict_scores["rf"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["rf"]) ,'Sparsity': str(notears_l2_sparse_dict_scores["rf"]),'Dimensionality': str(notears_l2_dimension_dict_scores["rf"]) })
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Random Forest (entropy)','Linear': str(notears_l2_linear_dict_scores["rf_e"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["rf_e"]),'Sparsity': str(notears_l2_sparse_dict_scores["rf_e"]) ,'Dimensionality': str(notears_l2_dimension_dict_scores["rf_e"]) })
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(notears_l2_linear_dict_scores["lr"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["lr"]),'Sparsity': str(notears_l2_sparse_dict_scores["lr"]),'Dimensionality': str(notears_l2_dimension_dict_scores["lr"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Logistic Regression (l1)','Linear': str(notears_l2_linear_dict_scores["lr_l1"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["lr_l1"]),'Sparsity': str(notears_l2_sparse_dict_scores["lr_l1"]),'Dimensionality': str(notears_l2_dimension_dict_scores["lr_l1"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Logistic Regression (l2)','Linear': str(notears_l2_linear_dict_scores["lr_l2"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["lr_l2"]),'Sparsity': str(notears_l2_sparse_dict_scores["lr_l2"]),'Dimensionality': str(notears_l2_dimension_dict_scores["lr_l2"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(notears_l2_linear_dict_scores["lr_e"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["lr_e"]),'Sparsity': str(notears_l2_sparse_dict_scores["lr_e"]),'Dimensionality': str(notears_l2_dimension_dict_scores["lr_e"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(notears_l2_linear_dict_scores["nb"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["nb"]) ,'Sparsity': str(notears_l2_sparse_dict_scores["nb"]),'Dimensionality': str(notears_l2_dimension_dict_scores["nb"]) })
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(notears_l2_linear_dict_scores["nb_m"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["nb_m"]),'Sparsity': str(notears_l2_sparse_dict_scores["nb_m"]),'Dimensionality': str(notears_l2_dimension_dict_scores["nb_m"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(notears_l2_linear_dict_scores["nb_g"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["nb_g"]),'Sparsity': str(notears_l2_sparse_dict_scores["nb_g"]),'Dimensionality': str(notears_l2_dimension_dict_scores["nb_g"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Naive Bayes (Complement)','Linear': str(notears_l2_linear_dict_scores["nb_c"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["nb_c"]),'Sparsity': str(notears_l2_sparse_dict_scores["nb_c"]),'Dimensionality': str(notears_l2_dimension_dict_scores["nb_c"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(notears_l2_linear_dict_scores["svm"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["svm"]),'Sparsity': str(notears_l2_sparse_dict_scores["svm"]),'Dimensionality': str(notears_l2_dimension_dict_scores["svm"])})
        #thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_l"])) + "," + str(max(notears_l2_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_l"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_l"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_l"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_l"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (poly)','Linear': str(notears_l2_linear_dict_scores["svm_po"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["svm_po"]),'Sparsity': str(notears_l2_sparse_dict_scores["svm_po"]),'Dimensionality': str(notears_l2_dimension_dict_scores["svm_po"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (rbf)','Linear': str(notears_l2_linear_dict_scores["svm_r"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["svm_r"]),'Sparsity': str(notears_l2_sparse_dict_scores["svm_r"]),'Dimensionality': str(notears_l2_dimension_dict_scores["svm_r"]) })
        #thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_pr"])) + "," + str(max(notears_l2_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_pr"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_pr"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_pr"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_pr"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(notears_l2_linear_dict_scores["knn"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["knn"]),'Sparsity': str(notears_l2_sparse_dict_scores["knn"]),'Dimensionality': str(notears_l2_dimension_dict_scores["knn"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(notears_l2_linear_dict_scores["knn_d"]),'Non-linear': str(notears_l2_nonlinear_dict_scores["knn_d"]),'Sparsity': str(notears_l2_sparse_dict_scores["knn_d"]),'Dimensionality': str(notears_l2_dimension_dict_scores["knn_d"])})

        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Decision Tree (gini)','Linear': str(notears_linear_dict_scores["dt"]), 'Non-linear': str(notears_nonlinear_dict_scores["dt"]), 'Sparsity': str(notears_sparse_dict_scores["dt"]), 'Dimensionality': str(notears_dimension_dict_scores["dt"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Decision Tree (entropy)','Linear': str(notears_linear_dict_scores["dt_e"]),'Non-linear': str(notears_nonlinear_dict_scores["dt_e"]),'Sparsity': str(notears_sparse_dict_scores["dt_e"]),'Dimensionality': str(notears_dimension_dict_scores["dt_e"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Random Forest (gini)', 'Linear': str(notears_linear_dict_scores["rf"]), 'Non-linear': str(notears_nonlinear_dict_scores["rf"]), 'Sparsity': str(notears_sparse_dict_scores["rf"]), 'Dimensionality': str(notears_dimension_dict_scores["rf"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Random Forest (entropy)','Linear': str(notears_linear_dict_scores["rf_e"]),'Non-linear': str(notears_nonlinear_dict_scores["rf_e"]),'Sparsity': str(notears_sparse_dict_scores["rf_e"]),'Dimensionality': str(notears_dimension_dict_scores["rf_e"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Logistic Regression (penalty-none)', 'Linear': str(notears_linear_dict_scores["lr"]), 'Non-linear': str(notears_nonlinear_dict_scores["lr"]), 'Sparsity': str(notears_sparse_dict_scores["lr"]), 'Dimensionality': str(notears_dimension_dict_scores["lr"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Logistic Regression (l1)','Linear': str(notears_linear_dict_scores["lr_l1"]),'Non-linear': str(notears_nonlinear_dict_scores["lr_l1"]) ,'Sparsity': str(notears_sparse_dict_scores["lr_l1"]),'Dimensionality': str(notears_dimension_dict_scores["lr_l1"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Logistic Regression (l2)','Linear': str(notears_linear_dict_scores["lr_l2"]),'Non-linear': str(notears_nonlinear_dict_scores["lr_l2"]),'Sparsity': str(notears_sparse_dict_scores["lr_l2"]),'Dimensionality': str(notears_dimension_dict_scores["lr_l2"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(notears_linear_dict_scores["lr_e"]),'Non-linear': str(notears_nonlinear_dict_scores["lr_e"]),'Sparsity': str(notears_sparse_dict_scores["lr_e"]),'Dimensionality': str(notears_dimension_dict_scores["lr_e"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Naive Bayes (Bernoulli)', 'Linear': str(notears_linear_dict_scores["nb"]),'Non-linear': str(notears_nonlinear_dict_scores["nb"]), 'Sparsity': str(notears_sparse_dict_scores["nb"]), 'Dimensionality': str(notears_dimension_dict_scores["nb"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(notears_linear_dict_scores["nb_m"]),'Non-linear': str(notears_nonlinear_dict_scores["nb_m"]),'Sparsity': str(notears_sparse_dict_scores["nb_m"]),'Dimensionality': str(notears_dimension_dict_scores["nb_m"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(notears_linear_dict_scores["nb_g"]),'Non-linear': str(notears_nonlinear_dict_scores["nb_g"]),'Sparsity': str(notears_sparse_dict_scores["nb_g"]),'Dimensionality': str(notears_dimension_dict_scores["nb_g"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Naive Bayes (Complement)','Linear': str(notears_linear_dict_scores["nb_c"]),'Non-linear': str(notears_nonlinear_dict_scores["nb_c"]),'Sparsity': str(notears_sparse_dict_scores["nb_c"]),'Dimensionality': str(notears_dimension_dict_scores["nb_c"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Support Vector Machines (sigmoid)', 'Linear': str(notears_linear_dict_scores["svm"]),'Non-linear': str(notears_nonlinear_dict_scores["svm"]), 'Sparsity': str(notears_sparse_dict_scores["svm"]), 'Dimensionality': str(notears_dimension_dict_scores["svm"])})
        #thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(notears_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_linear_dict_scores["svm_l"])) + "," + str(max(notears_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["svm_l"])) + "," + str(max(notears_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_sparse_dict_scores["svm_l"])) + "," + str(max(notears_sparse_dict_scores["svm_l"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_dimension_dict_scores["svm_l"])) + "," + str(max(notears_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Support Vector Machines (poly)','Linear': str(notears_linear_dict_scores["svm_po"]),'Non-linear': str(notears_nonlinear_dict_scores["svm_po"]),'Sparsity': str(notears_sparse_dict_scores["svm_po"]) ,'Dimensionality': str(notears_dimension_dict_scores["svm_po"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Support Vector Machines (rbf)','Linear': str(notears_linear_dict_scores["svm_r"]),'Non-linear': str(notears_nonlinear_dict_scores["svm_r"]),'Sparsity': str(notears_sparse_dict_scores["svm_r"]),'Dimensionality': str(notears_dimension_dict_scores["svm_r"])})
        #thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(notears_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_linear_dict_scores["svm_pr"])) + "," + str(max(notears_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(notears_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_nonlinear_dict_scores["svm_pr"])) + "," + str(max(notears_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(notears_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_sparse_dict_scores["svm_pr"])) + "," + str(max(notears_sparse_dict_scores["svm_pr"])) + "}",'Dimensionality': str(round(mean(notears_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_dimension_dict_scores["svm_pr"])) + "," + str(max(notears_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'K Nearest Neighbor (uniform)', 'Linear': str(notears_linear_dict_scores["knn"]),'Non-linear': str(notears_nonlinear_dict_scores["knn"]), 'Sparsity': str(notears_sparse_dict_scores["knn"]), 'Dimensionality': str(notears_dimension_dict_scores["knn"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Logistic)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(notears_linear_dict_scores["knn_d"]),'Non-linear': str(notears_nonlinear_dict_scores["knn_d"]),'Sparsity': str(notears_sparse_dict_scores["knn_d"]),'Dimensionality': str(notears_dimension_dict_scores["knn_d"]) })

        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Decision Tree (gini)','Linear': str(notears_poisson_linear_dict_scores["dt"]),'Non-linear': str(notears_poisson_nonlinear_dict_scores["dt"]),'Sparsity': str(notears_poisson_sparse_dict_scores["dt"]),'Dimensionality': str(notears_poisson_dimension_dict_scores["dt"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Decision Tree (entropy)','Linear': str(notears_poisson_linear_dict_scores["dt_e"]),'Non-linear': str(notears_poisson_nonlinear_dict_scores["dt_e"]),'Sparsity': str(notears_poisson_sparse_dict_scores["dt_e"]), 'Dimensionality': str(notears_poisson_dimension_dict_scores["dt_e"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Random Forest (gini)','Linear': str(notears_poisson_linear_dict_scores["rf"]),'Non-linear': str(notears_poisson_nonlinear_dict_scores["rf"]),'Sparsity': str(notears_poisson_sparse_dict_scores["rf"]),'Dimensionality': str(notears_poisson_dimension_dict_scores["rf"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Random Forest (entropy)','Linear': str(notears_poisson_linear_dict_scores["rf_e"]),'Non-linear': str(notears_poisson_nonlinear_dict_scores["rf_e"]),'Sparsity': str(notears_poisson_sparse_dict_scores["rf_e"]), 'Dimensionality': str(notears_poisson_dimension_dict_scores["rf_e"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(notears_poisson_linear_dict_scores["lr"]),'Non-linear': str(notears_poisson_nonlinear_dict_scores["lr"]),'Sparsity': str(notears_poisson_sparse_dict_scores["lr"]),'Dimensionality': str(notears_poisson_dimension_dict_scores["lr"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Logistic Regression (l1)','Linear': str(notears_poisson_linear_dict_scores["lr_l1"]),'Non-linear': str(notears_poisson_nonlinear_dict_scores["lr_l1"]),'Sparsity': str(notears_poisson_sparse_dict_scores["lr_l1"]), 'Dimensionality': str(notears_poisson_dimension_dict_scores["lr_l1"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Logistic Regression (l2)','Linear': str(notears_poisson_linear_dict_scores["lr_l2"]),'Non-linear': str(notears_poisson_nonlinear_dict_scores["lr_l2"]),'Sparsity': str(notears_poisson_sparse_dict_scores["lr_l2"]), 'Dimensionality': str(notears_poisson_dimension_dict_scores["lr_l2"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(notears_poisson_linear_dict_scores["lr_e"]),'Non-linear': str(notears_poisson_nonlinear_dict_scores["lr_e"]),'Sparsity': str(notears_poisson_sparse_dict_scores["lr_e"]), 'Dimensionality': str(notears_poisson_dimension_dict_scores["lr_e"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(notears_poisson_linear_dict_scores["nb"]),'Non-linear': str(notears_poisson_nonlinear_dict_scores["nb"]),'Sparsity': str(notears_poisson_sparse_dict_scores["nb"]) ,'Dimensionality': str(notears_poisson_dimension_dict_scores["nb"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(notears_poisson_linear_dict_scores["nb_m"]),'Non-linear': str(notears_poisson_nonlinear_dict_scores["nb_m"]),'Sparsity': str(notears_poisson_sparse_dict_scores["nb_m"]), 'Dimensionality': str(notears_poisson_dimension_dict_scores["nb_m"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(notears_poisson_linear_dict_scores["nb_g"]),'Non-linear': str(notears_poisson_nonlinear_dict_scores["nb_g"]),'Sparsity': str(notears_poisson_sparse_dict_scores["nb_g"]), 'Dimensionality': str(notears_poisson_dimension_dict_scores["nb_g"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Naive Bayes (Complement)','Linear': str(notears_poisson_linear_dict_scores["nb_c"]),'Non-linear': str(notears_poisson_nonlinear_dict_scores["nb_c"]),'Sparsity': str(notears_poisson_sparse_dict_scores["nb_c"]), 'Dimensionality': str(notears_poisson_dimension_dict_scores["nb_c"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(notears_poisson_linear_dict_scores["svm"]),'Non-linear': str(notears_poisson_nonlinear_dict_scores["svm"]),'Sparsity': str(notears_poisson_sparse_dict_scores["svm"]),'Dimensionality': str(notears_poisson_dimension_dict_scores["svm"])})
        #thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(notears_poisson_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["svm_l"])) + "," + str(max(notears_poisson_linear_dict_scores["svm_l"])) + "}", 'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["svm_l"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["svm_l"])) + "," + str(max(notears_poisson_sparse_dict_scores["svm_l"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["svm_l"])) + "," + str(max(notears_poisson_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Support Vector Machines (poly)','Linear': str(notears_poisson_linear_dict_scores["svm_po"]), 'Non-linear': str(notears_poisson_nonlinear_dict_scores["svm_po"]),'Sparsity': str(notears_poisson_sparse_dict_scores["svm_po"]), 'Dimensionality': str(notears_poisson_dimension_dict_scores["svm_po"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Support Vector Machines (rbf)','Linear': str(notears_poisson_linear_dict_scores["svm_r"]), 'Non-linear': str(notears_poisson_nonlinear_dict_scores["svm_r"]),'Sparsity': str(notears_poisson_sparse_dict_scores["svm_r"]), 'Dimensionality': str(notears_poisson_dimension_dict_scores["svm_r"])})
        #thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(notears_poisson_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_poisson_linear_dict_scores["svm_pr"])) + "," + str(max(notears_poisson_linear_dict_scores["svm_pr"])) + "}", 'Non-linear': str(round(mean(notears_poisson_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_poisson_nonlinear_dict_scores["svm_pr"])) + "," + str(max(notears_poisson_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(notears_poisson_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_poisson_sparse_dict_scores["svm_pr"])) + "," + str(max(notears_poisson_sparse_dict_scores["svm_pr"])) + "}", 'Dimensionality': str(round(mean(notears_poisson_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_poisson_dimension_dict_scores["svm_pr"])) + "," + str(max(notears_poisson_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(notears_poisson_linear_dict_scores["knn"]),'Non-linear': str(notears_poisson_nonlinear_dict_scores["knn"]),'Sparsity': str(notears_poisson_sparse_dict_scores["knn"]),'Dimensionality': str(notears_poisson_dimension_dict_scores["knn"])})
        thewriter.writerow({'Algorithm': 'NO TEARS (Loss-Poisson)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(notears_poisson_linear_dict_scores["knn_d"]), 'Non-linear': str(notears_poisson_nonlinear_dict_scores["knn_d"]),'Sparsity': str(notears_poisson_sparse_dict_scores["knn_d"]), 'Dimensionality': str(notears_poisson_dimension_dict_scores["knn_d"]) })

        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Decision Tree (gini)', 'Linear': str(bnlearn_linear_dict_scores["dt"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["dt"]), 'Sparsity': str(bnlearn_sparse_dict_scores["dt"]), 'Dimensionality': str(bnlearn_dimension_dict_scores["dt"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Decision Tree (entropy)','Linear': str(bnlearn_linear_dict_scores["dt_e"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["dt_e"]),'Sparsity': str(bnlearn_sparse_dict_scores["dt_e"]) ,'Dimensionality': str(bnlearn_dimension_dict_scores["dt_e"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Random Forest (gini)', 'Linear': str(bnlearn_linear_dict_scores["rf"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["rf"]), 'Sparsity': str(bnlearn_sparse_dict_scores["rf"]), 'Dimensionality': str(bnlearn_dimension_dict_scores["rf"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Random Forest (entropy)','Linear': str(bnlearn_linear_dict_scores["rf_e"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["rf_e"]),'Sparsity': str(bnlearn_sparse_dict_scores["rf_e"]),'Dimensionality': str(bnlearn_dimension_dict_scores["rf_e"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Logistic Regression (penalty-none)', 'Linear': str(bnlearn_linear_dict_scores["lr"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["lr"]), 'Sparsity': str(bnlearn_sparse_dict_scores["lr"]), 'Dimensionality': str(bnlearn_dimension_dict_scores["lr"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Logistic Regression (l1)','Linear': str(bnlearn_linear_dict_scores["lr_l1"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["lr_l1"]),'Sparsity': str(bnlearn_sparse_dict_scores["lr_l1"]),'Dimensionality': str(bnlearn_dimension_dict_scores["lr_l1"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Logistic Regression (l2)','Linear': str(bnlearn_linear_dict_scores["lr_l2"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["lr_l2"]),'Sparsity': str(bnlearn_sparse_dict_scores["lr_l2"]),'Dimensionality': str(bnlearn_dimension_dict_scores["lr_l2"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(bnlearn_linear_dict_scores["lr_e"]) ,'Non-linear': str(bnlearn_nonlinear_dict_scores["lr_e"]),'Sparsity': str(bnlearn_sparse_dict_scores["lr_e"]),'Dimensionality': str(bnlearn_dimension_dict_scores["lr_e"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Naive Bayes (Bernoulli)', 'Linear': str(bnlearn_linear_dict_scores["nb"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["nb"]), 'Sparsity': str(bnlearn_sparse_dict_scores["nb"]), 'Dimensionality': str(bnlearn_dimension_dict_scores["nb"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(bnlearn_linear_dict_scores["nb_m"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["nb_m"]),'Sparsity': str(bnlearn_sparse_dict_scores["nb_m"]),'Dimensionality': str(bnlearn_dimension_dict_scores["nb_m"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(bnlearn_linear_dict_scores["nb_g"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["nb_g"]),'Sparsity': str(bnlearn_sparse_dict_scores["nb_g"]) ,'Dimensionality': str(bnlearn_dimension_dict_scores["nb_g"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Naive Bayes (Complement)','Linear': str(bnlearn_linear_dict_scores["nb_c"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["nb_c"]) ,'Sparsity': str(bnlearn_sparse_dict_scores["nb_c"]) ,'Dimensionality': str(bnlearn_dimension_dict_scores["nb_c"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Support Vector Machines (sigmoid)', 'Linear': str(bnlearn_linear_dict_scores["svm"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["svm"]), 'Sparsity': str(bnlearn_sparse_dict_scores["svm"]), 'Dimensionality': str(bnlearn_dimension_dict_scores["svm"])})
        #thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["svm_l"])) + "," + str(max(bnlearn_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["svm_l"])) + "," + str(max(bnlearn_sparse_dict_scores["svm_l"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["svm_l"])) + "," + str(max(bnlearn_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Support Vector Machines (poly)','Linear': str(bnlearn_linear_dict_scores["svm_po"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["svm_po"]) ,'Sparsity': str(bnlearn_sparse_dict_scores["svm_po"]),'Dimensionality': str(bnlearn_dimension_dict_scores["svm_po"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Support Vector Machines (rbf)','Linear': str(bnlearn_linear_dict_scores["svm_r"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["svm_r"]) ,'Sparsity': str(bnlearn_sparse_dict_scores["svm_r"]) ,'Dimensionality': str(bnlearn_dimension_dict_scores["svm_r"]) })
        #thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(bnlearn_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_nonlinear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(bnlearn_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_sparse_dict_scores["svm_pr"])) + "," + str(max(bnlearn_sparse_dict_scores["svm_pr"])) + "}",'Dimensionality': str(round(mean(bnlearn_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_dimension_dict_scores["svm_pr"])) + "," + str(max(bnlearn_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'K Nearest Neighbor (uniform)', 'Linear': str(bnlearn_linear_dict_scores["knn"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["knn"]), 'Sparsity': str(bnlearn_sparse_dict_scores["knn"]), 'Dimensionality': str(bnlearn_dimension_dict_scores["knn"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (HC)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(bnlearn_linear_dict_scores["knn_d"]),'Non-linear': str(bnlearn_nonlinear_dict_scores["knn_d"]) ,'Sparsity': str(bnlearn_sparse_dict_scores["knn_d"]) ,'Dimensionality': str(bnlearn_dimension_dict_scores["knn_d"]) })

        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Decision Tree (gini)', 'Linear': str(bnlearn_tabu_linear_dict_scores["dt"]),'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["dt"]), 'Sparsity': str(bnlearn_tabu_sparse_dict_scores["dt"]), 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["dt"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Decision Tree (entropy)','Linear': str(bnlearn_tabu_linear_dict_scores["dt_e"]),'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["dt_e"]),'Sparsity': str(bnlearn_tabu_sparse_dict_scores["dt_e"]), 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["dt_e"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Random Forest (gini)', 'Linear': str(bnlearn_tabu_linear_dict_scores["rf"]),'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["rf"]), 'Sparsity': str(bnlearn_tabu_sparse_dict_scores["rf"]), 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["rf"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Random Forest (entropy)','Linear': str(bnlearn_tabu_linear_dict_scores["rf_e"]),'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["rf_e"]),'Sparsity': str(bnlearn_tabu_sparse_dict_scores["rf_e"]) , 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["rf_e"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Logistic Regression (penalty-none)', 'Linear': str(bnlearn_tabu_linear_dict_scores["lr"]),'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["lr"]), 'Sparsity': str(bnlearn_tabu_sparse_dict_scores["lr"]), 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["lr"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Logistic Regression (l1)','Linear': str(bnlearn_tabu_linear_dict_scores["lr_l1"]) ,'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["lr_l1"]) ,'Sparsity': str(bnlearn_tabu_sparse_dict_scores["lr_l1"]) , 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["lr_l1"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Logistic Regression (l2)','Linear': str(bnlearn_tabu_linear_dict_scores["lr_l2"]) ,'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["lr_l2"]),'Sparsity': str(bnlearn_tabu_sparse_dict_scores["lr_l2"]) , 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["lr_l2"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(bnlearn_tabu_linear_dict_scores["lr_e"]) ,'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["lr_e"]),'Sparsity': str(bnlearn_tabu_sparse_dict_scores["lr_e"]) , 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["lr_e"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Naive Bayes (Bernoulli)', 'Linear': str(bnlearn_tabu_linear_dict_scores["nb"]),'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["nb"]), 'Sparsity': str(bnlearn_tabu_sparse_dict_scores["nb"]), 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["nb"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(bnlearn_tabu_linear_dict_scores["nb_m"]) ,'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["nb_m"]) ,'Sparsity': str(bnlearn_tabu_sparse_dict_scores["nb_m"]) , 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["nb_m"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(bnlearn_tabu_linear_dict_scores["nb_g"]) ,'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["nb_g"]),'Sparsity': str(bnlearn_tabu_sparse_dict_scores["nb_g"]) , 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["nb_g"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Naive Bayes (Complement)','Linear': str(bnlearn_tabu_linear_dict_scores["nb_c"]),'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["nb_c"]) ,'Sparsity': str(bnlearn_tabu_sparse_dict_scores["nb_c"]) , 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["nb_c"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Support Vector Machines (sigmoid)', 'Linear': str(bnlearn_tabu_linear_dict_scores["svm"]),'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["svm"]), 'Sparsity': str(bnlearn_tabu_sparse_dict_scores["svm"]), 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["svm"])})
        #thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["svm_l"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["svm_l"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["svm_l"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["svm_l"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Support Vector Machines (poly)','Linear': str(bnlearn_tabu_linear_dict_scores["svm_po"]) ,'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["svm_po"]) ,'Sparsity': str(bnlearn_tabu_sparse_dict_scores["svm_po"]) , 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["svm_po"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Support Vector Machines (rbf)','Linear': str(bnlearn_tabu_linear_dict_scores["svm_r"]) ,'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["svm_r"]),'Sparsity': str(bnlearn_tabu_sparse_dict_scores["svm_r"]) , 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["svm_r"]) })
        #thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_tabu_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_tabu_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_tabu_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(bnlearn_tabu_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_tabu_nonlinear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_tabu_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(bnlearn_tabu_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_tabu_sparse_dict_scores["svm_pr"])) + "," + str(max(bnlearn_tabu_sparse_dict_scores["svm_pr"])) + "}", 'Dimensionality': str(round(mean(bnlearn_tabu_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_tabu_dimension_dict_scores["svm_pr"])) + "," + str(max(bnlearn_tabu_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'K Nearest Neighbor (uniform)', 'Linear': str(bnlearn_tabu_linear_dict_scores["knn"]),'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["knn"]), 'Sparsity': str(bnlearn_tabu_sparse_dict_scores["knn"]), 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["knn"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (TABU)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(bnlearn_tabu_linear_dict_scores["knn_d"]) ,'Non-linear': str(bnlearn_tabu_nonlinear_dict_scores["knn_d"]) ,'Sparsity': str(bnlearn_tabu_sparse_dict_scores["knn_d"]) , 'Dimensionality': str(bnlearn_tabu_dimension_dict_scores["knn_d"]) })

        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Decision Tree (gini)','Linear': str(bnlearn_pc_linear_dict_scores["dt"]),'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Decision Tree (entropy)','Linear': str(bnlearn_pc_linear_dict_scores["dt_e"]),'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Random Forest (gini)','Linear': str(bnlearn_pc_linear_dict_scores["rf"]) ,'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Random Forest (entropy)','Linear': str(bnlearn_pc_linear_dict_scores["rf_e"]) ,'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(bnlearn_pc_linear_dict_scores["lr"]) ,'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Logistic Regression (l1)','Linear': str(bnlearn_pc_linear_dict_scores["lr_l1"]),'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Logistic Regression (l2)','Linear': str(bnlearn_pc_linear_dict_scores["lr_l2"]) ,'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(bnlearn_pc_linear_dict_scores["lr_e"]),'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(bnlearn_pc_linear_dict_scores["nb"]) ,'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(bnlearn_pc_linear_dict_scores["nb_m"]),'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(bnlearn_pc_linear_dict_scores["nb_g"]) ,'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Naive Bayes (Complement)','Linear': str(bnlearn_pc_linear_dict_scores["nb_c"]) ,'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(bnlearn_pc_linear_dict_scores["svm"]) ,'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_pc_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(bnlearn_pc_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_pc_nonlinear_dict_scores["svm_l"])) + "," + str(max(bnlearn_pc_nonlinear_dict_scores["svm_l"])) + "}", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Support Vector Machines (poly)','Linear': str(bnlearn_pc_linear_dict_scores["svm_po"]) ,'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Support Vector Machines (rbf)','Linear': str(bnlearn_pc_linear_dict_scores["svm_r"]) ,'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_pc_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_pc_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_pc_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(bnlearn_pc_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_pc_nonlinear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_pc_nonlinear_dict_scores["svm_pr"])) + "}", 'Sparsity': "NA",'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(bnlearn_pc_linear_dict_scores["knn"]) ,'Non-linear': "NA",'Sparsity': "NA", 'Dimensionality': "NA"})
        thewriter.writerow({'Algorithm': 'BN LEARN (PC)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(bnlearn_pc_linear_dict_scores["knn_d"]) ,'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})

        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Decision Tree (gini)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["dt"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["dt"])) + "," + str(max(bnlearn_gs_linear_dict_scores["dt"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Decision Tree (entropy)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["dt_e"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["dt_e"])) + "," + str(max(bnlearn_gs_linear_dict_scores["dt_e"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Random Forest (gini)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["rf"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["rf"])) + "," + str(max(bnlearn_gs_linear_dict_scores["rf"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Random Forest (entropy)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["rf_e"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["rf_e"])) + "," + str(max(bnlearn_gs_linear_dict_scores["rf_e"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["lr"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["lr"])) + "," + str(max(bnlearn_gs_linear_dict_scores["lr"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Logistic Regression (l1)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["lr_l1"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["lr_l1"])) + "," + str(max(bnlearn_gs_linear_dict_scores["lr_l1"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Logistic Regression (l2)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["lr_l2"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["lr_l2"])) + "," + str(max(bnlearn_gs_linear_dict_scores["lr_l2"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["lr_e"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["lr_e"])) + "," + str(max(bnlearn_gs_linear_dict_scores["lr_e"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["nb"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["nb"])) + "," + str(max(bnlearn_gs_linear_dict_scores["nb"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["nb_m"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["nb_m"])) + "," + str(max(bnlearn_gs_linear_dict_scores["nb_m"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["nb_g"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["nb_g"])) + "," + str(max(bnlearn_gs_linear_dict_scores["nb_g"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Naive Bayes (Complement)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["nb_c"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["nb_c"])) + "," + str(max(bnlearn_gs_linear_dict_scores["nb_c"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["svm"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["svm"])) + "," + str(max(bnlearn_gs_linear_dict_scores["svm"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        ##thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_gs_linear_dict_scores["svm_l"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Support Vector Machines (poly)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["svm_po"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["svm_po"])) + "," + str(max(bnlearn_gs_linear_dict_scores["svm_po"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Support Vector Machines (rbf)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["svm_r"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["svm_r"])) + "," + str(max(bnlearn_gs_linear_dict_scores["svm_r"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        ##thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_gs_linear_dict_scores["svm_pr"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["knn"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["knn"])) + "," + str(max(bnlearn_gs_linear_dict_scores["knn"])) + "}",'Non-linear': "NA",'Sparsity': "NA",'Dimensionality': "NA"})
        #thewriter.writerow({'Algorithm': 'BN LEARN (GS)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(round(mean(bnlearn_gs_linear_dict_scores["knn_d"]), 2)) + " {" + str(min(bnlearn_gs_linear_dict_scores["knn_d"])) + "," + str(max(bnlearn_gs_linear_dict_scores["knn_d"])) + "}", 'Non-linear': "NA", 'Sparsity': "NA",'Dimensionality': "NA"})

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

        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Decision Tree (gini)','Linear': str(bnlearn_mmhc_linear_dict_scores["dt"]) ,'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["dt"]) ,'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["dt"]) ,'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["dt"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Decision Tree (entropy)','Linear': str(bnlearn_mmhc_linear_dict_scores["dt_e"]),'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["dt_e"]) ,'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["dt_e"]) , 'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["dt_e"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Random Forest (gini)','Linear': str(bnlearn_mmhc_linear_dict_scores["rf"]) ,'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["rf"]),'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["rf"]) ,'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["rf"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Random Forest (entropy)','Linear': str(bnlearn_mmhc_linear_dict_scores["rf_e"]) ,'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["rf_e"]) ,'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["rf_e"]) , 'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["rf_e"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(bnlearn_mmhc_linear_dict_scores["lr"]) ,'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["lr"]) ,'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["lr"]) ,'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["lr"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Logistic Regression (l1)','Linear': str(bnlearn_mmhc_linear_dict_scores["lr_l1"]),'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["lr_l1"]) ,'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["lr_l1"]) , 'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["lr_l1"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Logistic Regression (l2)','Linear': str(bnlearn_mmhc_linear_dict_scores["lr_l2"]),'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["lr_l2"]),'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["lr_l2"]) , 'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["lr_l2"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(bnlearn_mmhc_linear_dict_scores["lr_e"]) ,'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["lr_e"]) ,'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["lr_e"]) , 'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["lr_e"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(bnlearn_mmhc_linear_dict_scores["nb"]) ,'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["nb"]) ,'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["nb"]) ,'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["nb"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(bnlearn_mmhc_linear_dict_scores["nb_m"]) ,'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["nb_m"]) ,'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["nb_m"]) , 'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["nb_m"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(bnlearn_mmhc_linear_dict_scores["nb_g"]),'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["nb_g"]) ,'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["nb_g"]) , 'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["nb_g"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Naive Bayes (Complement)','Linear': str(bnlearn_mmhc_linear_dict_scores["nb_c"]) ,'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["nb_c"]) ,'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["nb_c"]) , 'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["nb_c"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(bnlearn_mmhc_linear_dict_scores["svm"]) ,'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["svm"]) ,'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["svm"]) ,'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["svm"]) })
        #thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["svm_l"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["svm_l"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["svm_l"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["svm_l"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Support Vector Machines (poly)','Linear': str(bnlearn_mmhc_linear_dict_scores["svm_po"]) ,'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["svm_po"]) ,'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["svm_po"]) , 'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["svm_po"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Support Vector Machines (rbf)','Linear': str(bnlearn_mmhc_linear_dict_scores["svm_r"]) ,'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["svm_r"]) ,'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["svm_r"]) , 'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["svm_r"]) })
        #thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_mmhc_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_mmhc_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_mmhc_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(bnlearn_mmhc_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_mmhc_nonlinear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_mmhc_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(bnlearn_mmhc_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_mmhc_sparse_dict_scores["svm_pr"])) + "," + str(max(bnlearn_mmhc_sparse_dict_scores["svm_pr"])) + "}", 'Dimensionality': str(round(mean(bnlearn_mmhc_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_mmhc_dimension_dict_scores["svm_pr"])) + "," + str(max(bnlearn_mmhc_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(bnlearn_mmhc_linear_dict_scores["knn"]),'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["knn"]),'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["knn"]),'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["knn"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (MMHC)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(bnlearn_mmhc_linear_dict_scores["knn_d"]),'Non-linear': str(bnlearn_mmhc_nonlinear_dict_scores["knn_d"]) ,'Sparsity': str(bnlearn_mmhc_sparse_dict_scores["knn_d"]), 'Dimensionality': str(bnlearn_mmhc_dimension_dict_scores["knn_d"]) })

        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Decision Tree (gini)','Linear': str(bnlearn_rsmax2_linear_dict_scores["dt"]),'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["dt"]) ,'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["dt"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["dt"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Decision Tree (entropy)','Linear': str(bnlearn_rsmax2_linear_dict_scores["dt_e"]) ,'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["dt_e"]),'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["dt_e"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["dt_e"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Random Forest (gini)','Linear': str(bnlearn_rsmax2_linear_dict_scores["rf"]) ,'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["rf"]) ,'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["rf"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["rf"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Random Forest (entropy)','Linear': str(bnlearn_rsmax2_linear_dict_scores["rf_e"]),'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["rf_e"]) ,'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["rf_e"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["rf_e"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(bnlearn_rsmax2_linear_dict_scores["lr"]),'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["lr"]) ,'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["lr"]), 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["lr"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Logistic Regression (l1)','Linear': str(bnlearn_rsmax2_linear_dict_scores["lr_l1"]) ,'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["lr_l1"]) ,'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["lr_l1"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["lr_l1"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Logistic Regression (l2)','Linear': str(bnlearn_rsmax2_linear_dict_scores["lr_l2"]) ,'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["lr_l2"]),'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["lr_l2"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["lr_l2"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(bnlearn_rsmax2_linear_dict_scores["lr_e"]),'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["lr_e"]) ,'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["lr_e"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["lr_e"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(bnlearn_rsmax2_linear_dict_scores["nb"]) ,'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["nb"]),'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["nb"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["nb"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(bnlearn_rsmax2_linear_dict_scores["nb_m"]) ,'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["nb_m"]) ,'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["nb_m"]), 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["nb_m"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(bnlearn_rsmax2_linear_dict_scores["nb_g"]) ,'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["nb_g"]),'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["nb_g"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["nb_g"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Naive Bayes (Complement)','Linear': str(bnlearn_rsmax2_linear_dict_scores["nb_c"]),'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["nb_c"]) ,'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["nb_c"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["nb_c"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(bnlearn_rsmax2_linear_dict_scores["svm"]) ,'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["svm"]) ,'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["svm"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["svm"]) })
        #thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["svm_l"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["svm_l"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["svm_l"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["svm_l"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Support Vector Machines (poly)','Linear': str(bnlearn_rsmax2_linear_dict_scores["svm_po"]),'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["svm_po"]) ,'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["svm_po"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["svm_po"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Support Vector Machines (rbf)','Linear': str(bnlearn_rsmax2_linear_dict_scores["svm_r"]) ,'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["svm_r"]) ,'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["svm_r"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["svm_r"])})
        #thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_rsmax2_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_rsmax2_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_rsmax2_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(bnlearn_rsmax2_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_rsmax2_nonlinear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_rsmax2_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(bnlearn_rsmax2_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_rsmax2_sparse_dict_scores["svm_pr"])) + "," + str(max(bnlearn_rsmax2_sparse_dict_scores["svm_pr"])) + "}", 'Dimensionality': str(round(mean(bnlearn_rsmax2_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_rsmax2_dimension_dict_scores["svm_pr"])) + "," + str(max(bnlearn_rsmax2_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(bnlearn_rsmax2_linear_dict_scores["knn"]) ,'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["knn"]) ,'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["knn"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["knn"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (RSMAX2)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(bnlearn_rsmax2_linear_dict_scores["knn_d"]),'Non-linear': str(bnlearn_rsmax2_nonlinear_dict_scores["knn_d"]),'Sparsity': str(bnlearn_rsmax2_sparse_dict_scores["knn_d"]) , 'Dimensionality': str(bnlearn_rsmax2_dimension_dict_scores["knn_d"])})

        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Decision Tree (gini)','Linear': str(bnlearn_h2pc_linear_dict_scores["dt"]) ,'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["dt"]) ,'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["dt"]), 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["dt"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Decision Tree (entropy)','Linear': str(bnlearn_h2pc_linear_dict_scores["dt_e"]),'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["dt_e"]),'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["dt_e"]) , 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["dt_e"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Random Forest (gini)','Linear': str(bnlearn_h2pc_linear_dict_scores["rf"]),'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["rf"]) ,'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["rf"]), 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["rf"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Random Forest (entropy)','Linear': str(bnlearn_h2pc_linear_dict_scores["rf_e"]) ,'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["rf_e"]) ,'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["rf_e"]) , 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["rf_e"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Logistic Regression (penalty-none)','Linear': str(bnlearn_h2pc_linear_dict_scores["lr"]),'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["lr"]) ,'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["lr"]) , 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["lr"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Logistic Regression (l1)','Linear': str(bnlearn_h2pc_linear_dict_scores["lr_l1"]) ,'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["lr_l1"]),'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["lr_l1"]) , 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["lr_l1"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Logistic Regression (l2)','Linear': str(bnlearn_h2pc_linear_dict_scores["lr_l2"]),'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["lr_l2"]) ,'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["lr_l2"]) , 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["lr_l2"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Logistic Regression (elasticnet)','Linear': str(bnlearn_h2pc_linear_dict_scores["lr_e"]) ,'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["lr_e"]) ,'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["lr_e"]) , 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["lr_e"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Naive Bayes (Bernoulli)','Linear': str(bnlearn_h2pc_linear_dict_scores["nb"]) ,'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["nb"]) ,'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["nb"]) , 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["nb"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Naive Bayes (Multinomial)','Linear': str(bnlearn_h2pc_linear_dict_scores["nb_m"]) ,'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["nb_m"]) ,'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["nb_m"]) , 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["nb_m"])})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Naive Bayes (Gaussian)','Linear': str(bnlearn_h2pc_linear_dict_scores["nb_g"]),'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["nb_g"]) ,'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["nb_g"]) , 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["nb_g"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Naive Bayes (Complement)','Linear': str(bnlearn_h2pc_linear_dict_scores["nb_c"]) ,'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["nb_c"]) ,'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["nb_c"]) , 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["nb_c"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Support Vector Machines (sigmoid)','Linear': str(bnlearn_h2pc_linear_dict_scores["svm"]),'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["svm"]),'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["svm"]), 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["svm"])})
        #thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["svm_l"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["svm_l"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["svm_l"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["svm_l"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["svm_l"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Support Vector Machines (poly)','Linear': str(bnlearn_h2pc_linear_dict_scores["svm_po"]),'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["svm_po"]) ,'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["svm_po"]) , 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["svm_po"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Support Vector Machines (rbf)','Linear': str(bnlearn_h2pc_linear_dict_scores["svm_r"]) ,'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["svm_r"]) ,'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["svm_r"]) , 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["svm_r"]) })
        #thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(bnlearn_h2pc_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_h2pc_linear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_h2pc_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(bnlearn_h2pc_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_h2pc_nonlinear_dict_scores["svm_pr"])) + "," + str(max(bnlearn_h2pc_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(bnlearn_h2pc_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_h2pc_sparse_dict_scores["svm_pr"])) + "," + str(max(bnlearn_h2pc_sparse_dict_scores["svm_pr"])) + "}", 'Dimensionality': str(round(mean(bnlearn_h2pc_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(bnlearn_h2pc_dimension_dict_scores["svm_pr"])) + "," + str(max(bnlearn_h2pc_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'K Nearest Neighbor (uniform)','Linear': str(bnlearn_h2pc_linear_dict_scores["knn"]) ,'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["knn"]) ,'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["knn"]) , 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["knn"]) })
        thewriter.writerow({'Algorithm': 'BN LEARN (H2PC)', 'Model': 'K Nearest Neighbor (distance)','Linear': str(bnlearn_h2pc_linear_dict_scores["knn_d"]) ,'Non-linear': str(bnlearn_h2pc_nonlinear_dict_scores["knn_d"]) ,'Sparsity': str(bnlearn_h2pc_sparse_dict_scores["knn_d"]) , 'Dimensionality': str(bnlearn_h2pc_dimension_dict_scores["knn_d"]) })

        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Decision Tree (gini)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["dt"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["dt"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["dt"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["dt"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Decision Tree (entropy)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["dt_e"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["dt_e"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["dt_e"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["dt_e"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Random Forest (gini)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["rf"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["rf"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["rf"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["rf"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Random Forest (entropy)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["rf_e"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["rf_e"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["rf_e"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["rf_e"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Logistic Regression (penalty-none)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["lr"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["lr"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["lr"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["lr"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Logistic Regression (l1)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["lr_l1"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["lr_l1"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["lr_l1"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["lr_l1"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Logistic Regression (l2)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["lr_l2"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["lr_l2"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["lr_l2"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["lr_l2"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Logistic Regression (elasticnet)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["lr_e"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["lr_e"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["lr_e"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["lr_e"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Naive Bayes (Bernoulli)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["nb"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["nb"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["nb"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["nb"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Naive Bayes (Multinomial)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["nb_m"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["nb_m"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["nb_m"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["nb_m"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Naive Bayes (Gaussian)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["nb_g"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["nb_g"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["nb_g"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["nb_g"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Naive Bayes (Complement)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["nb_c"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["nb_c"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["nb_c"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["nb_c"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Support Vector Machines (sigmoid)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["svm"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["svm"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["svm"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["svm"])})
        # thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_l"])) + "," + str(max(notears_l2_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_l"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_l"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_l"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_l"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Support Vector Machines (poly)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["svm_po"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["svm_po"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["svm_po"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["svm_po"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'Support Vector Machines (rbf)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["svm_r"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["svm_r"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["svm_r"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["svm_r"])})
        # thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_pr"])) + "," + str(max(notears_l2_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_pr"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_pr"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_pr"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_pr"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'K Nearest Neighbor (uniform)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["knn"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["knn"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["knn"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["knn"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Exact)', 'Model': 'K Nearest Neighbor (distance)',
                            'Linear': str(pomegranate_exact_linear_dict_scores["knn_d"]),
                            'Non-linear': str(pomegranate_exact_nonlinear_dict_scores["knn_d"]),
                            'Sparsity': str(pomegranate_exact_sparse_dict_scores["knn_d"]),
                            'Dimensionality': str(pomegranate_exact_dimension_dict_scores["knn_d"])})

        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Decision Tree (gini)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["dt"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["dt"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["dt"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["dt"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Decision Tree (entropy)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["dt_e"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["dt_e"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["dt_e"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["dt_e"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Random Forest (gini)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["rf"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["rf"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["rf"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["rf"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Random Forest (entropy)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["rf_e"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["rf_e"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["rf_e"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["rf_e"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Logistic Regression (penalty-none)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["lr"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["lr"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["lr"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["lr"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Logistic Regression (l1)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["lr_l1"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["lr_l1"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["lr_l1"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["lr_l1"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Logistic Regression (l2)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["lr_l2"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["lr_l2"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["lr_l2"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["lr_l2"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Logistic Regression (elasticnet)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["lr_e"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["lr_e"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["lr_e"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["lr_e"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Naive Bayes (Bernoulli)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["nb"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["nb"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["nb"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["nb"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Naive Bayes (Multinomial)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["nb_m"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["nb_m"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["nb_m"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["nb_m"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Naive Bayes (Gaussian)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["nb_g"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["nb_g"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["nb_g"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["nb_g"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Naive Bayes (Complement)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["nb_c"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["nb_c"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["nb_c"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["nb_c"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Support Vector Machines (sigmoid)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["svm"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["svm"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["svm"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["svm"])})
        # thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_l"])) + "," + str(max(notears_l2_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_l"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_l"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_l"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_l"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Support Vector Machines (poly)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["svm_po"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["svm_po"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["svm_po"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["svm_po"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'Support Vector Machines (rbf)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["svm_r"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["svm_r"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["svm_r"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["svm_r"])})
        # thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_pr"])) + "," + str(max(notears_l2_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_pr"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_pr"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_pr"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_pr"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'K Nearest Neighbor (uniform)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["knn"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["knn"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["knn"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["knn"])})
        thewriter.writerow({'Algorithm': 'Pomegranate (Greedy)', 'Model': 'K Nearest Neighbor (distance)',
                            'Linear': str(pomegranate_greedy_linear_dict_scores["knn_d"]),
                            'Non-linear': str(pomegranate_greedy_nonlinear_dict_scores["knn_d"]),
                            'Sparsity': str(pomegranate_greedy_sparse_dict_scores["knn_d"]),
                            'Dimensionality': str(pomegranate_greedy_dimension_dict_scores["knn_d"])})

        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Decision Tree (gini)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["dt"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["dt"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["dt"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["dt"])})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Decision Tree (entropy)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["dt_e"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["dt_e"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["dt_e"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["dt_e"])})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Random Forest (gini)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["rf"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["rf"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["rf"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["rf"])})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Random Forest (entropy)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["rf_e"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["rf_e"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["rf_e"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["rf_e"])})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Logistic Regression (penalty-none)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["lr"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["lr"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["lr"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["lr"])})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Logistic Regression (l1)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["lr_l1"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["lr_l1"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["lr_l1"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["lr_l1"])})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Logistic Regression (l2)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["lr_l2"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["lr_l2"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["lr_l2"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["lr_l2"])})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Logistic Regression (elasticnet)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["lr_e"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["lr_e"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["lr_e"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["lr_e"])})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Naive Bayes (Bernoulli)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["nb"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["nb"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["nb"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["nb"])})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Naive Bayes (Multinomial)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["nb_m"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["nb_m"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["nb_m"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["nb_m"])})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Naive Bayes (Gaussian)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["nb_g"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["nb_g"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["nb_g"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["nb_g"])})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Naive Bayes (Complement)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["nb_c"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["nb_c"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["nb_c"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["nb_c"])})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Support Vector Machines (sigmoid)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["svm"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["svm"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["svm"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["svm"])})
        # thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_l"])) + "," + str(max(notears_l2_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_l"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_l"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_l"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_l"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Support Vector Machines (poly)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["svm_po"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["svm_po"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["svm_po"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["svm_po"])})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'Support Vector Machines (rbf)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["svm_r"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["svm_r"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["svm_r"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["svm_r"])})
        # thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_pr"])) + "," + str(max(notears_l2_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_pr"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_pr"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_pr"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_pr"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'K Nearest Neighbor (uniform)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["knn"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["knn"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["knn"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["knn"])})
        thewriter.writerow({'Algorithm': 'PGMPY (HC)', 'Model': 'K Nearest Neighbor (distance)',
                            'Linear': str(pgmpy_hc_linear_dict_scores["knn_d"]),
                            'Non-linear': str(pgmpy_hc_nonlinear_dict_scores["knn_d"]),
                            'Sparsity': str(pgmpy_hc_sparse_dict_scores["knn_d"]),
                            'Dimensionality': str(pgmpy_hc_dimension_dict_scores["knn_d"])})

        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Decision Tree (gini)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["dt"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["dt"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["dt"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["dt"])})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Decision Tree (entropy)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["dt_e"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["dt_e"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["dt_e"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["dt_e"])})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Random Forest (gini)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["rf"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["rf"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["rf"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["rf"])})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Random Forest (entropy)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["rf_e"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["rf_e"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["rf_e"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["rf_e"])})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Logistic Regression (penalty-none)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["lr"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["lr"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["lr"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["lr"])})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Logistic Regression (l1)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["lr_l1"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["lr_l1"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["lr_l1"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["lr_l1"])})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Logistic Regression (l2)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["lr_l2"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["lr_l2"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["lr_l2"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["lr_l2"])})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Logistic Regression (elasticnet)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["lr_e"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["lr_e"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["lr_e"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["lr_e"])})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Naive Bayes (Bernoulli)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["nb"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["nb"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["nb"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["nb"])})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Naive Bayes (Multinomial)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["nb_m"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["nb_m"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["nb_m"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["nb_m"])})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Naive Bayes (Gaussian)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["nb_g"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["nb_g"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["nb_g"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["nb_g"])})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Naive Bayes (Complement)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["nb_c"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["nb_c"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["nb_c"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["nb_c"])})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Support Vector Machines (sigmoid)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["svm"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["svm"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["svm"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["svm"])})
        # thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_l"])) + "," + str(max(notears_l2_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_l"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_l"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_l"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_l"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Support Vector Machines (poly)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["svm_po"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["svm_po"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["svm_po"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["svm_po"])})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'Support Vector Machines (rbf)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["svm_r"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["svm_r"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["svm_r"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["svm_r"])})
        # thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_pr"])) + "," + str(max(notears_l2_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_pr"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_pr"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_pr"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_pr"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'K Nearest Neighbor (uniform)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["knn"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["knn"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["knn"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["knn"])})
        thewriter.writerow({'Algorithm': 'PGMPY (MMHC)', 'Model': 'K Nearest Neighbor (distance)',
                            'Linear': str(pgmpy_mmhc_linear_dict_scores["knn_d"]),
                            'Non-linear': str(pgmpy_mmhc_nonlinear_dict_scores["knn_d"]),
                            'Sparsity': str(pgmpy_mmhc_sparse_dict_scores["knn_d"]),
                            'Dimensionality': str(pgmpy_mmhc_dimension_dict_scores["knn_d"])})

        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Decision Tree (gini)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["dt"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["dt"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["dt"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["dt"])})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Decision Tree (entropy)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["dt_e"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["dt_e"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["dt_e"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["dt_e"])})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Random Forest (gini)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["rf"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["rf"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["rf"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["rf"])})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Random Forest (entropy)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["rf_e"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["rf_e"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["rf_e"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["rf_e"])})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Logistic Regression (penalty-none)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["lr"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["lr"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["lr"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["lr"])})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Logistic Regression (l1)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["lr_l1"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["lr_l1"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["lr_l1"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["lr_l1"])})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Logistic Regression (l2)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["lr_l2"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["lr_l2"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["lr_l2"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["lr_l2"])})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Logistic Regression (elasticnet)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["lr_e"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["lr_e"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["lr_e"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["lr_e"])})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Naive Bayes (Bernoulli)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["nb"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["nb"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["nb"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["nb"])})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Naive Bayes (Multinomial)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["nb_m"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["nb_m"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["nb_m"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["nb_m"])})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Naive Bayes (Gaussian)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["nb_g"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["nb_g"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["nb_g"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["nb_g"])})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Naive Bayes (Complement)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["nb_c"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["nb_c"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["nb_c"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["nb_c"])})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Support Vector Machines (sigmoid)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["svm"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["svm"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["svm"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["svm"])})
        # thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (linear)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_l"])) + "," + str(max(notears_l2_linear_dict_scores["svm_l"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_l"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_l"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_l"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_l"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_l"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_l"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_l"])) + "}"})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Support Vector Machines (poly)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["svm_po"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["svm_po"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["svm_po"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["svm_po"])})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'Support Vector Machines (rbf)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["svm_r"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["svm_r"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["svm_r"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["svm_r"])})
        # thewriter.writerow({'Algorithm': 'NO TEARS (Loss-L2)', 'Model': 'Support Vector Machines (precomputed)','Linear': str(round(mean(notears_l2_linear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_linear_dict_scores["svm_pr"])) + "," + str(max(notears_l2_linear_dict_scores["svm_pr"])) + "}",'Non-linear': str(round(mean(notears_l2_nonlinear_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_nonlinear_dict_scores["svm_pr"])) + "," + str(max(notears_l2_nonlinear_dict_scores["svm_pr"])) + "}",'Sparsity': str(round(mean(notears_l2_sparse_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_sparse_dict_scores["svm_pr"])) + "," + str(max(notears_l2_sparse_dict_scores["svm_pr"])) + "}",'Dimensionality': str(round(mean(notears_l2_dimension_dict_scores["svm_pr"]), 2)) + " {" + str(min(notears_l2_dimension_dict_scores["svm_pr"])) + "," + str(max(notears_l2_dimension_dict_scores["svm_pr"])) + "}"})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'K Nearest Neighbor (uniform)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["knn"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["knn"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["knn"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["knn"])})
        thewriter.writerow({'Algorithm': 'PGMPY (TREE)', 'Model': 'K Nearest Neighbor (distance)',
                            'Linear': str(pgmpy_tree_linear_dict_scores["knn_d"]),
                            'Non-linear': str(pgmpy_tree_nonlinear_dict_scores["knn_d"]),
                            'Sparsity': str(pgmpy_tree_sparse_dict_scores["knn_d"]),
                            'Dimensionality': str(pgmpy_tree_dimension_dict_scores["knn_d"])})

write_learned_to_csv()

def write_real_to_csv():
    experiments = ['Model', 'Linear', 'Non-linear', 'Sparsity', 'Dimensionality']
    with open('real_experiments_summary.csv', 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Linear', 'Non-linear', 'Sparsity', 'Dimensionality']
        thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        thewriter.writeheader()
        thewriter.writerow({'Model': 'Decision Tree (gini)','Linear': str(real_linear_dt_scores), 'Non-linear': str(real_nonlinear_dt_scores), 'Sparsity': str(real_sparse_dt_scores), 'Dimensionality': str(real_dimension_dt_scores)})
        thewriter.writerow({'Model': 'Decision Tree (entropy)', 'Linear': str(real_linear_dt_entropy_scores),'Non-linear': str(real_nonlinear_dt_entropy_scores),'Sparsity': str(real_sparse_dt_entropy_scores),'Dimensionality': str(real_dimension_dt_entropy_scores)})
        thewriter.writerow({'Model': 'Random Forest (gini)', 'Linear': str(real_linear_rf_scores), 'Non-linear': str(real_nonlinear_rf_scores), 'Sparsity': str(real_sparse_rf_scores), 'Dimensionality': str(real_dimension_rf_scores)})
        thewriter.writerow({'Model': 'Random Forest (entropy)', 'Linear': str(real_linear_rf_entropy_scores),'Non-linear': str(real_nonlinear_rf_entropy_scores),'Sparsity': str(real_sparse_rf_entropy_scores),'Dimensionality': str(real_dimension_rf_entropy_scores)})
        thewriter.writerow({'Model': 'Logistic Regression (penalty-none)', 'Linear': str(real_linear_lr_scores), 'Non-linear': str(real_nonlinear_lr_scores), 'Sparsity': str(real_sparse_lr_scores), 'Dimensionality': str(real_dimension_lr_scores)})
        thewriter.writerow({'Model': 'Logistic Regression (l1)', 'Linear': str(real_linear_lr_l1_scores),'Non-linear': str(real_nonlinear_lr_l1_scores),'Sparsity': str(real_sparse_lr_l1_scores),'Dimensionality': str(real_dimension_lr_l1_scores)})
        thewriter.writerow({'Model': 'Logistic Regression (l2)', 'Linear': str(real_linear_lr_l2_scores),'Non-linear': str(real_nonlinear_lr_l2_scores),'Sparsity': str(real_sparse_lr_l2_scores),'Dimensionality': str(real_dimension_lr_l2_scores)})
        thewriter.writerow({'Model': 'Logistic Regression (elasticnet)', 'Linear': str(real_linear_lr_elastic_scores),'Non-linear': str(real_nonlinear_lr_elastic_scores),'Sparsity': str(real_sparse_lr_elastic_scores),'Dimensionality': str(real_dimension_lr_elastic_scores)})
        thewriter.writerow({'Model': 'Naive Bayes (Bernoulli)', 'Linear': str(real_linear_gb_scores),'Non-linear': str(real_nonlinear_gb_scores), 'Sparsity': str(real_sparse_gb_scores), 'Dimensionality': str(real_dimension_gb_scores)})
        thewriter.writerow({'Model': 'Naive Bayes (Multinomial)', 'Linear': str(real_linear_gb_multi_scores),'Non-linear': str(real_nonlinear_gb_multi_scores) ,'Sparsity': str(real_sparse_gb_multi_scores),'Dimensionality': str(real_dimension_gb_multi_scores)})
        thewriter.writerow({'Model': 'Naive Bayes (Gaussian)','Linear': str(real_linear_gb_gaussian_scores),'Non-linear': str(real_nonlinear_gb_gaussian_scores),'Sparsity': str(real_sparse_gb_gaussian_scores),'Dimensionality': str(real_dimension_gb_gaussian_scores)})
        thewriter.writerow({'Model': 'Naive Bayes (Complement)','Linear': str(real_linear_gb_complement_scores),'Non-linear': str(real_nonlinear_gb_complement_scores),'Sparsity': str(real_sparse_gb_complement_scores) ,'Dimensionality': str(real_dimension_gb_complement_scores)})
        thewriter.writerow({'Model': 'Support Vector Machines (sigmoid)', 'Linear': str(real_linear_svm_scores),'Non-linear': str(real_nonlinear_svm_scores), 'Sparsity': str(real_sparse_svm_scores), 'Dimensionality': str(real_dimension_svm_scores)})
        #thewriter.writerow({'Model': 'Support Vector Machines (linear)','Linear': str(mean(real_linear_svm_linear_scores)) + " {" + str(min(real_linear_svm_linear_scores)) + "," + str(max(real_linear_svm_linear_scores)) + "}",'Non-linear': str(mean(real_nonlinear_svm_linear_scores)) + " {" + str(min(real_nonlinear_svm_linear_scores)) + "," + str(max(real_nonlinear_svm_linear_scores)) + "}",'Sparsity': str(mean(real_sparse_svm_linear_scores)) + " {" + str(min(real_sparse_svm_linear_scores)) + "," + str(max(real_sparse_svm_linear_scores)) + "}",'Dimensionality': str(mean(real_dimension_svm_linear_scores)) + " {" + str(min(real_dimension_svm_linear_scores)) + "," + str(max(real_dimension_svm_linear_scores)) + "}"})
        thewriter.writerow({'Model': 'Support Vector Machines (poly)','Linear': str(real_linear_svm_poly_scores),'Non-linear': str(real_nonlinear_svm_poly_scores) ,'Sparsity': str(real_sparse_svm_poly_scores),'Dimensionality': str(real_dimension_svm_poly_scores)})
        thewriter.writerow({'Model': 'Support Vector Machines (rbf)','Linear': str(real_linear_svm_rbf_scores),'Non-linear': str(real_nonlinear_svm_rbf_scores) ,'Sparsity': str(real_sparse_svm_rbf_scores),'Dimensionality': str(real_dimension_svm_rbf_scores)})
        #thewriter.writerow({'Model': 'Support Vector Machines (precomputed)','Linear': str(mean(real_linear_svm_precomputed_scores)) + " {" + str(min(real_linear_svm_precomputed_scores)) + "," + str(max(real_linear_svm_precomputed_scores)) + "}",'Non-linear': str(mean(real_nonlinear_svm_precomputed_scores)) + " {" + str(min(real_nonlinear_svm_precomputed_scores)) + "," + str(max(real_nonlinear_svm_precomputed_scores)) + "}",'Sparsity': str(mean(real_sparse_svm_precomputed_scores)) + " {" + str(min(real_sparse_svm_precomputed_scores)) + "," + str(max(real_sparse_svm_precomputed_scores)) + "}",'Dimensionality': str(mean(real_dimension_svm_precomputed_scores)) + " {" + str(min(real_dimension_svm_precomputed_scores)) + "," + str(max(real_dimension_svm_precomputed_scores)) + "}"})
        thewriter.writerow({'Model': 'K Nearest Neighbor (uniform)', 'Linear': str(real_linear_knn_scores),'Non-linear': str(real_nonlinear_knn_scores), 'Sparsity': str(real_sparse_knn_scores), 'Dimensionality': str(real_dimension_knn_scores)})
        thewriter.writerow({'Model': 'K Nearest Neighbor (distance)', 'Linear': str(real_linear_knn_distance_scores),'Non-linear': str(real_nonlinear_knn_distance_scores), 'Sparsity': str(real_sparse_knn_distance_scores), 'Dimensionality': str(real_dimension_knn_distance_scores)})

write_real_to_csv()

def write_real_to_figures():

    # Produce Linear Problem by Library on Problem (test set from real world)
    # Group by figure
    labels = ['DT_G', 'DT_E', 'RF_G', 'RF_E', 'LR', 'LR_L1', 'LR_L2', 'LR_E', 'NB_B', 'NB_G', 'NB_M', 'NB_C', 'SVM_S',
              'SVM_P', 'SVM_R', 'KNN_W', 'KNN_D']
    bn_means = [bnlearn_linear_dict_scores["dt"], bnlearn_linear_dict_scores["dt_e"], bnlearn_linear_dict_scores["rf"], bnlearn_linear_dict_scores["rf_e"], bnlearn_linear_dict_scores["lr"], bnlearn_linear_dict_scores["lr_l1"], bnlearn_linear_dict_scores["lr_l2"], bnlearn_linear_dict_scores["lr_e"], bnlearn_linear_dict_scores["nb"], bnlearn_linear_dict_scores["nb_g"], bnlearn_linear_dict_scores["nb_m"], bnlearn_linear_dict_scores["nb_c"], bnlearn_linear_dict_scores["svm"], bnlearn_linear_dict_scores["svm_po"], bnlearn_linear_dict_scores["svm_r"], bnlearn_linear_dict_scores["knn"], bnlearn_linear_dict_scores["knn_d"]]
    bn_tabu_means = [bnlearn_tabu_linear_dict_scores["dt"], bnlearn_tabu_linear_dict_scores["dt_e"],
                bnlearn_tabu_linear_dict_scores["rf"], bnlearn_tabu_linear_dict_scores["rf_e"],
                bnlearn_tabu_linear_dict_scores["lr"], bnlearn_tabu_linear_dict_scores["lr_l1"],
                bnlearn_tabu_linear_dict_scores["lr_l2"], bnlearn_tabu_linear_dict_scores["lr_e"],
                bnlearn_tabu_linear_dict_scores["nb"], bnlearn_tabu_linear_dict_scores["nb_g"],
                bnlearn_tabu_linear_dict_scores["nb_m"], bnlearn_tabu_linear_dict_scores["nb_c"],
                bnlearn_tabu_linear_dict_scores["svm"], bnlearn_tabu_linear_dict_scores["svm_po"],
                bnlearn_tabu_linear_dict_scores["svm_r"], bnlearn_tabu_linear_dict_scores["knn"],
                bnlearn_tabu_linear_dict_scores["knn_d"]]
    bn_pc_means = [bnlearn_pc_linear_dict_scores["dt"], bnlearn_pc_linear_dict_scores["dt_e"],
                bnlearn_pc_linear_dict_scores["rf"], bnlearn_pc_linear_dict_scores["rf_e"],
                bnlearn_pc_linear_dict_scores["lr"], bnlearn_pc_linear_dict_scores["lr_l1"],
                bnlearn_pc_linear_dict_scores["lr_l2"], bnlearn_pc_linear_dict_scores["lr_e"],
                bnlearn_pc_linear_dict_scores["nb"], bnlearn_pc_linear_dict_scores["nb_g"],
                bnlearn_pc_linear_dict_scores["nb_m"], bnlearn_pc_linear_dict_scores["nb_c"],
                bnlearn_pc_linear_dict_scores["svm"], bnlearn_pc_linear_dict_scores["svm_po"],
                bnlearn_pc_linear_dict_scores["svm_r"], bnlearn_pc_linear_dict_scores["knn"],
                bnlearn_pc_linear_dict_scores["knn_d"]]
    bn_mmhc_means = [bnlearn_mmhc_linear_dict_scores["dt"], bnlearn_mmhc_linear_dict_scores["dt_e"],
                bnlearn_mmhc_linear_dict_scores["rf"], bnlearn_mmhc_linear_dict_scores["rf_e"],
                bnlearn_mmhc_linear_dict_scores["lr"], bnlearn_mmhc_linear_dict_scores["lr_l1"],
                bnlearn_mmhc_linear_dict_scores["lr_l2"], bnlearn_mmhc_linear_dict_scores["lr_e"],
                bnlearn_mmhc_linear_dict_scores["nb"], bnlearn_mmhc_linear_dict_scores["nb_g"],
                bnlearn_mmhc_linear_dict_scores["nb_m"], bnlearn_mmhc_linear_dict_scores["nb_c"],
                bnlearn_mmhc_linear_dict_scores["svm"], bnlearn_mmhc_linear_dict_scores["svm_po"],
                bnlearn_mmhc_linear_dict_scores["svm_r"], bnlearn_mmhc_linear_dict_scores["knn"],
                bnlearn_mmhc_linear_dict_scores["knn_d"]]
    bn_rsmax2_means = [bnlearn_rsmax2_linear_dict_scores["dt"], bnlearn_rsmax2_linear_dict_scores["dt_e"],
                bnlearn_rsmax2_linear_dict_scores["rf"], bnlearn_rsmax2_linear_dict_scores["rf_e"],
                bnlearn_rsmax2_linear_dict_scores["lr"], bnlearn_rsmax2_linear_dict_scores["lr_l1"],
                bnlearn_rsmax2_linear_dict_scores["lr_l2"], bnlearn_rsmax2_linear_dict_scores["lr_e"],
                bnlearn_rsmax2_linear_dict_scores["nb"], bnlearn_rsmax2_linear_dict_scores["nb_g"],
                bnlearn_rsmax2_linear_dict_scores["nb_m"], bnlearn_rsmax2_linear_dict_scores["nb_c"],
                bnlearn_rsmax2_linear_dict_scores["svm"], bnlearn_rsmax2_linear_dict_scores["svm_po"],
                bnlearn_rsmax2_linear_dict_scores["svm_r"], bnlearn_rsmax2_linear_dict_scores["knn"],
                bnlearn_rsmax2_linear_dict_scores["knn_d"]]
    bn_h2pc_means = [bnlearn_h2pc_linear_dict_scores["dt"], bnlearn_h2pc_linear_dict_scores["dt_e"],
                bnlearn_h2pc_linear_dict_scores["rf"], bnlearn_h2pc_linear_dict_scores["rf_e"],
                bnlearn_h2pc_linear_dict_scores["lr"], bnlearn_h2pc_linear_dict_scores["lr_l1"],
                bnlearn_h2pc_linear_dict_scores["lr_l2"], bnlearn_h2pc_linear_dict_scores["lr_e"],
                bnlearn_h2pc_linear_dict_scores["nb"], bnlearn_h2pc_linear_dict_scores["nb_g"],
                bnlearn_h2pc_linear_dict_scores["nb_m"], bnlearn_h2pc_linear_dict_scores["nb_c"],
                bnlearn_h2pc_linear_dict_scores["svm"], bnlearn_h2pc_linear_dict_scores["svm_po"],
                bnlearn_h2pc_linear_dict_scores["svm_r"], bnlearn_h2pc_linear_dict_scores["knn"],
                bnlearn_h2pc_linear_dict_scores["knn_d"]]

    nt_means = [notears_linear_dict_scores["dt"], notears_linear_dict_scores["dt_e"], notears_linear_dict_scores["rf"], notears_linear_dict_scores["rf_e"], notears_linear_dict_scores["lr"], notears_linear_dict_scores["lr_l1"], notears_linear_dict_scores["lr_l2"], notears_linear_dict_scores["lr_e"], notears_linear_dict_scores["nb"], notears_linear_dict_scores["nb_g"], notears_linear_dict_scores["nb_m"], notears_linear_dict_scores["nb_c"], notears_linear_dict_scores["svm"], notears_linear_dict_scores["svm_po"], notears_linear_dict_scores["svm_r"], notears_linear_dict_scores["knn"], notears_linear_dict_scores["knn_d"]]
    nt_l2_means = [notears_l2_linear_dict_scores["dt"], notears_l2_linear_dict_scores["dt_e"],
                notears_l2_linear_dict_scores["rf"], notears_l2_linear_dict_scores["rf_e"],
                notears_l2_linear_dict_scores["lr"], notears_l2_linear_dict_scores["lr_l1"],
                notears_l2_linear_dict_scores["lr_l2"], notears_l2_linear_dict_scores["lr_e"],
                notears_l2_linear_dict_scores["nb"], notears_l2_linear_dict_scores["nb_g"],
                notears_l2_linear_dict_scores["nb_m"], notears_l2_linear_dict_scores["nb_c"],
                notears_l2_linear_dict_scores["svm"], notears_l2_linear_dict_scores["svm_po"],
                notears_l2_linear_dict_scores["svm_r"], notears_l2_linear_dict_scores["knn"],
                notears_l2_linear_dict_scores["knn_d"]]
    nt_p_means = [notears_poisson_linear_dict_scores["dt"], notears_poisson_linear_dict_scores["dt_e"],
                notears_poisson_linear_dict_scores["rf"], notears_poisson_linear_dict_scores["rf_e"],
                notears_poisson_linear_dict_scores["lr"], notears_poisson_linear_dict_scores["lr_l1"],
                notears_poisson_linear_dict_scores["lr_l2"], notears_poisson_linear_dict_scores["lr_e"],
                notears_poisson_linear_dict_scores["nb"], notears_poisson_linear_dict_scores["nb_g"],
                notears_poisson_linear_dict_scores["nb_m"], notears_poisson_linear_dict_scores["nb_c"],
                notears_poisson_linear_dict_scores["svm"], notears_poisson_linear_dict_scores["svm_po"],
                notears_poisson_linear_dict_scores["svm_r"], notears_poisson_linear_dict_scores["knn"],
                notears_poisson_linear_dict_scores["knn_d"]]

    p_means = [pomegranate_exact_linear_dict_scores["dt"], pomegranate_exact_linear_dict_scores["dt_e"], pomegranate_exact_linear_dict_scores["rf"], pomegranate_exact_linear_dict_scores["rf_e"], pomegranate_exact_linear_dict_scores["lr"], pomegranate_exact_linear_dict_scores["lr_l1"], pomegranate_exact_linear_dict_scores["lr_l2"], pomegranate_exact_linear_dict_scores["lr_e"], pomegranate_exact_linear_dict_scores["nb"], pomegranate_exact_linear_dict_scores["nb_g"], pomegranate_exact_linear_dict_scores["nb_m"], pomegranate_exact_linear_dict_scores["nb_c"], pomegranate_exact_linear_dict_scores["svm"], pomegranate_exact_linear_dict_scores["svm_po"], pomegranate_exact_linear_dict_scores["svm_r"], pomegranate_exact_linear_dict_scores["knn"], pomegranate_exact_linear_dict_scores["knn_d"]]
    p_g_means = [pomegranate_greedy_linear_dict_scores["dt"],
               pomegranate_greedy_linear_dict_scores["dt_e"],
               pomegranate_greedy_linear_dict_scores["rf"],
               pomegranate_greedy_linear_dict_scores["rf_e"],
               pomegranate_greedy_linear_dict_scores["lr"],
               pomegranate_greedy_linear_dict_scores["lr_l1"],
               pomegranate_greedy_linear_dict_scores["lr_l2"],
               pomegranate_greedy_linear_dict_scores["lr_e"],
               pomegranate_greedy_linear_dict_scores["nb"],
               pomegranate_greedy_linear_dict_scores["nb_g"],
               pomegranate_greedy_linear_dict_scores["nb_m"],
               pomegranate_greedy_linear_dict_scores["nb_c"],
               pomegranate_greedy_linear_dict_scores["svm"],
               pomegranate_greedy_linear_dict_scores["svm_po"],
               pomegranate_greedy_linear_dict_scores["svm_r"],
               pomegranate_greedy_linear_dict_scores["knn"],
               pomegranate_greedy_linear_dict_scores["knn_d"]]

    pgmpy_tree_means = [pgmpy_tree_linear_dict_scores["dt"],
                      pgmpy_tree_linear_dict_scores["dt_e"],
                      pgmpy_tree_linear_dict_scores["rf"],
                      pgmpy_tree_linear_dict_scores["rf_e"],
                      pgmpy_tree_linear_dict_scores["lr"],
                      pgmpy_tree_linear_dict_scores["lr_l1"],
                      pgmpy_tree_linear_dict_scores["lr_l2"],
                      pgmpy_tree_linear_dict_scores["lr_e"],
                      pgmpy_tree_linear_dict_scores["nb"],
                      pgmpy_tree_linear_dict_scores["nb_g"],
                      pgmpy_tree_linear_dict_scores["nb_m"],
                      pgmpy_tree_linear_dict_scores["nb_c"],
                      pgmpy_tree_linear_dict_scores["svm"],
                      pgmpy_tree_linear_dict_scores["svm_po"],
                      pgmpy_tree_linear_dict_scores["svm_r"],
                      pgmpy_tree_linear_dict_scores["knn"],
                      pgmpy_tree_linear_dict_scores["knn_d"]]
    pgmpy_hc_means = [pgmpy_hc_linear_dict_scores["dt"],
                        pgmpy_hc_linear_dict_scores["dt_e"],
                        pgmpy_hc_linear_dict_scores["rf"],
                        pgmpy_hc_linear_dict_scores["rf_e"],
                        pgmpy_hc_linear_dict_scores["lr"],
                        pgmpy_hc_linear_dict_scores["lr_l1"],
                        pgmpy_hc_linear_dict_scores["lr_l2"],
                        pgmpy_hc_linear_dict_scores["lr_e"],
                        pgmpy_hc_linear_dict_scores["nb"],
                        pgmpy_hc_linear_dict_scores["nb_g"],
                        pgmpy_hc_linear_dict_scores["nb_m"],
                        pgmpy_hc_linear_dict_scores["nb_c"],
                        pgmpy_hc_linear_dict_scores["svm"],
                        pgmpy_hc_linear_dict_scores["svm_po"],
                        pgmpy_hc_linear_dict_scores["svm_r"],
                        pgmpy_hc_linear_dict_scores["knn"],
                        pgmpy_hc_linear_dict_scores["knn_d"]]
    pgmpy_mmhc_means = [pgmpy_mmhc_linear_dict_scores["dt"],
                      pgmpy_mmhc_linear_dict_scores["dt_e"],
                      pgmpy_mmhc_linear_dict_scores["rf"],
                      pgmpy_mmhc_linear_dict_scores["rf_e"],
                      pgmpy_mmhc_linear_dict_scores["lr"],
                      pgmpy_mmhc_linear_dict_scores["lr_l1"],
                      pgmpy_mmhc_linear_dict_scores["lr_l2"],
                      pgmpy_mmhc_linear_dict_scores["lr_e"],
                      pgmpy_mmhc_linear_dict_scores["nb"],
                      pgmpy_mmhc_linear_dict_scores["nb_g"],
                      pgmpy_mmhc_linear_dict_scores["nb_m"],
                      pgmpy_mmhc_linear_dict_scores["nb_c"],
                      pgmpy_mmhc_linear_dict_scores["svm"],
                      pgmpy_mmhc_linear_dict_scores["svm_po"],
                      pgmpy_mmhc_linear_dict_scores["svm_r"],
                      pgmpy_mmhc_linear_dict_scores["knn"],
                      pgmpy_mmhc_linear_dict_scores["knn_d"]]

    plt.rcParams["figure.figsize"] = [18, 18]
    plt.rcParams["figure.autolayout"] = True

    x_axis = np.arange(len(labels))
    w = 0.05  # the width of the bars

    plt.bar(x_axis +w, bn_means, width=0.05, label = "BN_LEARN (HC)", color="lightsteelblue")
    plt.bar(x_axis + w * 2, nt_means, width=0.05, label="BN_LEARN (TABU)", color="cornflowerblue")
    plt.bar(x_axis + w * 3, bn_pc_means, width=0.05, label="BN_LEARN (PC)", color="royalblue")
    plt.bar(x_axis + w * 4, bn_mmhc_means, width=0.05, label="BN_LEARN (MMHC)", color="blue")
    plt.bar(x_axis + w * 5, bn_rsmax2_means, width=0.05, label="BN_LEARN (RSMAX2)", color="mediumblue")
    plt.bar(x_axis + w * 6, bn_h2pc_means, width=0.05, label="BN_LEARN (H2PC)", color="navy")
    plt.bar(x_axis +w*7, nt_means, width=0.05, label="NO_TEARS (logistic)", color="limegreen")
    plt.bar(x_axis +w*8, nt_l2_means, width=0.05, label="NO_TEARS (l2)", color="forestgreen")
    plt.bar(x_axis + w * 9, nt_p_means, width=0.05, label="NO_TEARS (poisson)", color="darkgreen")
    plt.bar(x_axis + w * 10, p_means, width=0.05, label="POMEGRANATE (exact)", color="darkviolet")
    plt.bar(x_axis + w * 11, p_g_means, width=0.05, label="POMEGRANATE (greed)", color="rebeccapurple")
    plt.bar(x_axis + w * 12, pgmpy_mmhc_means, width=0.05, label="PGMPY (MMHC)", color="#FA8072")
    plt.bar(x_axis + w * 13, pgmpy_hc_means, width=0.05, label="PGMPY (HC)", color="#FF2400")
    plt.bar(x_axis + w * 14, pgmpy_tree_means, width=0.05, label="PGMPY (TREE)", color="#7C0A02")

    plt.xticks(x_axis, labels)
    plt.legend()
    plt.style.use("fivethirtyeight")
    plt.ylabel('Accuracy')
    plt.xlabel('ML Technique', labelpad=15)
    plt.title('Linear Problem - Performance by library on ML technique')
    #plt.ylim(0.6, 1)
    #plt.tick_params(rotation=45)
    plt.savefig('pipeline_summary_benchmark_for_linear_by_library_groupbar.png', bbox_inches='tight')
    plt.show()

    # Produce Non-Linear Problem by Library on Problem
    # Group by figure
    labels = ['DT_G', 'DT_E', 'RF_G', 'RF_E', 'LR', 'LR_L1', 'LR_L2', 'LR_E', 'NB_B', 'NB_G', 'NB_M', 'NB_C', 'SVM_S',
              'SVM_P', 'SVM_R', 'KNN_W', 'KNN_D']
    bn_non_means = [bnlearn_nonlinear_dict_scores["dt"], bnlearn_nonlinear_dict_scores["dt_e"],
                bnlearn_nonlinear_dict_scores["rf"], bnlearn_nonlinear_dict_scores["rf_e"],
                bnlearn_nonlinear_dict_scores["lr"], bnlearn_nonlinear_dict_scores["lr_l1"],
                bnlearn_nonlinear_dict_scores["lr_l2"], bnlearn_nonlinear_dict_scores["lr_e"],
                bnlearn_nonlinear_dict_scores["nb"], bnlearn_nonlinear_dict_scores["nb_g"],
                bnlearn_nonlinear_dict_scores["nb_m"], bnlearn_nonlinear_dict_scores["nb_c"],
                bnlearn_nonlinear_dict_scores["svm"], bnlearn_nonlinear_dict_scores["svm_po"],
                bnlearn_nonlinear_dict_scores["svm_r"], bnlearn_nonlinear_dict_scores["knn"],
                bnlearn_nonlinear_dict_scores["knn_d"]]
    bn_tabu_non_means = [bnlearn_tabu_nonlinear_dict_scores["dt"],
                     bnlearn_tabu_nonlinear_dict_scores["dt_e"],
                     bnlearn_tabu_nonlinear_dict_scores["rf"],
                     bnlearn_tabu_nonlinear_dict_scores["rf_e"],
                     bnlearn_tabu_nonlinear_dict_scores["lr"],
                     bnlearn_tabu_nonlinear_dict_scores["lr_l1"],
                     bnlearn_tabu_nonlinear_dict_scores["lr_l2"],
                     bnlearn_tabu_nonlinear_dict_scores["lr_e"],
                     bnlearn_tabu_nonlinear_dict_scores["nb"],
                     bnlearn_tabu_nonlinear_dict_scores["nb_g"],
                     bnlearn_tabu_nonlinear_dict_scores["nb_m"],
                     bnlearn_tabu_nonlinear_dict_scores["nb_c"],
                     bnlearn_tabu_nonlinear_dict_scores["svm"],
                     bnlearn_tabu_nonlinear_dict_scores["svm_po"],
                     bnlearn_tabu_nonlinear_dict_scores["svm_r"],
                     bnlearn_tabu_nonlinear_dict_scores["knn"],
                     bnlearn_tabu_nonlinear_dict_scores["knn_d"]]
    bn_mmhc_non_means = [bnlearn_mmhc_nonlinear_dict_scores["dt"],
                     bnlearn_mmhc_nonlinear_dict_scores["dt_e"],
                     bnlearn_mmhc_nonlinear_dict_scores["rf"],
                     bnlearn_mmhc_nonlinear_dict_scores["rf_e"],
                     bnlearn_mmhc_nonlinear_dict_scores["lr"],
                     bnlearn_mmhc_nonlinear_dict_scores["lr_l1"],
                     bnlearn_mmhc_nonlinear_dict_scores["lr_l2"],
                     bnlearn_mmhc_nonlinear_dict_scores["lr_e"],
                     bnlearn_mmhc_nonlinear_dict_scores["nb"],
                     bnlearn_mmhc_nonlinear_dict_scores["nb_g"],
                     bnlearn_mmhc_nonlinear_dict_scores["nb_m"],
                     bnlearn_mmhc_nonlinear_dict_scores["nb_c"],
                     bnlearn_mmhc_nonlinear_dict_scores["svm"],
                     bnlearn_mmhc_nonlinear_dict_scores["svm_po"],
                     bnlearn_mmhc_nonlinear_dict_scores["svm_r"],
                     bnlearn_mmhc_nonlinear_dict_scores["knn"],
                     bnlearn_mmhc_nonlinear_dict_scores["knn_d"]]
    bn_rsmax2_non_means = [bnlearn_rsmax2_nonlinear_dict_scores["dt"],
                       bnlearn_rsmax2_nonlinear_dict_scores["dt_e"],
                       bnlearn_rsmax2_nonlinear_dict_scores["rf"],
                       bnlearn_rsmax2_nonlinear_dict_scores["rf_e"],
                       bnlearn_rsmax2_nonlinear_dict_scores["lr"],
                       bnlearn_rsmax2_nonlinear_dict_scores["lr_l1"],
                       bnlearn_rsmax2_nonlinear_dict_scores["lr_l2"],
                       bnlearn_rsmax2_nonlinear_dict_scores["lr_e"],
                       bnlearn_rsmax2_nonlinear_dict_scores["nb"],
                       bnlearn_rsmax2_nonlinear_dict_scores["nb_g"],
                       bnlearn_rsmax2_nonlinear_dict_scores["nb_m"],
                       bnlearn_rsmax2_nonlinear_dict_scores["nb_c"],
                       bnlearn_rsmax2_nonlinear_dict_scores["svm"],
                       bnlearn_rsmax2_nonlinear_dict_scores["svm_po"],
                       bnlearn_rsmax2_nonlinear_dict_scores["svm_r"],
                       bnlearn_rsmax2_nonlinear_dict_scores["knn"],
                       bnlearn_rsmax2_nonlinear_dict_scores["knn_d"]]
    bn_h2pc_non_means = [bnlearn_h2pc_nonlinear_dict_scores["dt"],
                     bnlearn_h2pc_nonlinear_dict_scores["dt_e"],
                     bnlearn_h2pc_nonlinear_dict_scores["rf"],
                     bnlearn_h2pc_nonlinear_dict_scores["rf_e"],
                     bnlearn_h2pc_nonlinear_dict_scores["lr"],
                     bnlearn_h2pc_nonlinear_dict_scores["lr_l1"],
                     bnlearn_h2pc_nonlinear_dict_scores["lr_l2"],
                     bnlearn_h2pc_nonlinear_dict_scores["lr_e"],
                     bnlearn_h2pc_nonlinear_dict_scores["nb"],
                     bnlearn_h2pc_nonlinear_dict_scores["nb_g"],
                     bnlearn_h2pc_nonlinear_dict_scores["nb_m"],
                     bnlearn_h2pc_nonlinear_dict_scores["nb_c"],
                     bnlearn_h2pc_nonlinear_dict_scores["svm"],
                     bnlearn_h2pc_nonlinear_dict_scores["svm_po"],
                     bnlearn_h2pc_nonlinear_dict_scores["svm_r"],
                     bnlearn_h2pc_nonlinear_dict_scores["knn"],
                     bnlearn_h2pc_nonlinear_dict_scores["knn_d"]]

    nt_non_means = [notears_nonlinear_dict_scores["dt"], notears_nonlinear_dict_scores["dt_e"],
                notears_nonlinear_dict_scores["rf"], notears_nonlinear_dict_scores["rf_e"],
                notears_nonlinear_dict_scores["lr"], notears_nonlinear_dict_scores["lr_l1"],
                notears_nonlinear_dict_scores["lr_l2"], notears_nonlinear_dict_scores["lr_e"],
                notears_nonlinear_dict_scores["nb"], notears_nonlinear_dict_scores["nb_g"],
                notears_nonlinear_dict_scores["nb_m"], notears_nonlinear_dict_scores["nb_c"],
                notears_nonlinear_dict_scores["svm"], notears_nonlinear_dict_scores["svm_po"],
                notears_nonlinear_dict_scores["svm_r"], notears_nonlinear_dict_scores["knn"],
                notears_nonlinear_dict_scores["knn_d"]]
    nt_l2_non_means = [notears_l2_nonlinear_dict_scores["dt"],
                   notears_l2_nonlinear_dict_scores["dt_e"],
                   notears_l2_nonlinear_dict_scores["rf"],
                   notears_l2_nonlinear_dict_scores["rf_e"],
                   notears_l2_nonlinear_dict_scores["lr"],
                   notears_l2_nonlinear_dict_scores["lr_l1"],
                   notears_l2_nonlinear_dict_scores["lr_l2"],
                   notears_l2_nonlinear_dict_scores["lr_e"],
                   notears_l2_nonlinear_dict_scores["nb"],
                   notears_l2_nonlinear_dict_scores["nb_g"],
                   notears_l2_nonlinear_dict_scores["nb_m"],
                   notears_l2_nonlinear_dict_scores["nb_c"],
                   notears_l2_nonlinear_dict_scores["svm"],
                   notears_l2_nonlinear_dict_scores["svm_po"],
                   notears_l2_nonlinear_dict_scores["svm_r"],
                   notears_l2_nonlinear_dict_scores["knn"],
                   notears_l2_nonlinear_dict_scores["knn_d"]]
    nt_p_non_means = [notears_poisson_nonlinear_dict_scores["dt"],
                  notears_poisson_nonlinear_dict_scores["dt_e"],
                  notears_poisson_nonlinear_dict_scores["rf"],
                  notears_poisson_nonlinear_dict_scores["rf_e"],
                  notears_poisson_nonlinear_dict_scores["lr"],
                  notears_poisson_nonlinear_dict_scores["lr_l1"],
                  notears_poisson_nonlinear_dict_scores["lr_l2"],
                  notears_poisson_nonlinear_dict_scores["lr_e"],
                  notears_poisson_nonlinear_dict_scores["nb"],
                  notears_poisson_nonlinear_dict_scores["nb_g"],
                  notears_poisson_nonlinear_dict_scores["nb_m"],
                  notears_poisson_nonlinear_dict_scores["nb_c"],
                  notears_poisson_nonlinear_dict_scores["svm"],
                  notears_poisson_nonlinear_dict_scores["svm_po"],
                  notears_poisson_nonlinear_dict_scores["svm_r"],
                  notears_poisson_nonlinear_dict_scores["knn"],
                  notears_poisson_nonlinear_dict_scores["knn_d"]]

    p_non_means = [pomegranate_exact_nonlinear_dict_scores["dt"],
               pomegranate_exact_nonlinear_dict_scores["dt_e"],
               pomegranate_exact_nonlinear_dict_scores["rf"],
               pomegranate_exact_nonlinear_dict_scores["rf_e"],
               pomegranate_exact_nonlinear_dict_scores["lr"],
               pomegranate_exact_nonlinear_dict_scores["lr_l1"],
               pomegranate_exact_nonlinear_dict_scores["lr_l2"],
               pomegranate_exact_nonlinear_dict_scores["lr_e"],
               pomegranate_exact_nonlinear_dict_scores["nb"],
               pomegranate_exact_nonlinear_dict_scores["nb_g"],
               pomegranate_exact_nonlinear_dict_scores["nb_m"],
               pomegranate_exact_nonlinear_dict_scores["nb_c"],
               pomegranate_exact_nonlinear_dict_scores["svm"],
               pomegranate_exact_nonlinear_dict_scores["svm_po"],
               pomegranate_exact_nonlinear_dict_scores["svm_r"],
               pomegranate_exact_nonlinear_dict_scores["knn"],
               pomegranate_exact_nonlinear_dict_scores["knn_d"]]
    p_g_non_means = [pomegranate_greedy_nonlinear_dict_scores["dt"],
                 pomegranate_greedy_nonlinear_dict_scores["dt_e"],
                 pomegranate_greedy_nonlinear_dict_scores["rf"],
                 pomegranate_greedy_nonlinear_dict_scores["rf_e"],
                 pomegranate_greedy_nonlinear_dict_scores["lr"],
                 pomegranate_greedy_nonlinear_dict_scores["lr_l1"],
                 pomegranate_greedy_nonlinear_dict_scores["lr_l2"],
                 pomegranate_greedy_nonlinear_dict_scores["lr_e"],
                 pomegranate_greedy_nonlinear_dict_scores["nb"],
                 pomegranate_greedy_nonlinear_dict_scores["nb_g"],
                 pomegranate_greedy_nonlinear_dict_scores["nb_m"],
                 pomegranate_greedy_nonlinear_dict_scores["nb_c"],
                 pomegranate_greedy_nonlinear_dict_scores["svm"],
                 pomegranate_greedy_nonlinear_dict_scores["svm_po"],
                 pomegranate_greedy_nonlinear_dict_scores["svm_r"],
                 pomegranate_greedy_nonlinear_dict_scores["knn"],
                 pomegranate_greedy_nonlinear_dict_scores["knn_d"]]

    pgmpy_tree_non_means = [pgmpy_tree_nonlinear_dict_scores["dt"],
                        pgmpy_tree_nonlinear_dict_scores["dt_e"],
                        pgmpy_tree_nonlinear_dict_scores["rf"],
                        pgmpy_tree_nonlinear_dict_scores["rf_e"],
                        pgmpy_tree_nonlinear_dict_scores["lr"],
                        pgmpy_tree_nonlinear_dict_scores["lr_l1"],
                        pgmpy_tree_nonlinear_dict_scores["lr_l2"],
                        pgmpy_tree_nonlinear_dict_scores["lr_e"],
                        pgmpy_tree_nonlinear_dict_scores["nb"],
                        pgmpy_tree_nonlinear_dict_scores["nb_g"],
                        pgmpy_tree_nonlinear_dict_scores["nb_m"],
                        pgmpy_tree_nonlinear_dict_scores["nb_c"],
                        pgmpy_tree_nonlinear_dict_scores["svm"],
                        pgmpy_tree_nonlinear_dict_scores["svm_po"],
                        pgmpy_tree_nonlinear_dict_scores["svm_r"],
                        pgmpy_tree_nonlinear_dict_scores["knn"],
                        pgmpy_tree_nonlinear_dict_scores["knn_d"]]
    pgmpy_hc_non_means = [pgmpy_hc_nonlinear_dict_scores["dt"],
                      pgmpy_hc_nonlinear_dict_scores["dt_e"],
                      pgmpy_hc_nonlinear_dict_scores["rf"],
                      pgmpy_hc_nonlinear_dict_scores["rf_e"],
                      pgmpy_hc_nonlinear_dict_scores["lr"],
                      pgmpy_hc_nonlinear_dict_scores["lr_l1"],
                      pgmpy_hc_nonlinear_dict_scores["lr_l2"],
                      pgmpy_hc_nonlinear_dict_scores["lr_e"],
                      pgmpy_hc_nonlinear_dict_scores["nb"],
                      pgmpy_hc_nonlinear_dict_scores["nb_g"],
                      pgmpy_hc_nonlinear_dict_scores["nb_m"],
                      pgmpy_hc_nonlinear_dict_scores["nb_c"],
                      pgmpy_hc_nonlinear_dict_scores["svm"],
                      pgmpy_hc_nonlinear_dict_scores["svm_po"],
                      pgmpy_hc_nonlinear_dict_scores["svm_r"],
                      pgmpy_hc_nonlinear_dict_scores["knn"],
                      pgmpy_hc_nonlinear_dict_scores["knn_d"]]
    pgmpy_mmhc_non_means = [pgmpy_mmhc_nonlinear_dict_scores["dt"],
                        pgmpy_mmhc_nonlinear_dict_scores["dt_e"],
                        pgmpy_mmhc_nonlinear_dict_scores["rf"],
                        pgmpy_mmhc_nonlinear_dict_scores["rf_e"],
                        pgmpy_mmhc_nonlinear_dict_scores["lr"],
                        pgmpy_mmhc_nonlinear_dict_scores["lr_l1"],
                        pgmpy_mmhc_nonlinear_dict_scores["lr_l2"],
                        pgmpy_mmhc_nonlinear_dict_scores["lr_e"],
                        pgmpy_mmhc_nonlinear_dict_scores["nb"],
                        pgmpy_mmhc_nonlinear_dict_scores["nb_g"],
                        pgmpy_mmhc_nonlinear_dict_scores["nb_m"],
                        pgmpy_mmhc_nonlinear_dict_scores["nb_c"],
                        pgmpy_mmhc_nonlinear_dict_scores["svm"],
                        pgmpy_mmhc_nonlinear_dict_scores["svm_po"],
                        pgmpy_mmhc_nonlinear_dict_scores["svm_r"],
                        pgmpy_mmhc_nonlinear_dict_scores["knn"],
                        pgmpy_mmhc_nonlinear_dict_scores["knn_d"]]

    plt.rcParams["figure.figsize"] = [18, 18]
    plt.rcParams["figure.autolayout"] = True

    x_axis = np.arange(len(labels))
    w = 0.05  # the width of the bars

    plt.bar(x_axis + w, bn_non_means, width=0.05, label="BN_LEARN (HC)", color="lightsteelblue")
    plt.bar(x_axis + w * 2, nt_non_means, width=0.05, label="BN_LEARN (TABU)", color="cornflowerblue")
    plt.bar(x_axis + w * 3, bn_mmhc_non_means, width=0.05, label="BN_LEARN (MMHC)", color="blue")
    plt.bar(x_axis + w * 4, bn_rsmax2_non_means, width=0.05, label="BN_LEARN (RSMAX2)", color="mediumblue")
    plt.bar(x_axis + w * 5, bn_h2pc_non_means, width=0.05, label="BN_LEARN (H2PC)", color="navy")
    plt.bar(x_axis + w * 6, nt_non_means, width=0.05, label="NO_TEARS (logistic)", color="limegreen")
    plt.bar(x_axis + w * 7, nt_l2_non_means, width=0.05, label="NO_TEARS (l2)", color="forestgreen")
    plt.bar(x_axis + w * 8, nt_p_non_means, width=0.05, label="NO_TEARS (poisson)", color="darkgreen")
    plt.bar(x_axis + w * 9, p_non_means, width=0.05, label="POMEGRANATE (exact)", color="darkviolet")
    plt.bar(x_axis + w * 10, p_g_non_means, width=0.05, label="POMEGRANATE (greed)", color="rebeccapurple")
    plt.bar(x_axis + w * 11, pgmpy_mmhc_non_means, width=0.05, label="PGMPY (MMHC)", color="#FA8072")
    plt.bar(x_axis + w * 12, pgmpy_hc_non_means, width=0.05, label="PGMPY (HC)", color="#FF2400")
    plt.bar(x_axis + w * 13, pgmpy_tree_non_means, width=0.05, label="PGMPY (TREE)", color="#7C0A02")

    plt.xticks(x_axis, labels)
    plt.legend()
    plt.style.use("fivethirtyeight")
    plt.ylabel('Accuracy')
    plt.xlabel('ML Technique', labelpad=15)
    plt.title('Non-Linear Problem - Performance by library on ML technique')
    #plt.ylim(0.6, 1)
    # plt.tick_params(rotation=45)
    plt.savefig('pipeline_summary_benchmark_for_nonlinear_by_library_groupbar.png', bbox_inches='tight')
    plt.show()

    # Produce Sparse Problem by Library on Problem
    # Group by figure
    labels = ['DT_G', 'DT_E', 'RF_G', 'RF_E', 'LR', 'LR_L1', 'LR_L2', 'LR_E', 'NB_B', 'NB_G', 'NB_M', 'NB_C', 'SVM_S',
              'SVM_P', 'SVM_R', 'KNN_W', 'KNN_D']
    bn_sparse_means = [bnlearn_sparse_dict_scores["dt"], bnlearn_sparse_dict_scores["dt_e"], bnlearn_sparse_dict_scores["rf"], bnlearn_sparse_dict_scores["rf_e"], bnlearn_sparse_dict_scores["lr"], bnlearn_sparse_dict_scores["lr_l1"], bnlearn_sparse_dict_scores["lr_l2"], bnlearn_sparse_dict_scores["lr_e"], bnlearn_sparse_dict_scores["nb"], bnlearn_sparse_dict_scores["nb_g"], bnlearn_sparse_dict_scores["nb_m"], bnlearn_sparse_dict_scores["nb_c"], bnlearn_sparse_dict_scores["svm"], bnlearn_sparse_dict_scores["svm_po"], bnlearn_sparse_dict_scores["svm_r"], bnlearn_sparse_dict_scores["knn"], bnlearn_sparse_dict_scores["knn_d"]]
    bn_tabu_sparse_means = [bnlearn_tabu_sparse_dict_scores["dt"], bnlearn_tabu_sparse_dict_scores["dt_e"],
                bnlearn_tabu_sparse_dict_scores["rf"], bnlearn_tabu_sparse_dict_scores["rf_e"],
                bnlearn_tabu_sparse_dict_scores["lr"], bnlearn_tabu_sparse_dict_scores["lr_l1"],
                bnlearn_tabu_sparse_dict_scores["lr_l2"], bnlearn_tabu_sparse_dict_scores["lr_e"],
                bnlearn_tabu_sparse_dict_scores["nb"], bnlearn_tabu_sparse_dict_scores["nb_g"],
                bnlearn_tabu_sparse_dict_scores["nb_m"], bnlearn_tabu_sparse_dict_scores["nb_c"],
                bnlearn_tabu_sparse_dict_scores["svm"], bnlearn_tabu_sparse_dict_scores["svm_po"],
                bnlearn_tabu_sparse_dict_scores["svm_r"], bnlearn_tabu_sparse_dict_scores["knn"],
                bnlearn_tabu_sparse_dict_scores["knn_d"]]
    bn_mmhc_sparse_means = [bnlearn_mmhc_sparse_dict_scores["dt"], bnlearn_mmhc_sparse_dict_scores["dt_e"],
                bnlearn_mmhc_sparse_dict_scores["rf"], bnlearn_mmhc_sparse_dict_scores["rf_e"],
                bnlearn_mmhc_sparse_dict_scores["lr"], bnlearn_mmhc_sparse_dict_scores["lr_l1"],
                bnlearn_mmhc_sparse_dict_scores["lr_l2"], bnlearn_mmhc_sparse_dict_scores["lr_e"],
                bnlearn_mmhc_sparse_dict_scores["nb"], bnlearn_mmhc_sparse_dict_scores["nb_g"],
                bnlearn_mmhc_sparse_dict_scores["nb_m"], bnlearn_mmhc_sparse_dict_scores["nb_c"],
                bnlearn_mmhc_sparse_dict_scores["svm"], bnlearn_mmhc_sparse_dict_scores["svm_po"],
                bnlearn_mmhc_sparse_dict_scores["svm_r"], bnlearn_mmhc_sparse_dict_scores["knn"],
                bnlearn_mmhc_sparse_dict_scores["knn_d"]]
    bn_rsmax2_sparse_means = [bnlearn_rsmax2_sparse_dict_scores["dt"], bnlearn_rsmax2_sparse_dict_scores["dt_e"],
                bnlearn_rsmax2_sparse_dict_scores["rf"], bnlearn_rsmax2_sparse_dict_scores["rf_e"],
                bnlearn_rsmax2_sparse_dict_scores["lr"], bnlearn_rsmax2_sparse_dict_scores["lr_l1"],
                bnlearn_rsmax2_sparse_dict_scores["lr_l2"], bnlearn_rsmax2_sparse_dict_scores["lr_e"],
                bnlearn_rsmax2_sparse_dict_scores["nb"], bnlearn_rsmax2_sparse_dict_scores["nb_g"],
                bnlearn_rsmax2_sparse_dict_scores["nb_m"], bnlearn_rsmax2_sparse_dict_scores["nb_c"],
                bnlearn_rsmax2_sparse_dict_scores["svm"], bnlearn_rsmax2_sparse_dict_scores["svm_po"],
                bnlearn_rsmax2_sparse_dict_scores["svm_r"], bnlearn_rsmax2_sparse_dict_scores["knn"],
                bnlearn_rsmax2_sparse_dict_scores["knn_d"]]
    bn_h2pc_sparse_means = [bnlearn_h2pc_sparse_dict_scores["dt"], bnlearn_h2pc_sparse_dict_scores["dt_e"],
                bnlearn_h2pc_sparse_dict_scores["rf"], bnlearn_h2pc_sparse_dict_scores["rf_e"],
                bnlearn_h2pc_sparse_dict_scores["lr"], bnlearn_h2pc_sparse_dict_scores["lr_l1"],
                bnlearn_h2pc_sparse_dict_scores["lr_l2"], bnlearn_h2pc_sparse_dict_scores["lr_e"],
                bnlearn_h2pc_sparse_dict_scores["nb"], bnlearn_h2pc_sparse_dict_scores["nb_g"],
                bnlearn_h2pc_sparse_dict_scores["nb_m"], bnlearn_h2pc_sparse_dict_scores["nb_c"],
                bnlearn_h2pc_sparse_dict_scores["svm"], bnlearn_h2pc_sparse_dict_scores["svm_po"],
                bnlearn_h2pc_sparse_dict_scores["svm_r"], bnlearn_h2pc_sparse_dict_scores["knn"],
                bnlearn_h2pc_sparse_dict_scores["knn_d"]]

    nt_sparse_means = [notears_sparse_dict_scores["dt"], notears_sparse_dict_scores["dt_e"], notears_sparse_dict_scores["rf"], notears_sparse_dict_scores["rf_e"], notears_sparse_dict_scores["lr"], notears_sparse_dict_scores["lr_l1"], notears_sparse_dict_scores["lr_l2"], notears_sparse_dict_scores["lr_e"], notears_sparse_dict_scores["nb"], notears_sparse_dict_scores["nb_g"], notears_sparse_dict_scores["nb_m"], notears_sparse_dict_scores["nb_c"], notears_sparse_dict_scores["svm"], notears_sparse_dict_scores["svm_po"], notears_sparse_dict_scores["svm_r"], notears_sparse_dict_scores["knn"], notears_sparse_dict_scores["knn_d"]]
    nt_l2_sparse_means = [notears_l2_sparse_dict_scores["dt"], notears_l2_sparse_dict_scores["dt_e"],
                notears_l2_sparse_dict_scores["rf"], notears_l2_sparse_dict_scores["rf_e"],
                notears_l2_sparse_dict_scores["lr"], notears_l2_sparse_dict_scores["lr_l1"],
                notears_l2_sparse_dict_scores["lr_l2"], notears_l2_sparse_dict_scores["lr_e"],
                notears_l2_sparse_dict_scores["nb"], notears_l2_sparse_dict_scores["nb_g"],
                notears_l2_sparse_dict_scores["nb_m"], notears_l2_sparse_dict_scores["nb_c"],
                notears_l2_sparse_dict_scores["svm"], notears_l2_sparse_dict_scores["svm_po"],
                notears_l2_sparse_dict_scores["svm_r"], notears_l2_sparse_dict_scores["knn"],
                notears_l2_sparse_dict_scores["knn_d"]]
    nt_p_sparse_means = [notears_poisson_sparse_dict_scores["dt"], notears_poisson_sparse_dict_scores["dt_e"],
                notears_poisson_sparse_dict_scores["rf"], notears_poisson_sparse_dict_scores["rf_e"],
                notears_poisson_sparse_dict_scores["lr"], notears_poisson_sparse_dict_scores["lr_l1"],
                notears_poisson_sparse_dict_scores["lr_l2"], notears_poisson_sparse_dict_scores["lr_e"],
                notears_poisson_sparse_dict_scores["nb"], notears_poisson_sparse_dict_scores["nb_g"],
                notears_poisson_sparse_dict_scores["nb_m"], notears_poisson_sparse_dict_scores["nb_c"],
                notears_poisson_sparse_dict_scores["svm"], notears_poisson_sparse_dict_scores["svm_po"],
                notears_poisson_sparse_dict_scores["svm_r"], notears_poisson_sparse_dict_scores["knn"],
                notears_poisson_sparse_dict_scores["knn_d"]]

    p_sparse_means = [pomegranate_exact_sparse_dict_scores["dt"], pomegranate_exact_sparse_dict_scores["dt_e"], pomegranate_exact_sparse_dict_scores["rf"], pomegranate_exact_sparse_dict_scores["rf_e"], pomegranate_exact_sparse_dict_scores["lr"], pomegranate_exact_sparse_dict_scores["lr_l1"], pomegranate_exact_sparse_dict_scores["lr_l2"], pomegranate_exact_sparse_dict_scores["lr_e"], pomegranate_exact_sparse_dict_scores["nb"], pomegranate_exact_sparse_dict_scores["nb_g"], pomegranate_exact_sparse_dict_scores["nb_m"], pomegranate_exact_sparse_dict_scores["nb_c"], pomegranate_exact_sparse_dict_scores["svm"], pomegranate_exact_sparse_dict_scores["svm_po"], pomegranate_exact_sparse_dict_scores["svm_r"], pomegranate_exact_sparse_dict_scores["knn"], pomegranate_exact_sparse_dict_scores["knn_d"]]
    p_g_sparse_means = [pomegranate_greedy_sparse_dict_scores["dt"],
               pomegranate_greedy_sparse_dict_scores["dt_e"],
               pomegranate_greedy_sparse_dict_scores["rf"],
               pomegranate_greedy_sparse_dict_scores["rf_e"],
               pomegranate_greedy_sparse_dict_scores["lr"],
               pomegranate_greedy_sparse_dict_scores["lr_l1"],
               pomegranate_greedy_sparse_dict_scores["lr_l2"],
               pomegranate_greedy_sparse_dict_scores["lr_e"],
               pomegranate_greedy_sparse_dict_scores["nb"],
               pomegranate_greedy_sparse_dict_scores["nb_g"],
               pomegranate_greedy_sparse_dict_scores["nb_m"],
               pomegranate_greedy_sparse_dict_scores["nb_c"],
               pomegranate_greedy_sparse_dict_scores["svm"],
               pomegranate_greedy_sparse_dict_scores["svm_po"],
               pomegranate_greedy_sparse_dict_scores["svm_r"],
               pomegranate_greedy_sparse_dict_scores["knn"],
               pomegranate_greedy_sparse_dict_scores["knn_d"]]

    pgmpy_tree_sparse_means = [pgmpy_tree_sparse_dict_scores["dt"],
                      pgmpy_tree_sparse_dict_scores["dt_e"],
                      pgmpy_tree_sparse_dict_scores["rf"],
                      pgmpy_tree_sparse_dict_scores["rf_e"],
                      pgmpy_tree_sparse_dict_scores["lr"],
                      pgmpy_tree_sparse_dict_scores["lr_l1"],
                      pgmpy_tree_sparse_dict_scores["lr_l2"],
                      pgmpy_tree_sparse_dict_scores["lr_e"],
                      pgmpy_tree_sparse_dict_scores["nb"],
                      pgmpy_tree_sparse_dict_scores["nb_g"],
                      pgmpy_tree_sparse_dict_scores["nb_m"],
                      pgmpy_tree_sparse_dict_scores["nb_c"],
                      pgmpy_tree_sparse_dict_scores["svm"],
                      pgmpy_tree_sparse_dict_scores["svm_po"],
                      pgmpy_tree_sparse_dict_scores["svm_r"],
                      pgmpy_tree_sparse_dict_scores["knn"],
                      pgmpy_tree_sparse_dict_scores["knn_d"]]
    pgmpy_hc_sparse_means = [pgmpy_hc_sparse_dict_scores["dt"],
                        pgmpy_hc_sparse_dict_scores["dt_e"],
                        pgmpy_hc_sparse_dict_scores["rf"],
                        pgmpy_hc_sparse_dict_scores["rf_e"],
                        pgmpy_hc_sparse_dict_scores["lr"],
                        pgmpy_hc_sparse_dict_scores["lr_l1"],
                        pgmpy_hc_sparse_dict_scores["lr_l2"],
                        pgmpy_hc_sparse_dict_scores["lr_e"],
                        pgmpy_hc_sparse_dict_scores["nb"],
                        pgmpy_hc_sparse_dict_scores["nb_g"],
                        pgmpy_hc_sparse_dict_scores["nb_m"],
                        pgmpy_hc_sparse_dict_scores["nb_c"],
                        pgmpy_hc_sparse_dict_scores["svm"],
                        pgmpy_hc_sparse_dict_scores["svm_po"],
                        pgmpy_hc_sparse_dict_scores["svm_r"],
                        pgmpy_hc_sparse_dict_scores["knn"],
                        pgmpy_hc_sparse_dict_scores["knn_d"]]
    pgmpy_mmhc_sparse_means = [pgmpy_mmhc_sparse_dict_scores["dt"],
                      pgmpy_mmhc_sparse_dict_scores["dt_e"],
                      pgmpy_mmhc_sparse_dict_scores["rf"],
                      pgmpy_mmhc_sparse_dict_scores["rf_e"],
                      pgmpy_mmhc_sparse_dict_scores["lr"],
                      pgmpy_mmhc_sparse_dict_scores["lr_l1"],
                      pgmpy_mmhc_sparse_dict_scores["lr_l2"],
                      pgmpy_mmhc_sparse_dict_scores["lr_e"],
                      pgmpy_mmhc_sparse_dict_scores["nb"],
                      pgmpy_mmhc_sparse_dict_scores["nb_g"],
                      pgmpy_mmhc_sparse_dict_scores["nb_m"],
                      pgmpy_mmhc_sparse_dict_scores["nb_c"],
                      pgmpy_mmhc_sparse_dict_scores["svm"],
                      pgmpy_mmhc_sparse_dict_scores["svm_po"],
                      pgmpy_mmhc_sparse_dict_scores["svm_r"],
                      pgmpy_mmhc_sparse_dict_scores["knn"],
                      pgmpy_mmhc_sparse_dict_scores["knn_d"]]

    plt.rcParams["figure.figsize"] = [18, 18]
    plt.rcParams["figure.autolayout"] = True

    x_axis = np.arange(len(labels))
    w = 0.05  # the width of the bars

    plt.bar(x_axis +w, bn_sparse_means, width=0.05, label = "BN_LEARN (HC)", color="lightsteelblue")
    plt.bar(x_axis + w * 2, nt_sparse_means, width=0.05, label="BN_LEARN (TABU)", color="cornflowerblue")
    plt.bar(x_axis + w * 3, bn_mmhc_sparse_means, width=0.05, label="BN_LEARN (MMHC)", color="blue")
    plt.bar(x_axis + w * 4, bn_rsmax2_sparse_means, width=0.05, label="BN_LEARN (RSMAX2)", color="mediumblue")
    plt.bar(x_axis + w * 5, bn_h2pc_sparse_means, width=0.05, label="BN_LEARN (H2PC)", color="navy")
    plt.bar(x_axis +w*6, nt_sparse_means, width=0.05, label="NO_TEARS (logistic)", color="limegreen")
    plt.bar(x_axis +w*7, nt_l2_sparse_means, width=0.05, label="NO_TEARS (l2)", color="forestgreen")
    plt.bar(x_axis + w * 8, nt_p_sparse_means, width=0.05, label="NO_TEARS (poisson)", color="darkgreen")
    plt.bar(x_axis + w * 9, p_sparse_means, width=0.05, label="POMEGRANATE (exact)", color="darkviolet")
    plt.bar(x_axis + w * 10, p_g_sparse_means, width=0.05, label="POMEGRANATE (greed)", color="rebeccapurple")
    plt.bar(x_axis + w * 11, pgmpy_mmhc_sparse_means, width=0.05, label="PGMPY (MMHC)", color="#FA8072")
    plt.bar(x_axis + w * 12, pgmpy_hc_sparse_means, width=0.05, label="PGMPY (HC)", color="#FF2400")
    plt.bar(x_axis + w * 13, pgmpy_tree_sparse_means, width=0.05, label="PGMPY (TREE)", color="#7C0A02")

    plt.xticks(x_axis, labels)
    plt.legend()
    plt.style.use("fivethirtyeight")
    plt.ylabel('Accuracy')
    plt.xlabel('ML Technique', labelpad=15)
    plt.title('Sparse Problem - Performance by library on ML technique')
    #plt.ylim(0.6, 1)
    #plt.tick_params(rotation=45)
    plt.savefig('pipeline_summary_benchmark_for_sparse_by_library_groupbar.png', bbox_inches='tight')
    plt.show()

    # Produce Dimensional Problem by Library on Problem
    # Group by figure
    labels = ['DT_G', 'DT_E', 'RF_G', 'RF_E', 'LR', 'LR_L1', 'LR_L2', 'LR_E', 'NB_B', 'NB_G', 'NB_M', 'NB_C', 'SVM_S',
              'SVM_P', 'SVM_R', 'KNN_W', 'KNN_D']
    bn_dimension_means = [bnlearn_dimension_dict_scores["dt"], bnlearn_dimension_dict_scores["dt_e"], bnlearn_dimension_dict_scores["rf"], bnlearn_dimension_dict_scores["rf_e"], bnlearn_dimension_dict_scores["lr"], bnlearn_dimension_dict_scores["lr_l1"], bnlearn_dimension_dict_scores["lr_l2"], bnlearn_dimension_dict_scores["lr_e"], bnlearn_dimension_dict_scores["nb"], bnlearn_dimension_dict_scores["nb_g"], bnlearn_dimension_dict_scores["nb_m"], bnlearn_dimension_dict_scores["nb_c"], bnlearn_dimension_dict_scores["svm"], bnlearn_dimension_dict_scores["svm_po"], bnlearn_dimension_dict_scores["svm_r"], bnlearn_dimension_dict_scores["knn"], bnlearn_dimension_dict_scores["knn_d"]]
    bn_tabu_dimension_means = [bnlearn_tabu_dimension_dict_scores["dt"], bnlearn_tabu_dimension_dict_scores["dt_e"],
                bnlearn_tabu_dimension_dict_scores["rf"], bnlearn_tabu_dimension_dict_scores["rf_e"],
                bnlearn_tabu_dimension_dict_scores["lr"], bnlearn_tabu_dimension_dict_scores["lr_l1"],
                bnlearn_tabu_dimension_dict_scores["lr_l2"], bnlearn_tabu_dimension_dict_scores["lr_e"],
                bnlearn_tabu_dimension_dict_scores["nb"], bnlearn_tabu_dimension_dict_scores["nb_g"],
                bnlearn_tabu_dimension_dict_scores["nb_m"], bnlearn_tabu_dimension_dict_scores["nb_c"],
                bnlearn_tabu_dimension_dict_scores["svm"], bnlearn_tabu_dimension_dict_scores["svm_po"],
                bnlearn_tabu_dimension_dict_scores["svm_r"], bnlearn_tabu_dimension_dict_scores["knn"],
                bnlearn_tabu_dimension_dict_scores["knn_d"]]
    bn_mmhc_dimension_means = [bnlearn_mmhc_dimension_dict_scores["dt"], bnlearn_mmhc_dimension_dict_scores["dt_e"],
                bnlearn_mmhc_dimension_dict_scores["rf"], bnlearn_mmhc_dimension_dict_scores["rf_e"],
                bnlearn_mmhc_dimension_dict_scores["lr"], bnlearn_mmhc_dimension_dict_scores["lr_l1"],
                bnlearn_mmhc_dimension_dict_scores["lr_l2"], bnlearn_mmhc_dimension_dict_scores["lr_e"],
                bnlearn_mmhc_dimension_dict_scores["nb"], bnlearn_mmhc_dimension_dict_scores["nb_g"],
                bnlearn_mmhc_dimension_dict_scores["nb_m"], bnlearn_mmhc_dimension_dict_scores["nb_c"],
                bnlearn_mmhc_dimension_dict_scores["svm"], bnlearn_mmhc_dimension_dict_scores["svm_po"],
                bnlearn_mmhc_dimension_dict_scores["svm_r"], bnlearn_mmhc_dimension_dict_scores["knn"],
                bnlearn_mmhc_dimension_dict_scores["knn_d"]]
    bn_rsmax2_dimension_means = [bnlearn_rsmax2_dimension_dict_scores["dt"], bnlearn_rsmax2_dimension_dict_scores["dt_e"],
                bnlearn_rsmax2_dimension_dict_scores["rf"], bnlearn_rsmax2_dimension_dict_scores["rf_e"],
                bnlearn_rsmax2_dimension_dict_scores["lr"], bnlearn_rsmax2_dimension_dict_scores["lr_l1"],
                bnlearn_rsmax2_dimension_dict_scores["lr_l2"], bnlearn_rsmax2_dimension_dict_scores["lr_e"],
                bnlearn_rsmax2_dimension_dict_scores["nb"], bnlearn_rsmax2_dimension_dict_scores["nb_g"],
                bnlearn_rsmax2_dimension_dict_scores["nb_m"], bnlearn_rsmax2_dimension_dict_scores["nb_c"],
                bnlearn_rsmax2_dimension_dict_scores["svm"], bnlearn_rsmax2_dimension_dict_scores["svm_po"],
                bnlearn_rsmax2_dimension_dict_scores["svm_r"], bnlearn_rsmax2_dimension_dict_scores["knn"],
                bnlearn_rsmax2_dimension_dict_scores["knn_d"]]
    bn_h2pc_dimension_means = [bnlearn_h2pc_dimension_dict_scores["dt"], bnlearn_h2pc_dimension_dict_scores["dt_e"],
                bnlearn_h2pc_dimension_dict_scores["rf"], bnlearn_h2pc_dimension_dict_scores["rf_e"],
                bnlearn_h2pc_dimension_dict_scores["lr"], bnlearn_h2pc_dimension_dict_scores["lr_l1"],
                bnlearn_h2pc_dimension_dict_scores["lr_l2"], bnlearn_h2pc_dimension_dict_scores["lr_e"],
                bnlearn_h2pc_dimension_dict_scores["nb"], bnlearn_h2pc_dimension_dict_scores["nb_g"],
                bnlearn_h2pc_dimension_dict_scores["nb_m"], bnlearn_h2pc_dimension_dict_scores["nb_c"],
                bnlearn_h2pc_dimension_dict_scores["svm"], bnlearn_h2pc_dimension_dict_scores["svm_po"],
                bnlearn_h2pc_dimension_dict_scores["svm_r"], bnlearn_h2pc_dimension_dict_scores["knn"],
                bnlearn_h2pc_dimension_dict_scores["knn_d"]]

    nt_dimension_means = [notears_dimension_dict_scores["dt"], notears_dimension_dict_scores["dt_e"], notears_dimension_dict_scores["rf"], notears_dimension_dict_scores["rf_e"], notears_dimension_dict_scores["lr"], notears_dimension_dict_scores["lr_l1"], notears_dimension_dict_scores["lr_l2"], notears_dimension_dict_scores["lr_e"], notears_dimension_dict_scores["nb"], notears_dimension_dict_scores["nb_g"], notears_dimension_dict_scores["nb_m"], notears_dimension_dict_scores["nb_c"], notears_dimension_dict_scores["svm"], notears_dimension_dict_scores["svm_po"], notears_dimension_dict_scores["svm_r"], notears_dimension_dict_scores["knn"], notears_dimension_dict_scores["knn_d"]]
    nt_l2_dimension_means = [notears_l2_dimension_dict_scores["dt"], notears_l2_dimension_dict_scores["dt_e"],
                notears_l2_dimension_dict_scores["rf"], notears_l2_dimension_dict_scores["rf_e"],
                notears_l2_dimension_dict_scores["lr"], notears_l2_dimension_dict_scores["lr_l1"],
                notears_l2_dimension_dict_scores["lr_l2"], notears_l2_dimension_dict_scores["lr_e"],
                notears_l2_dimension_dict_scores["nb"], notears_l2_dimension_dict_scores["nb_g"],
                notears_l2_dimension_dict_scores["nb_m"], notears_l2_dimension_dict_scores["nb_c"],
                notears_l2_dimension_dict_scores["svm"], notears_l2_dimension_dict_scores["svm_po"],
                notears_l2_dimension_dict_scores["svm_r"], notears_l2_dimension_dict_scores["knn"],
                notears_l2_dimension_dict_scores["knn_d"]]
    nt_p_dimension_means = [notears_poisson_dimension_dict_scores["dt"], notears_poisson_dimension_dict_scores["dt_e"],
                notears_poisson_dimension_dict_scores["rf"], notears_poisson_dimension_dict_scores["rf_e"],
                notears_poisson_dimension_dict_scores["lr"], notears_poisson_dimension_dict_scores["lr_l1"],
                notears_poisson_dimension_dict_scores["lr_l2"], notears_poisson_dimension_dict_scores["lr_e"],
                notears_poisson_dimension_dict_scores["nb"], notears_poisson_dimension_dict_scores["nb_g"],
                notears_poisson_dimension_dict_scores["nb_m"], notears_poisson_dimension_dict_scores["nb_c"],
                notears_poisson_dimension_dict_scores["svm"], notears_poisson_dimension_dict_scores["svm_po"],
                notears_poisson_dimension_dict_scores["svm_r"], notears_poisson_dimension_dict_scores["knn"],
                notears_poisson_dimension_dict_scores["knn_d"]]

    p_dimension_means = [pomegranate_exact_dimension_dict_scores["dt"], pomegranate_exact_dimension_dict_scores["dt_e"], pomegranate_exact_dimension_dict_scores["rf"], pomegranate_exact_dimension_dict_scores["rf_e"], pomegranate_exact_dimension_dict_scores["lr"], pomegranate_exact_dimension_dict_scores["lr_l1"], pomegranate_exact_dimension_dict_scores["lr_l2"], pomegranate_exact_dimension_dict_scores["lr_e"], pomegranate_exact_dimension_dict_scores["nb"], pomegranate_exact_dimension_dict_scores["nb_g"], pomegranate_exact_dimension_dict_scores["nb_m"], pomegranate_exact_dimension_dict_scores["nb_c"], pomegranate_exact_dimension_dict_scores["svm"], pomegranate_exact_dimension_dict_scores["svm_po"], pomegranate_exact_dimension_dict_scores["svm_r"], pomegranate_exact_dimension_dict_scores["knn"], pomegranate_exact_dimension_dict_scores["knn_d"]]
    p_g_dimension_means = [pomegranate_greedy_dimension_dict_scores["dt"],
               pomegranate_greedy_dimension_dict_scores["dt_e"],
               pomegranate_greedy_dimension_dict_scores["rf"],
               pomegranate_greedy_dimension_dict_scores["rf_e"],
               pomegranate_greedy_dimension_dict_scores["lr"],
               pomegranate_greedy_dimension_dict_scores["lr_l1"],
               pomegranate_greedy_dimension_dict_scores["lr_l2"],
               pomegranate_greedy_dimension_dict_scores["lr_e"],
               pomegranate_greedy_dimension_dict_scores["nb"],
               pomegranate_greedy_dimension_dict_scores["nb_g"],
               pomegranate_greedy_dimension_dict_scores["nb_m"],
               pomegranate_greedy_dimension_dict_scores["nb_c"],
               pomegranate_greedy_dimension_dict_scores["svm"],
               pomegranate_greedy_dimension_dict_scores["svm_po"],
               pomegranate_greedy_dimension_dict_scores["svm_r"],
               pomegranate_greedy_dimension_dict_scores["knn"],
               pomegranate_greedy_dimension_dict_scores["knn_d"]]

    pgmpy_tree_dimension_means = [pgmpy_tree_dimension_dict_scores["dt"],
                      pgmpy_tree_dimension_dict_scores["dt_e"],
                      pgmpy_tree_dimension_dict_scores["rf"],
                      pgmpy_tree_dimension_dict_scores["rf_e"],
                      pgmpy_tree_dimension_dict_scores["lr"],
                      pgmpy_tree_dimension_dict_scores["lr_l1"],
                      pgmpy_tree_dimension_dict_scores["lr_l2"],
                      pgmpy_tree_dimension_dict_scores["lr_e"],
                      pgmpy_tree_dimension_dict_scores["nb"],
                      pgmpy_tree_dimension_dict_scores["nb_g"],
                      pgmpy_tree_dimension_dict_scores["nb_m"],
                      pgmpy_tree_dimension_dict_scores["nb_c"],
                      pgmpy_tree_dimension_dict_scores["svm"],
                      pgmpy_tree_dimension_dict_scores["svm_po"],
                      pgmpy_tree_dimension_dict_scores["svm_r"],
                      pgmpy_tree_dimension_dict_scores["knn"],
                      pgmpy_tree_dimension_dict_scores["knn_d"]]
    pgmpy_hc_dimension_means = [pgmpy_hc_dimension_dict_scores["dt"],
                        pgmpy_hc_dimension_dict_scores["dt_e"],
                        pgmpy_hc_dimension_dict_scores["rf"],
                        pgmpy_hc_dimension_dict_scores["rf_e"],
                        pgmpy_hc_dimension_dict_scores["lr"],
                        pgmpy_hc_dimension_dict_scores["lr_l1"],
                        pgmpy_hc_dimension_dict_scores["lr_l2"],
                        pgmpy_hc_dimension_dict_scores["lr_e"],
                        pgmpy_hc_dimension_dict_scores["nb"],
                        pgmpy_hc_dimension_dict_scores["nb_g"],
                        pgmpy_hc_dimension_dict_scores["nb_m"],
                        pgmpy_hc_dimension_dict_scores["nb_c"],
                        pgmpy_hc_dimension_dict_scores["svm"],
                        pgmpy_hc_dimension_dict_scores["svm_po"],
                        pgmpy_hc_dimension_dict_scores["svm_r"],
                        pgmpy_hc_dimension_dict_scores["knn"],
                        pgmpy_hc_dimension_dict_scores["knn_d"]]
    pgmpy_mmhc_dimension_means = [pgmpy_mmhc_dimension_dict_scores["dt"],
                      pgmpy_mmhc_dimension_dict_scores["dt_e"],
                      pgmpy_mmhc_dimension_dict_scores["rf"],
                      pgmpy_mmhc_dimension_dict_scores["rf_e"],
                      pgmpy_mmhc_dimension_dict_scores["lr"],
                      pgmpy_mmhc_dimension_dict_scores["lr_l1"],
                      pgmpy_mmhc_dimension_dict_scores["lr_l2"],
                      pgmpy_mmhc_dimension_dict_scores["lr_e"],
                      pgmpy_mmhc_dimension_dict_scores["nb"],
                      pgmpy_mmhc_dimension_dict_scores["nb_g"],
                      pgmpy_mmhc_dimension_dict_scores["nb_m"],
                      pgmpy_mmhc_dimension_dict_scores["nb_c"],
                      pgmpy_mmhc_dimension_dict_scores["svm"],
                      pgmpy_mmhc_dimension_dict_scores["svm_po"],
                      pgmpy_mmhc_dimension_dict_scores["svm_r"],
                      pgmpy_mmhc_dimension_dict_scores["knn"],
                      pgmpy_mmhc_dimension_dict_scores["knn_d"]]

    plt.rcParams["figure.figsize"] = [18, 18]
    plt.rcParams["figure.autolayout"] = True

    x_axis = np.arange(len(labels))
    w = 0.05  # the width of the bars

    plt.bar(x_axis +w, bn_dimension_means, width=0.05, label = "BN_LEARN (HC)", color="lightsteelblue")
    plt.bar(x_axis + w * 2, nt_dimension_means, width=0.05, label="BN_LEARN (TABU)", color="cornflowerblue")
    plt.bar(x_axis + w * 3, bn_mmhc_dimension_means, width=0.05, label="BN_LEARN (MMHC)", color="blue")
    plt.bar(x_axis + w * 4, bn_rsmax2_dimension_means, width=0.05, label="BN_LEARN (RSMAX2)", color="mediumblue")
    plt.bar(x_axis + w * 5, bn_h2pc_dimension_means, width=0.05, label="BN_LEARN (H2PC)", color="navy")
    plt.bar(x_axis +w*6, nt_dimension_means, width=0.05, label="NO_TEARS (logistic)", color="limegreen")
    plt.bar(x_axis +w*7, nt_l2_dimension_means, width=0.05, label="NO_TEARS (l2)", color="forestgreen")
    plt.bar(x_axis + w * 8, nt_p_dimension_means, width=0.05, label="NO_TEARS (poisson)", color="darkgreen")
    plt.bar(x_axis + w * 9, p_dimension_means, width=0.05, label="POMEGRANATE (exact)", color="darkviolet")
    plt.bar(x_axis + w * 10, p_g_dimension_means, width=0.05, label="POMEGRANATE (greed)", color="rebeccapurple")
    plt.bar(x_axis + w * 11, pgmpy_mmhc_dimension_means, width=0.05, label="PGMPY (MMHC)", color="#FA8072")
    plt.bar(x_axis + w * 12, pgmpy_hc_dimension_means, width=0.05, label="PGMPY (HC)", color="#FF2400")
    plt.bar(x_axis + w * 13, pgmpy_tree_dimension_means, width=0.05, label="PGMPY (TREE)", color="#7C0A02")

    plt.xticks(x_axis, labels)
    plt.legend()
    plt.style.use("fivethirtyeight")
    plt.ylabel('Accuracy')
    plt.xlabel('ML Technique', labelpad=15)
    plt.title('Dimension Problem - Performance by library on ML technique')
    #plt.ylim(0.6, 1)
    #plt.tick_params(rotation=45)
    plt.savefig('pipeline_summary_benchmark_for_dimension_by_library_groupbar.png', bbox_inches='tight')
    plt.show()
    
    
    #--------------

    # Produce Linear Problem by Library on Problem (test set from learned world)
    # Group by figure
    labels = ['DT_G', 'DT_E', 'RF_G', 'RF_E', 'LR', 'LR_L1', 'LR_L2', 'LR_E', 'NB_B', 'NB_G', 'NB_M', 'NB_C', 'SVM_S',
              'SVM_P', 'SVM_R', 'KNN_W', 'KNN_D']
    bn_means = [bnlearn_linear_dict_scores_simtest["dt"], bnlearn_linear_dict_scores_simtest["dt_e"], bnlearn_linear_dict_scores_simtest["rf"],
                bnlearn_linear_dict_scores_simtest["rf_e"], bnlearn_linear_dict_scores_simtest["lr"],
                bnlearn_linear_dict_scores_simtest["lr_l1"], bnlearn_linear_dict_scores_simtest["lr_l2"],
                bnlearn_linear_dict_scores_simtest["lr_e"], bnlearn_linear_dict_scores_simtest["nb"],
                bnlearn_linear_dict_scores_simtest["nb_g"], bnlearn_linear_dict_scores_simtest["nb_m"],
                bnlearn_linear_dict_scores_simtest["nb_c"], bnlearn_linear_dict_scores_simtest["svm"],
                bnlearn_linear_dict_scores_simtest["svm_po"], bnlearn_linear_dict_scores_simtest["svm_r"],
                bnlearn_linear_dict_scores_simtest["knn"], bnlearn_linear_dict_scores_simtest["knn_d"]]
    bn_tabu_means = [bnlearn_tabu_linear_dict_scores_simtest["dt"], bnlearn_tabu_linear_dict_scores_simtest["dt_e"],
                     bnlearn_tabu_linear_dict_scores_simtest["rf"], bnlearn_tabu_linear_dict_scores_simtest["rf_e"],
                     bnlearn_tabu_linear_dict_scores_simtest["lr"], bnlearn_tabu_linear_dict_scores_simtest["lr_l1"],
                     bnlearn_tabu_linear_dict_scores_simtest["lr_l2"], bnlearn_tabu_linear_dict_scores_simtest["lr_e"],
                     bnlearn_tabu_linear_dict_scores_simtest["nb"], bnlearn_tabu_linear_dict_scores_simtest["nb_g"],
                     bnlearn_tabu_linear_dict_scores_simtest["nb_m"], bnlearn_tabu_linear_dict_scores_simtest["nb_c"],
                     bnlearn_tabu_linear_dict_scores_simtest["svm"], bnlearn_tabu_linear_dict_scores_simtest["svm_po"],
                     bnlearn_tabu_linear_dict_scores_simtest["svm_r"], bnlearn_tabu_linear_dict_scores_simtest["knn"],
                     bnlearn_tabu_linear_dict_scores_simtest["knn_d"]]
    bn_pc_means = [bnlearn_pc_linear_dict_scores_simtest["dt"], bnlearn_pc_linear_dict_scores_simtest["dt_e"],
                   bnlearn_pc_linear_dict_scores_simtest["rf"], bnlearn_pc_linear_dict_scores_simtest["rf_e"],
                   bnlearn_pc_linear_dict_scores_simtest["lr"], bnlearn_pc_linear_dict_scores_simtest["lr_l1"],
                   bnlearn_pc_linear_dict_scores_simtest["lr_l2"], bnlearn_pc_linear_dict_scores_simtest["lr_e"],
                   bnlearn_pc_linear_dict_scores_simtest["nb"], bnlearn_pc_linear_dict_scores_simtest["nb_g"],
                   bnlearn_pc_linear_dict_scores_simtest["nb_m"], bnlearn_pc_linear_dict_scores_simtest["nb_c"],
                   bnlearn_pc_linear_dict_scores_simtest["svm"], bnlearn_pc_linear_dict_scores_simtest["svm_po"],
                   bnlearn_pc_linear_dict_scores_simtest["svm_r"], bnlearn_pc_linear_dict_scores_simtest["knn"],
                   bnlearn_pc_linear_dict_scores_simtest["knn_d"]]
    bn_mmhc_means = [bnlearn_mmhc_linear_dict_scores_simtest["dt"], bnlearn_mmhc_linear_dict_scores_simtest["dt_e"],
                     bnlearn_mmhc_linear_dict_scores_simtest["rf"], bnlearn_mmhc_linear_dict_scores_simtest["rf_e"],
                     bnlearn_mmhc_linear_dict_scores_simtest["lr"], bnlearn_mmhc_linear_dict_scores_simtest["lr_l1"],
                     bnlearn_mmhc_linear_dict_scores_simtest["lr_l2"], bnlearn_mmhc_linear_dict_scores_simtest["lr_e"],
                     bnlearn_mmhc_linear_dict_scores_simtest["nb"], bnlearn_mmhc_linear_dict_scores_simtest["nb_g"],
                     bnlearn_mmhc_linear_dict_scores_simtest["nb_m"], bnlearn_mmhc_linear_dict_scores_simtest["nb_c"],
                     bnlearn_mmhc_linear_dict_scores_simtest["svm"], bnlearn_mmhc_linear_dict_scores_simtest["svm_po"],
                     bnlearn_mmhc_linear_dict_scores_simtest["svm_r"], bnlearn_mmhc_linear_dict_scores_simtest["knn"],
                     bnlearn_mmhc_linear_dict_scores_simtest["knn_d"]]
    bn_rsmax2_means = [bnlearn_rsmax2_linear_dict_scores_simtest["dt"], bnlearn_rsmax2_linear_dict_scores_simtest["dt_e"],
                       bnlearn_rsmax2_linear_dict_scores_simtest["rf"], bnlearn_rsmax2_linear_dict_scores_simtest["rf_e"],
                       bnlearn_rsmax2_linear_dict_scores_simtest["lr"], bnlearn_rsmax2_linear_dict_scores_simtest["lr_l1"],
                       bnlearn_rsmax2_linear_dict_scores_simtest["lr_l2"], bnlearn_rsmax2_linear_dict_scores_simtest["lr_e"],
                       bnlearn_rsmax2_linear_dict_scores_simtest["nb"], bnlearn_rsmax2_linear_dict_scores_simtest["nb_g"],
                       bnlearn_rsmax2_linear_dict_scores_simtest["nb_m"], bnlearn_rsmax2_linear_dict_scores_simtest["nb_c"],
                       bnlearn_rsmax2_linear_dict_scores_simtest["svm"], bnlearn_rsmax2_linear_dict_scores_simtest["svm_po"],
                       bnlearn_rsmax2_linear_dict_scores_simtest["svm_r"], bnlearn_rsmax2_linear_dict_scores_simtest["knn"],
                       bnlearn_rsmax2_linear_dict_scores_simtest["knn_d"]]
    bn_h2pc_means = [bnlearn_h2pc_linear_dict_scores_simtest["dt"], bnlearn_h2pc_linear_dict_scores_simtest["dt_e"],
                     bnlearn_h2pc_linear_dict_scores_simtest["rf"], bnlearn_h2pc_linear_dict_scores_simtest["rf_e"],
                     bnlearn_h2pc_linear_dict_scores_simtest["lr"], bnlearn_h2pc_linear_dict_scores_simtest["lr_l1"],
                     bnlearn_h2pc_linear_dict_scores_simtest["lr_l2"], bnlearn_h2pc_linear_dict_scores_simtest["lr_e"],
                     bnlearn_h2pc_linear_dict_scores_simtest["nb"], bnlearn_h2pc_linear_dict_scores_simtest["nb_g"],
                     bnlearn_h2pc_linear_dict_scores_simtest["nb_m"], bnlearn_h2pc_linear_dict_scores_simtest["nb_c"],
                     bnlearn_h2pc_linear_dict_scores_simtest["svm"], bnlearn_h2pc_linear_dict_scores_simtest["svm_po"],
                     bnlearn_h2pc_linear_dict_scores_simtest["svm_r"], bnlearn_h2pc_linear_dict_scores_simtest["knn"],
                     bnlearn_h2pc_linear_dict_scores_simtest["knn_d"]]

    nt_means = [notears_linear_dict_scores_simtest["dt"], notears_linear_dict_scores_simtest["dt_e"], notears_linear_dict_scores_simtest["rf"],
                notears_linear_dict_scores_simtest["rf_e"], notears_linear_dict_scores_simtest["lr"],
                notears_linear_dict_scores_simtest["lr_l1"], notears_linear_dict_scores_simtest["lr_l2"],
                notears_linear_dict_scores_simtest["lr_e"], notears_linear_dict_scores_simtest["nb"],
                notears_linear_dict_scores_simtest["nb_g"], notears_linear_dict_scores_simtest["nb_m"],
                notears_linear_dict_scores_simtest["nb_c"], notears_linear_dict_scores_simtest["svm"],
                notears_linear_dict_scores_simtest["svm_po"], notears_linear_dict_scores_simtest["svm_r"],
                notears_linear_dict_scores_simtest["knn"], notears_linear_dict_scores_simtest["knn_d"]]
    nt_l2_means = [notears_l2_linear_dict_scores_simtest["dt"], notears_l2_linear_dict_scores_simtest["dt_e"],
                   notears_l2_linear_dict_scores_simtest["rf"], notears_l2_linear_dict_scores_simtest["rf_e"],
                   notears_l2_linear_dict_scores_simtest["lr"], notears_l2_linear_dict_scores_simtest["lr_l1"],
                   notears_l2_linear_dict_scores_simtest["lr_l2"], notears_l2_linear_dict_scores_simtest["lr_e"],
                   notears_l2_linear_dict_scores_simtest["nb"], notears_l2_linear_dict_scores_simtest["nb_g"],
                   notears_l2_linear_dict_scores_simtest["nb_m"], notears_l2_linear_dict_scores_simtest["nb_c"],
                   notears_l2_linear_dict_scores_simtest["svm"], notears_l2_linear_dict_scores_simtest["svm_po"],
                   notears_l2_linear_dict_scores_simtest["svm_r"], notears_l2_linear_dict_scores_simtest["knn"],
                   notears_l2_linear_dict_scores_simtest["knn_d"]]
    nt_p_means = [notears_poisson_linear_dict_scores_simtest["dt"], notears_poisson_linear_dict_scores_simtest["dt_e"],
                  notears_poisson_linear_dict_scores_simtest["rf"], notears_poisson_linear_dict_scores_simtest["rf_e"],
                  notears_poisson_linear_dict_scores_simtest["lr"], notears_poisson_linear_dict_scores_simtest["lr_l1"],
                  notears_poisson_linear_dict_scores_simtest["lr_l2"], notears_poisson_linear_dict_scores_simtest["lr_e"],
                  notears_poisson_linear_dict_scores_simtest["nb"], notears_poisson_linear_dict_scores_simtest["nb_g"],
                  notears_poisson_linear_dict_scores_simtest["nb_m"], notears_poisson_linear_dict_scores_simtest["nb_c"],
                  notears_poisson_linear_dict_scores_simtest["svm"], notears_poisson_linear_dict_scores_simtest["svm_po"],
                  notears_poisson_linear_dict_scores_simtest["svm_r"], notears_poisson_linear_dict_scores_simtest["knn"],
                  notears_poisson_linear_dict_scores_simtest["knn_d"]]

    p_means = [pomegranate_exact_linear_dict_scores_simtest["dt"], pomegranate_exact_linear_dict_scores_simtest["dt_e"],
               pomegranate_exact_linear_dict_scores_simtest["rf"], pomegranate_exact_linear_dict_scores_simtest["rf_e"],
               pomegranate_exact_linear_dict_scores_simtest["lr"], pomegranate_exact_linear_dict_scores_simtest["lr_l1"],
               pomegranate_exact_linear_dict_scores_simtest["lr_l2"], pomegranate_exact_linear_dict_scores_simtest["lr_e"],
               pomegranate_exact_linear_dict_scores_simtest["nb"], pomegranate_exact_linear_dict_scores_simtest["nb_g"],
               pomegranate_exact_linear_dict_scores_simtest["nb_m"], pomegranate_exact_linear_dict_scores_simtest["nb_c"],
               pomegranate_exact_linear_dict_scores_simtest["svm"], pomegranate_exact_linear_dict_scores_simtest["svm_po"],
               pomegranate_exact_linear_dict_scores_simtest["svm_r"], pomegranate_exact_linear_dict_scores_simtest["knn"],
               pomegranate_exact_linear_dict_scores_simtest["knn_d"]]
    p_g_means = [pomegranate_greedy_linear_dict_scores_simtest["dt"],
                 pomegranate_greedy_linear_dict_scores_simtest["dt_e"],
                 pomegranate_greedy_linear_dict_scores_simtest["rf"],
                 pomegranate_greedy_linear_dict_scores_simtest["rf_e"],
                 pomegranate_greedy_linear_dict_scores_simtest["lr"],
                 pomegranate_greedy_linear_dict_scores_simtest["lr_l1"],
                 pomegranate_greedy_linear_dict_scores_simtest["lr_l2"],
                 pomegranate_greedy_linear_dict_scores_simtest["lr_e"],
                 pomegranate_greedy_linear_dict_scores_simtest["nb"],
                 pomegranate_greedy_linear_dict_scores_simtest["nb_g"],
                 pomegranate_greedy_linear_dict_scores_simtest["nb_m"],
                 pomegranate_greedy_linear_dict_scores_simtest["nb_c"],
                 pomegranate_greedy_linear_dict_scores_simtest["svm"],
                 pomegranate_greedy_linear_dict_scores_simtest["svm_po"],
                 pomegranate_greedy_linear_dict_scores_simtest["svm_r"],
                 pomegranate_greedy_linear_dict_scores_simtest["knn"],
                 pomegranate_greedy_linear_dict_scores_simtest["knn_d"]]

    pgmpy_tree_means = [pgmpy_tree_linear_dict_scores_simtest["dt"],
                        pgmpy_tree_linear_dict_scores_simtest["dt_e"],
                        pgmpy_tree_linear_dict_scores_simtest["rf"],
                        pgmpy_tree_linear_dict_scores_simtest["rf_e"],
                        pgmpy_tree_linear_dict_scores_simtest["lr"],
                        pgmpy_tree_linear_dict_scores_simtest["lr_l1"],
                        pgmpy_tree_linear_dict_scores_simtest["lr_l2"],
                        pgmpy_tree_linear_dict_scores_simtest["lr_e"],
                        pgmpy_tree_linear_dict_scores_simtest["nb"],
                        pgmpy_tree_linear_dict_scores_simtest["nb_g"],
                        pgmpy_tree_linear_dict_scores_simtest["nb_m"],
                        pgmpy_tree_linear_dict_scores_simtest["nb_c"],
                        pgmpy_tree_linear_dict_scores_simtest["svm"],
                        pgmpy_tree_linear_dict_scores_simtest["svm_po"],
                        pgmpy_tree_linear_dict_scores_simtest["svm_r"],
                        pgmpy_tree_linear_dict_scores_simtest["knn"],
                        pgmpy_tree_linear_dict_scores_simtest["knn_d"]]
    pgmpy_hc_means = [pgmpy_hc_linear_dict_scores_simtest["dt"],
                      pgmpy_hc_linear_dict_scores_simtest["dt_e"],
                      pgmpy_hc_linear_dict_scores_simtest["rf"],
                      pgmpy_hc_linear_dict_scores_simtest["rf_e"],
                      pgmpy_hc_linear_dict_scores_simtest["lr"],
                      pgmpy_hc_linear_dict_scores_simtest["lr_l1"],
                      pgmpy_hc_linear_dict_scores_simtest["lr_l2"],
                      pgmpy_hc_linear_dict_scores_simtest["lr_e"],
                      pgmpy_hc_linear_dict_scores_simtest["nb"],
                      pgmpy_hc_linear_dict_scores_simtest["nb_g"],
                      pgmpy_hc_linear_dict_scores_simtest["nb_m"],
                      pgmpy_hc_linear_dict_scores_simtest["nb_c"],
                      pgmpy_hc_linear_dict_scores_simtest["svm"],
                      pgmpy_hc_linear_dict_scores_simtest["svm_po"],
                      pgmpy_hc_linear_dict_scores_simtest["svm_r"],
                      pgmpy_hc_linear_dict_scores_simtest["knn"],
                      pgmpy_hc_linear_dict_scores_simtest["knn_d"]]
    pgmpy_mmhc_means = [pgmpy_mmhc_linear_dict_scores_simtest["dt"],
                        pgmpy_mmhc_linear_dict_scores_simtest["dt_e"],
                        pgmpy_mmhc_linear_dict_scores_simtest["rf"],
                        pgmpy_mmhc_linear_dict_scores_simtest["rf_e"],
                        pgmpy_mmhc_linear_dict_scores_simtest["lr"],
                        pgmpy_mmhc_linear_dict_scores_simtest["lr_l1"],
                        pgmpy_mmhc_linear_dict_scores_simtest["lr_l2"],
                        pgmpy_mmhc_linear_dict_scores_simtest["lr_e"],
                        pgmpy_mmhc_linear_dict_scores_simtest["nb"],
                        pgmpy_mmhc_linear_dict_scores_simtest["nb_g"],
                        pgmpy_mmhc_linear_dict_scores_simtest["nb_m"],
                        pgmpy_mmhc_linear_dict_scores_simtest["nb_c"],
                        pgmpy_mmhc_linear_dict_scores_simtest["svm"],
                        pgmpy_mmhc_linear_dict_scores_simtest["svm_po"],
                        pgmpy_mmhc_linear_dict_scores_simtest["svm_r"],
                        pgmpy_mmhc_linear_dict_scores_simtest["knn"],
                        pgmpy_mmhc_linear_dict_scores_simtest["knn_d"]]

    plt.rcParams["figure.figsize"] = [18, 18]
    plt.rcParams["figure.autolayout"] = True

    x_axis = np.arange(len(labels))
    w = 0.05  # the width of the bars

    plt.bar(x_axis + w, bn_means, width=0.05, label="BN_LEARN (HC)", color="lightsteelblue")
    plt.bar(x_axis + w * 2, nt_means, width=0.05, label="BN_LEARN (TABU)", color="cornflowerblue")
    plt.bar(x_axis + w * 3, bn_pc_means, width=0.05, label="BN_LEARN (PC)", color="royalblue")
    plt.bar(x_axis + w * 4, bn_mmhc_means, width=0.05, label="BN_LEARN (MMHC)", color="blue")
    plt.bar(x_axis + w * 5, bn_rsmax2_means, width=0.05, label="BN_LEARN (RSMAX2)", color="mediumblue")
    plt.bar(x_axis + w * 6, bn_h2pc_means, width=0.05, label="BN_LEARN (H2PC)", color="navy")
    plt.bar(x_axis + w * 7, nt_means, width=0.05, label="NO_TEARS (logistic)", color="limegreen")
    plt.bar(x_axis + w * 8, nt_l2_means, width=0.05, label="NO_TEARS (l2)", color="forestgreen")
    plt.bar(x_axis + w * 9, nt_p_means, width=0.05, label="NO_TEARS (poisson)", color="darkgreen")
    plt.bar(x_axis + w * 10, p_means, width=0.05, label="POMEGRANATE (exact)", color="darkviolet")
    plt.bar(x_axis + w * 11, p_g_means, width=0.05, label="POMEGRANATE (greed)", color="rebeccapurple")
    plt.bar(x_axis + w * 12, pgmpy_mmhc_means, width=0.05, label="PGMPY (MMHC)", color="#FA8072")
    plt.bar(x_axis + w * 13, pgmpy_hc_means, width=0.05, label="PGMPY (HC)", color="#FF2400")
    plt.bar(x_axis + w * 14, pgmpy_tree_means, width=0.05, label="PGMPY (TREE)", color="#7C0A02")

    plt.xticks(x_axis, labels)
    plt.legend()
    plt.style.use("fivethirtyeight")
    plt.ylabel('Accuracy')
    plt.xlabel('ML Technique', labelpad=15)
    plt.title('Linear Problem - Performance by library on ML technique')
    # plt.ylim(0.6, 1)
    # plt.tick_params(rotation=45)
    plt.savefig('pipeline_summary_benchmark_for_linear_by_library_groupbar_simtest.png', bbox_inches='tight')
    plt.show()

    # Produce Non-Linear Problem by Library on Problem
    # Group by figure
    labels = ['DT_G', 'DT_E', 'RF_G', 'RF_E', 'LR', 'LR_L1', 'LR_L2', 'LR_E', 'NB_B', 'NB_G', 'NB_M', 'NB_C', 'SVM_S',
              'SVM_P', 'SVM_R', 'KNN_W', 'KNN_D']
    bn_non_means = [bnlearn_nonlinear_dict_scores_simtest["dt"], bnlearn_nonlinear_dict_scores_simtest["dt_e"],
                    bnlearn_nonlinear_dict_scores_simtest["rf"], bnlearn_nonlinear_dict_scores_simtest["rf_e"],
                    bnlearn_nonlinear_dict_scores_simtest["lr"], bnlearn_nonlinear_dict_scores_simtest["lr_l1"],
                    bnlearn_nonlinear_dict_scores_simtest["lr_l2"], bnlearn_nonlinear_dict_scores_simtest["lr_e"],
                    bnlearn_nonlinear_dict_scores_simtest["nb"], bnlearn_nonlinear_dict_scores_simtest["nb_g"],
                    bnlearn_nonlinear_dict_scores_simtest["nb_m"], bnlearn_nonlinear_dict_scores_simtest["nb_c"],
                    bnlearn_nonlinear_dict_scores_simtest["svm"], bnlearn_nonlinear_dict_scores_simtest["svm_po"],
                    bnlearn_nonlinear_dict_scores_simtest["svm_r"], bnlearn_nonlinear_dict_scores_simtest["knn"],
                    bnlearn_nonlinear_dict_scores_simtest["knn_d"]]
    bn_tabu_non_means = [bnlearn_tabu_nonlinear_dict_scores_simtest["dt"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["dt_e"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["rf"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["rf_e"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["lr"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["lr_l1"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["lr_l2"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["lr_e"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["nb"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["nb_g"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["nb_m"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["nb_c"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["svm"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["svm_po"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["svm_r"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["knn"],
                         bnlearn_tabu_nonlinear_dict_scores_simtest["knn_d"]]
    bn_mmhc_non_means = [bnlearn_mmhc_nonlinear_dict_scores_simtest["dt"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["dt_e"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["rf"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["rf_e"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["lr"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["lr_l1"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["lr_l2"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["lr_e"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["nb"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["nb_g"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["nb_m"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["nb_c"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["svm"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["svm_po"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["svm_r"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["knn"],
                         bnlearn_mmhc_nonlinear_dict_scores_simtest["knn_d"]]
    bn_rsmax2_non_means = [bnlearn_rsmax2_nonlinear_dict_scores_simtest["dt"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["dt_e"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["rf"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["rf_e"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["lr"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["lr_l1"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["lr_l2"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["lr_e"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["nb"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["nb_g"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["nb_m"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["nb_c"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["svm"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["svm_po"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["svm_r"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["knn"],
                           bnlearn_rsmax2_nonlinear_dict_scores_simtest["knn_d"]]
    bn_h2pc_non_means = [bnlearn_h2pc_nonlinear_dict_scores_simtest["dt"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["dt_e"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["rf"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["rf_e"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["lr"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["lr_l1"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["lr_l2"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["lr_e"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["nb"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["nb_g"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["nb_m"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["nb_c"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["svm"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["svm_po"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["svm_r"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["knn"],
                         bnlearn_h2pc_nonlinear_dict_scores_simtest["knn_d"]]

    nt_non_means = [notears_nonlinear_dict_scores_simtest["dt"], notears_nonlinear_dict_scores_simtest["dt_e"],
                    notears_nonlinear_dict_scores_simtest["rf"], notears_nonlinear_dict_scores_simtest["rf_e"],
                    notears_nonlinear_dict_scores_simtest["lr"], notears_nonlinear_dict_scores_simtest["lr_l1"],
                    notears_nonlinear_dict_scores_simtest["lr_l2"], notears_nonlinear_dict_scores_simtest["lr_e"],
                    notears_nonlinear_dict_scores_simtest["nb"], notears_nonlinear_dict_scores_simtest["nb_g"],
                    notears_nonlinear_dict_scores_simtest["nb_m"], notears_nonlinear_dict_scores_simtest["nb_c"],
                    notears_nonlinear_dict_scores_simtest["svm"], notears_nonlinear_dict_scores_simtest["svm_po"],
                    notears_nonlinear_dict_scores_simtest["svm_r"], notears_nonlinear_dict_scores_simtest["knn"],
                    notears_nonlinear_dict_scores_simtest["knn_d"]]
    nt_l2_non_means = [notears_l2_nonlinear_dict_scores_simtest["dt"],
                       notears_l2_nonlinear_dict_scores_simtest["dt_e"],
                       notears_l2_nonlinear_dict_scores_simtest["rf"],
                       notears_l2_nonlinear_dict_scores_simtest["rf_e"],
                       notears_l2_nonlinear_dict_scores_simtest["lr"],
                       notears_l2_nonlinear_dict_scores_simtest["lr_l1"],
                       notears_l2_nonlinear_dict_scores_simtest["lr_l2"],
                       notears_l2_nonlinear_dict_scores_simtest["lr_e"],
                       notears_l2_nonlinear_dict_scores_simtest["nb"],
                       notears_l2_nonlinear_dict_scores_simtest["nb_g"],
                       notears_l2_nonlinear_dict_scores_simtest["nb_m"],
                       notears_l2_nonlinear_dict_scores_simtest["nb_c"],
                       notears_l2_nonlinear_dict_scores_simtest["svm"],
                       notears_l2_nonlinear_dict_scores_simtest["svm_po"],
                       notears_l2_nonlinear_dict_scores_simtest["svm_r"],
                       notears_l2_nonlinear_dict_scores_simtest["knn"],
                       notears_l2_nonlinear_dict_scores_simtest["knn_d"]]
    nt_p_non_means = [notears_poisson_nonlinear_dict_scores_simtest["dt"],
                      notears_poisson_nonlinear_dict_scores_simtest["dt_e"],
                      notears_poisson_nonlinear_dict_scores_simtest["rf"],
                      notears_poisson_nonlinear_dict_scores_simtest["rf_e"],
                      notears_poisson_nonlinear_dict_scores_simtest["lr"],
                      notears_poisson_nonlinear_dict_scores_simtest["lr_l1"],
                      notears_poisson_nonlinear_dict_scores_simtest["lr_l2"],
                      notears_poisson_nonlinear_dict_scores_simtest["lr_e"],
                      notears_poisson_nonlinear_dict_scores_simtest["nb"],
                      notears_poisson_nonlinear_dict_scores_simtest["nb_g"],
                      notears_poisson_nonlinear_dict_scores_simtest["nb_m"],
                      notears_poisson_nonlinear_dict_scores_simtest["nb_c"],
                      notears_poisson_nonlinear_dict_scores_simtest["svm"],
                      notears_poisson_nonlinear_dict_scores_simtest["svm_po"],
                      notears_poisson_nonlinear_dict_scores_simtest["svm_r"],
                      notears_poisson_nonlinear_dict_scores_simtest["knn"],
                      notears_poisson_nonlinear_dict_scores_simtest["knn_d"]]

    p_non_means = [pomegranate_exact_nonlinear_dict_scores_simtest["dt"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["dt_e"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["rf"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["rf_e"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["lr"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["lr_l1"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["lr_l2"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["lr_e"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["nb"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["nb_g"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["nb_m"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["nb_c"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["svm"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["svm_po"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["svm_r"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["knn"],
                   pomegranate_exact_nonlinear_dict_scores_simtest["knn_d"]]
    p_g_non_means = [pomegranate_greedy_nonlinear_dict_scores_simtest["dt"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["dt_e"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["rf"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["rf_e"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["lr"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["lr_l1"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["lr_l2"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["lr_e"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["nb"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["nb_g"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["nb_m"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["nb_c"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["svm"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["svm_po"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["svm_r"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["knn"],
                     pomegranate_greedy_nonlinear_dict_scores_simtest["knn_d"]]

    pgmpy_tree_non_means = [pgmpy_tree_nonlinear_dict_scores_simtest["dt"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["dt_e"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["rf"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["rf_e"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["lr"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["lr_l1"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["lr_l2"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["lr_e"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["nb"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["nb_g"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["nb_m"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["nb_c"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["svm"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["svm_po"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["svm_r"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["knn"],
                            pgmpy_tree_nonlinear_dict_scores_simtest["knn_d"]]
    pgmpy_hc_non_means = [pgmpy_hc_nonlinear_dict_scores_simtest["dt"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["dt_e"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["rf"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["rf_e"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["lr"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["lr_l1"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["lr_l2"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["lr_e"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["nb"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["nb_g"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["nb_m"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["nb_c"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["svm"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["svm_po"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["svm_r"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["knn"],
                          pgmpy_hc_nonlinear_dict_scores_simtest["knn_d"]]
    pgmpy_mmhc_non_means = [pgmpy_mmhc_nonlinear_dict_scores_simtest["dt"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["dt_e"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["rf"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["rf_e"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["lr"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["lr_l1"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["lr_l2"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["lr_e"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["nb"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["nb_g"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["nb_m"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["nb_c"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["svm"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["svm_po"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["svm_r"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["knn"],
                            pgmpy_mmhc_nonlinear_dict_scores_simtest["knn_d"]]

    plt.rcParams["figure.figsize"] = [18, 18]
    plt.rcParams["figure.autolayout"] = True

    x_axis = np.arange(len(labels))
    w = 0.05  # the width of the bars

    plt.bar(x_axis + w, bn_non_means, width=0.05, label="BN_LEARN (HC)", color="lightsteelblue")
    plt.bar(x_axis + w * 2, nt_non_means, width=0.05, label="BN_LEARN (TABU)", color="cornflowerblue")
    plt.bar(x_axis + w * 3, bn_mmhc_non_means, width=0.05, label="BN_LEARN (MMHC)", color="blue")
    plt.bar(x_axis + w * 4, bn_rsmax2_non_means, width=0.05, label="BN_LEARN (RSMAX2)", color="mediumblue")
    plt.bar(x_axis + w * 5, bn_h2pc_non_means, width=0.05, label="BN_LEARN (H2PC)", color="navy")
    plt.bar(x_axis + w * 6, nt_non_means, width=0.05, label="NO_TEARS (logistic)", color="limegreen")
    plt.bar(x_axis + w * 7, nt_l2_non_means, width=0.05, label="NO_TEARS (l2)", color="forestgreen")
    plt.bar(x_axis + w * 8, nt_p_non_means, width=0.05, label="NO_TEARS (poisson)", color="darkgreen")
    plt.bar(x_axis + w * 9, p_non_means, width=0.05, label="POMEGRANATE (exact)", color="darkviolet")
    plt.bar(x_axis + w * 10, p_g_non_means, width=0.05, label="POMEGRANATE (greed)", color="rebeccapurple")
    plt.bar(x_axis + w * 11, pgmpy_mmhc_non_means, width=0.05, label="PGMPY (MMHC)", color="#FA8072")
    plt.bar(x_axis + w * 12, pgmpy_hc_non_means, width=0.05, label="PGMPY (HC)", color="#FF2400")
    plt.bar(x_axis + w * 13, pgmpy_tree_non_means, width=0.05, label="PGMPY (TREE)", color="#7C0A02")

    plt.xticks(x_axis, labels)
    plt.legend()
    plt.style.use("fivethirtyeight")
    plt.ylabel('Accuracy')
    plt.xlabel('ML Technique', labelpad=15)
    plt.title('Non-Linear Problem - Performance by library on ML technique')
    # plt.ylim(0.6, 1)
    # plt.tick_params(rotation=45)
    plt.savefig('pipeline_summary_benchmark_for_nonlinear_by_library_groupbar_simtest.png', bbox_inches='tight')
    plt.show()

    # Produce Sparse Problem by Library on Problem
    # Group by figure
    labels = ['DT_G', 'DT_E', 'RF_G', 'RF_E', 'LR', 'LR_L1', 'LR_L2', 'LR_E', 'NB_B', 'NB_G', 'NB_M', 'NB_C', 'SVM_S',
              'SVM_P', 'SVM_R', 'KNN_W', 'KNN_D']
    bn_sparse_means = [bnlearn_sparse_dict_scores_simtest["dt"], bnlearn_sparse_dict_scores_simtest["dt_e"],
                       bnlearn_sparse_dict_scores_simtest["rf"], bnlearn_sparse_dict_scores_simtest["rf_e"],
                       bnlearn_sparse_dict_scores_simtest["lr"], bnlearn_sparse_dict_scores_simtest["lr_l1"],
                       bnlearn_sparse_dict_scores_simtest["lr_l2"], bnlearn_sparse_dict_scores_simtest["lr_e"],
                       bnlearn_sparse_dict_scores_simtest["nb"], bnlearn_sparse_dict_scores_simtest["nb_g"],
                       bnlearn_sparse_dict_scores_simtest["nb_m"], bnlearn_sparse_dict_scores_simtest["nb_c"],
                       bnlearn_sparse_dict_scores_simtest["svm"], bnlearn_sparse_dict_scores_simtest["svm_po"],
                       bnlearn_sparse_dict_scores_simtest["svm_r"], bnlearn_sparse_dict_scores_simtest["knn"],
                       bnlearn_sparse_dict_scores_simtest["knn_d"]]
    bn_tabu_sparse_means = [bnlearn_tabu_sparse_dict_scores_simtest["dt"], bnlearn_tabu_sparse_dict_scores_simtest["dt_e"],
                            bnlearn_tabu_sparse_dict_scores_simtest["rf"], bnlearn_tabu_sparse_dict_scores_simtest["rf_e"],
                            bnlearn_tabu_sparse_dict_scores_simtest["lr"], bnlearn_tabu_sparse_dict_scores_simtest["lr_l1"],
                            bnlearn_tabu_sparse_dict_scores_simtest["lr_l2"], bnlearn_tabu_sparse_dict_scores_simtest["lr_e"],
                            bnlearn_tabu_sparse_dict_scores_simtest["nb"], bnlearn_tabu_sparse_dict_scores_simtest["nb_g"],
                            bnlearn_tabu_sparse_dict_scores_simtest["nb_m"], bnlearn_tabu_sparse_dict_scores_simtest["nb_c"],
                            bnlearn_tabu_sparse_dict_scores_simtest["svm"], bnlearn_tabu_sparse_dict_scores_simtest["svm_po"],
                            bnlearn_tabu_sparse_dict_scores_simtest["svm_r"], bnlearn_tabu_sparse_dict_scores_simtest["knn"],
                            bnlearn_tabu_sparse_dict_scores_simtest["knn_d"]]
    bn_mmhc_sparse_means = [bnlearn_mmhc_sparse_dict_scores_simtest["dt"], bnlearn_mmhc_sparse_dict_scores_simtest["dt_e"],
                            bnlearn_mmhc_sparse_dict_scores_simtest["rf"], bnlearn_mmhc_sparse_dict_scores_simtest["rf_e"],
                            bnlearn_mmhc_sparse_dict_scores_simtest["lr"], bnlearn_mmhc_sparse_dict_scores_simtest["lr_l1"],
                            bnlearn_mmhc_sparse_dict_scores_simtest["lr_l2"], bnlearn_mmhc_sparse_dict_scores_simtest["lr_e"],
                            bnlearn_mmhc_sparse_dict_scores_simtest["nb"], bnlearn_mmhc_sparse_dict_scores_simtest["nb_g"],
                            bnlearn_mmhc_sparse_dict_scores_simtest["nb_m"], bnlearn_mmhc_sparse_dict_scores_simtest["nb_c"],
                            bnlearn_mmhc_sparse_dict_scores_simtest["svm"], bnlearn_mmhc_sparse_dict_scores_simtest["svm_po"],
                            bnlearn_mmhc_sparse_dict_scores_simtest["svm_r"], bnlearn_mmhc_sparse_dict_scores_simtest["knn"],
                            bnlearn_mmhc_sparse_dict_scores_simtest["knn_d"]]
    bn_rsmax2_sparse_means = [bnlearn_rsmax2_sparse_dict_scores_simtest["dt"], bnlearn_rsmax2_sparse_dict_scores_simtest["dt_e"],
                              bnlearn_rsmax2_sparse_dict_scores_simtest["rf"], bnlearn_rsmax2_sparse_dict_scores_simtest["rf_e"],
                              bnlearn_rsmax2_sparse_dict_scores_simtest["lr"], bnlearn_rsmax2_sparse_dict_scores_simtest["lr_l1"],
                              bnlearn_rsmax2_sparse_dict_scores_simtest["lr_l2"], bnlearn_rsmax2_sparse_dict_scores_simtest["lr_e"],
                              bnlearn_rsmax2_sparse_dict_scores_simtest["nb"], bnlearn_rsmax2_sparse_dict_scores_simtest["nb_g"],
                              bnlearn_rsmax2_sparse_dict_scores_simtest["nb_m"], bnlearn_rsmax2_sparse_dict_scores_simtest["nb_c"],
                              bnlearn_rsmax2_sparse_dict_scores_simtest["svm"], bnlearn_rsmax2_sparse_dict_scores_simtest["svm_po"],
                              bnlearn_rsmax2_sparse_dict_scores_simtest["svm_r"], bnlearn_rsmax2_sparse_dict_scores_simtest["knn"],
                              bnlearn_rsmax2_sparse_dict_scores_simtest["knn_d"]]
    bn_h2pc_sparse_means = [bnlearn_h2pc_sparse_dict_scores_simtest["dt"], bnlearn_h2pc_sparse_dict_scores_simtest["dt_e"],
                            bnlearn_h2pc_sparse_dict_scores_simtest["rf"], bnlearn_h2pc_sparse_dict_scores_simtest["rf_e"],
                            bnlearn_h2pc_sparse_dict_scores_simtest["lr"], bnlearn_h2pc_sparse_dict_scores_simtest["lr_l1"],
                            bnlearn_h2pc_sparse_dict_scores_simtest["lr_l2"], bnlearn_h2pc_sparse_dict_scores_simtest["lr_e"],
                            bnlearn_h2pc_sparse_dict_scores_simtest["nb"], bnlearn_h2pc_sparse_dict_scores_simtest["nb_g"],
                            bnlearn_h2pc_sparse_dict_scores_simtest["nb_m"], bnlearn_h2pc_sparse_dict_scores_simtest["nb_c"],
                            bnlearn_h2pc_sparse_dict_scores_simtest["svm"], bnlearn_h2pc_sparse_dict_scores_simtest["svm_po"],
                            bnlearn_h2pc_sparse_dict_scores_simtest["svm_r"], bnlearn_h2pc_sparse_dict_scores_simtest["knn"],
                            bnlearn_h2pc_sparse_dict_scores_simtest["knn_d"]]

    nt_sparse_means = [notears_sparse_dict_scores_simtest["dt"], notears_sparse_dict_scores_simtest["dt_e"],
                       notears_sparse_dict_scores_simtest["rf"], notears_sparse_dict_scores_simtest["rf_e"],
                       notears_sparse_dict_scores_simtest["lr"], notears_sparse_dict_scores_simtest["lr_l1"],
                       notears_sparse_dict_scores_simtest["lr_l2"], notears_sparse_dict_scores_simtest["lr_e"],
                       notears_sparse_dict_scores_simtest["nb"], notears_sparse_dict_scores_simtest["nb_g"],
                       notears_sparse_dict_scores_simtest["nb_m"], notears_sparse_dict_scores_simtest["nb_c"],
                       notears_sparse_dict_scores_simtest["svm"], notears_sparse_dict_scores_simtest["svm_po"],
                       notears_sparse_dict_scores_simtest["svm_r"], notears_sparse_dict_scores_simtest["knn"],
                       notears_sparse_dict_scores_simtest["knn_d"]]
    nt_l2_sparse_means = [notears_l2_sparse_dict_scores_simtest["dt"], notears_l2_sparse_dict_scores_simtest["dt_e"],
                          notears_l2_sparse_dict_scores_simtest["rf"], notears_l2_sparse_dict_scores_simtest["rf_e"],
                          notears_l2_sparse_dict_scores_simtest["lr"], notears_l2_sparse_dict_scores_simtest["lr_l1"],
                          notears_l2_sparse_dict_scores_simtest["lr_l2"], notears_l2_sparse_dict_scores_simtest["lr_e"],
                          notears_l2_sparse_dict_scores_simtest["nb"], notears_l2_sparse_dict_scores_simtest["nb_g"],
                          notears_l2_sparse_dict_scores_simtest["nb_m"], notears_l2_sparse_dict_scores_simtest["nb_c"],
                          notears_l2_sparse_dict_scores_simtest["svm"], notears_l2_sparse_dict_scores_simtest["svm_po"],
                          notears_l2_sparse_dict_scores_simtest["svm_r"], notears_l2_sparse_dict_scores_simtest["knn"],
                          notears_l2_sparse_dict_scores_simtest["knn_d"]]
    nt_p_sparse_means = [notears_poisson_sparse_dict_scores_simtest["dt"], notears_poisson_sparse_dict_scores_simtest["dt_e"],
                         notears_poisson_sparse_dict_scores_simtest["rf"], notears_poisson_sparse_dict_scores_simtest["rf_e"],
                         notears_poisson_sparse_dict_scores_simtest["lr"], notears_poisson_sparse_dict_scores_simtest["lr_l1"],
                         notears_poisson_sparse_dict_scores_simtest["lr_l2"], notears_poisson_sparse_dict_scores_simtest["lr_e"],
                         notears_poisson_sparse_dict_scores_simtest["nb"], notears_poisson_sparse_dict_scores_simtest["nb_g"],
                         notears_poisson_sparse_dict_scores_simtest["nb_m"], notears_poisson_sparse_dict_scores_simtest["nb_c"],
                         notears_poisson_sparse_dict_scores_simtest["svm"], notears_poisson_sparse_dict_scores_simtest["svm_po"],
                         notears_poisson_sparse_dict_scores_simtest["svm_r"], notears_poisson_sparse_dict_scores_simtest["knn"],
                         notears_poisson_sparse_dict_scores_simtest["knn_d"]]

    p_sparse_means = [pomegranate_exact_sparse_dict_scores_simtest["dt"], pomegranate_exact_sparse_dict_scores_simtest["dt_e"],
                      pomegranate_exact_sparse_dict_scores_simtest["rf"], pomegranate_exact_sparse_dict_scores_simtest["rf_e"],
                      pomegranate_exact_sparse_dict_scores_simtest["lr"], pomegranate_exact_sparse_dict_scores_simtest["lr_l1"],
                      pomegranate_exact_sparse_dict_scores_simtest["lr_l2"], pomegranate_exact_sparse_dict_scores_simtest["lr_e"],
                      pomegranate_exact_sparse_dict_scores_simtest["nb"], pomegranate_exact_sparse_dict_scores_simtest["nb_g"],
                      pomegranate_exact_sparse_dict_scores_simtest["nb_m"], pomegranate_exact_sparse_dict_scores_simtest["nb_c"],
                      pomegranate_exact_sparse_dict_scores_simtest["svm"], pomegranate_exact_sparse_dict_scores_simtest["svm_po"],
                      pomegranate_exact_sparse_dict_scores_simtest["svm_r"], pomegranate_exact_sparse_dict_scores_simtest["knn"],
                      pomegranate_exact_sparse_dict_scores_simtest["knn_d"]]
    p_g_sparse_means = [pomegranate_greedy_sparse_dict_scores_simtest["dt"],
                        pomegranate_greedy_sparse_dict_scores_simtest["dt_e"],
                        pomegranate_greedy_sparse_dict_scores_simtest["rf"],
                        pomegranate_greedy_sparse_dict_scores_simtest["rf_e"],
                        pomegranate_greedy_sparse_dict_scores_simtest["lr"],
                        pomegranate_greedy_sparse_dict_scores_simtest["lr_l1"],
                        pomegranate_greedy_sparse_dict_scores_simtest["lr_l2"],
                        pomegranate_greedy_sparse_dict_scores_simtest["lr_e"],
                        pomegranate_greedy_sparse_dict_scores_simtest["nb"],
                        pomegranate_greedy_sparse_dict_scores_simtest["nb_g"],
                        pomegranate_greedy_sparse_dict_scores_simtest["nb_m"],
                        pomegranate_greedy_sparse_dict_scores_simtest["nb_c"],
                        pomegranate_greedy_sparse_dict_scores_simtest["svm"],
                        pomegranate_greedy_sparse_dict_scores_simtest["svm_po"],
                        pomegranate_greedy_sparse_dict_scores_simtest["svm_r"],
                        pomegranate_greedy_sparse_dict_scores_simtest["knn"],
                        pomegranate_greedy_sparse_dict_scores_simtest["knn_d"]]

    pgmpy_tree_sparse_means = [pgmpy_tree_sparse_dict_scores_simtest["dt"],
                               pgmpy_tree_sparse_dict_scores_simtest["dt_e"],
                               pgmpy_tree_sparse_dict_scores_simtest["rf"],
                               pgmpy_tree_sparse_dict_scores_simtest["rf_e"],
                               pgmpy_tree_sparse_dict_scores_simtest["lr"],
                               pgmpy_tree_sparse_dict_scores_simtest["lr_l1"],
                               pgmpy_tree_sparse_dict_scores_simtest["lr_l2"],
                               pgmpy_tree_sparse_dict_scores_simtest["lr_e"],
                               pgmpy_tree_sparse_dict_scores_simtest["nb"],
                               pgmpy_tree_sparse_dict_scores_simtest["nb_g"],
                               pgmpy_tree_sparse_dict_scores_simtest["nb_m"],
                               pgmpy_tree_sparse_dict_scores_simtest["nb_c"],
                               pgmpy_tree_sparse_dict_scores_simtest["svm"],
                               pgmpy_tree_sparse_dict_scores_simtest["svm_po"],
                               pgmpy_tree_sparse_dict_scores_simtest["svm_r"],
                               pgmpy_tree_sparse_dict_scores_simtest["knn"],
                               pgmpy_tree_sparse_dict_scores_simtest["knn_d"]]
    pgmpy_hc_sparse_means = [pgmpy_hc_sparse_dict_scores_simtest["dt"],
                             pgmpy_hc_sparse_dict_scores_simtest["dt_e"],
                             pgmpy_hc_sparse_dict_scores_simtest["rf"],
                             pgmpy_hc_sparse_dict_scores_simtest["rf_e"],
                             pgmpy_hc_sparse_dict_scores_simtest["lr"],
                             pgmpy_hc_sparse_dict_scores_simtest["lr_l1"],
                             pgmpy_hc_sparse_dict_scores_simtest["lr_l2"],
                             pgmpy_hc_sparse_dict_scores_simtest["lr_e"],
                             pgmpy_hc_sparse_dict_scores_simtest["nb"],
                             pgmpy_hc_sparse_dict_scores_simtest["nb_g"],
                             pgmpy_hc_sparse_dict_scores_simtest["nb_m"],
                             pgmpy_hc_sparse_dict_scores_simtest["nb_c"],
                             pgmpy_hc_sparse_dict_scores_simtest["svm"],
                             pgmpy_hc_sparse_dict_scores_simtest["svm_po"],
                             pgmpy_hc_sparse_dict_scores_simtest["svm_r"],
                             pgmpy_hc_sparse_dict_scores_simtest["knn"],
                             pgmpy_hc_sparse_dict_scores_simtest["knn_d"]]
    pgmpy_mmhc_sparse_means = [pgmpy_mmhc_sparse_dict_scores_simtest["dt"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["dt_e"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["rf"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["rf_e"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["lr"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["lr_l1"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["lr_l2"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["lr_e"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["nb"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["nb_g"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["nb_m"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["nb_c"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["svm"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["svm_po"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["svm_r"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["knn"],
                               pgmpy_mmhc_sparse_dict_scores_simtest["knn_d"]]

    plt.rcParams["figure.figsize"] = [18, 18]
    plt.rcParams["figure.autolayout"] = True

    x_axis = np.arange(len(labels))
    w = 0.05  # the width of the bars

    plt.bar(x_axis + w, bn_sparse_means, width=0.05, label="BN_LEARN (HC)", color="lightsteelblue")
    plt.bar(x_axis + w * 2, nt_sparse_means, width=0.05, label="BN_LEARN (TABU)", color="cornflowerblue")
    plt.bar(x_axis + w * 3, bn_mmhc_sparse_means, width=0.05, label="BN_LEARN (MMHC)", color="blue")
    plt.bar(x_axis + w * 4, bn_rsmax2_sparse_means, width=0.05, label="BN_LEARN (RSMAX2)", color="mediumblue")
    plt.bar(x_axis + w * 5, bn_h2pc_sparse_means, width=0.05, label="BN_LEARN (H2PC)", color="navy")
    plt.bar(x_axis + w * 6, nt_sparse_means, width=0.05, label="NO_TEARS (logistic)", color="limegreen")
    plt.bar(x_axis + w * 7, nt_l2_sparse_means, width=0.05, label="NO_TEARS (l2)", color="forestgreen")
    plt.bar(x_axis + w * 8, nt_p_sparse_means, width=0.05, label="NO_TEARS (poisson)", color="darkgreen")
    plt.bar(x_axis + w * 9, p_sparse_means, width=0.05, label="POMEGRANATE (exact)", color="darkviolet")
    plt.bar(x_axis + w * 10, p_g_sparse_means, width=0.05, label="POMEGRANATE (greed)", color="rebeccapurple")
    plt.bar(x_axis + w * 11, pgmpy_mmhc_sparse_means, width=0.05, label="PGMPY (MMHC)", color="#FA8072")
    plt.bar(x_axis + w * 12, pgmpy_hc_sparse_means, width=0.05, label="PGMPY (HC)", color="#FF2400")
    plt.bar(x_axis + w * 13, pgmpy_tree_sparse_means, width=0.05, label="PGMPY (TREE)", color="#7C0A02")

    plt.xticks(x_axis, labels)
    plt.legend()
    plt.style.use("fivethirtyeight")
    plt.ylabel('Accuracy')
    plt.xlabel('ML Technique', labelpad=15)
    plt.title('Sparse Problem - Performance by library on ML technique')
    # plt.ylim(0.6, 1)
    # plt.tick_params(rotation=45)
    plt.savefig('pipeline_summary_benchmark_for_sparse_by_library_groupbar_simtest.png', bbox_inches='tight')
    plt.show()

    # Produce Dimensional Problem by Library on Problem
    # Group by figure
    labels = ['DT_G', 'DT_E', 'RF_G', 'RF_E', 'LR', 'LR_L1', 'LR_L2', 'LR_E', 'NB_B', 'NB_G', 'NB_M', 'NB_C', 'SVM_S',
              'SVM_P', 'SVM_R', 'KNN_W', 'KNN_D']
    bn_dimension_means = [bnlearn_dimension_dict_scores_simtest["dt"], bnlearn_dimension_dict_scores_simtest["dt_e"],
                          bnlearn_dimension_dict_scores_simtest["rf"], bnlearn_dimension_dict_scores_simtest["rf_e"],
                          bnlearn_dimension_dict_scores_simtest["lr"], bnlearn_dimension_dict_scores_simtest["lr_l1"],
                          bnlearn_dimension_dict_scores_simtest["lr_l2"], bnlearn_dimension_dict_scores_simtest["lr_e"],
                          bnlearn_dimension_dict_scores_simtest["nb"], bnlearn_dimension_dict_scores_simtest["nb_g"],
                          bnlearn_dimension_dict_scores_simtest["nb_m"], bnlearn_dimension_dict_scores_simtest["nb_c"],
                          bnlearn_dimension_dict_scores_simtest["svm"], bnlearn_dimension_dict_scores_simtest["svm_po"],
                          bnlearn_dimension_dict_scores_simtest["svm_r"], bnlearn_dimension_dict_scores_simtest["knn"],
                          bnlearn_dimension_dict_scores_simtest["knn_d"]]
    bn_tabu_dimension_means = [bnlearn_tabu_dimension_dict_scores_simtest["dt"], bnlearn_tabu_dimension_dict_scores_simtest["dt_e"],
                               bnlearn_tabu_dimension_dict_scores_simtest["rf"], bnlearn_tabu_dimension_dict_scores_simtest["rf_e"],
                               bnlearn_tabu_dimension_dict_scores_simtest["lr"], bnlearn_tabu_dimension_dict_scores_simtest["lr_l1"],
                               bnlearn_tabu_dimension_dict_scores_simtest["lr_l2"], bnlearn_tabu_dimension_dict_scores_simtest["lr_e"],
                               bnlearn_tabu_dimension_dict_scores_simtest["nb"], bnlearn_tabu_dimension_dict_scores_simtest["nb_g"],
                               bnlearn_tabu_dimension_dict_scores_simtest["nb_m"], bnlearn_tabu_dimension_dict_scores_simtest["nb_c"],
                               bnlearn_tabu_dimension_dict_scores_simtest["svm"], bnlearn_tabu_dimension_dict_scores_simtest["svm_po"],
                               bnlearn_tabu_dimension_dict_scores_simtest["svm_r"], bnlearn_tabu_dimension_dict_scores_simtest["knn"],
                               bnlearn_tabu_dimension_dict_scores_simtest["knn_d"]]
    bn_mmhc_dimension_means = [bnlearn_mmhc_dimension_dict_scores_simtest["dt"], bnlearn_mmhc_dimension_dict_scores_simtest["dt_e"],
                               bnlearn_mmhc_dimension_dict_scores_simtest["rf"], bnlearn_mmhc_dimension_dict_scores_simtest["rf_e"],
                               bnlearn_mmhc_dimension_dict_scores_simtest["lr"], bnlearn_mmhc_dimension_dict_scores_simtest["lr_l1"],
                               bnlearn_mmhc_dimension_dict_scores_simtest["lr_l2"], bnlearn_mmhc_dimension_dict_scores_simtest["lr_e"],
                               bnlearn_mmhc_dimension_dict_scores_simtest["nb"], bnlearn_mmhc_dimension_dict_scores_simtest["nb_g"],
                               bnlearn_mmhc_dimension_dict_scores_simtest["nb_m"], bnlearn_mmhc_dimension_dict_scores_simtest["nb_c"],
                               bnlearn_mmhc_dimension_dict_scores_simtest["svm"], bnlearn_mmhc_dimension_dict_scores_simtest["svm_po"],
                               bnlearn_mmhc_dimension_dict_scores_simtest["svm_r"], bnlearn_mmhc_dimension_dict_scores_simtest["knn"],
                               bnlearn_mmhc_dimension_dict_scores_simtest["knn_d"]]
    bn_rsmax2_dimension_means = [bnlearn_rsmax2_dimension_dict_scores_simtest["dt"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["dt_e"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["rf"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["rf_e"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["lr"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["lr_l1"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["lr_l2"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["lr_e"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["nb"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["nb_g"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["nb_m"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["nb_c"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["svm"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["svm_po"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["svm_r"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["knn"],
                                 bnlearn_rsmax2_dimension_dict_scores_simtest["knn_d"]]
    bn_h2pc_dimension_means = [bnlearn_h2pc_dimension_dict_scores_simtest["dt"], bnlearn_h2pc_dimension_dict_scores_simtest["dt_e"],
                               bnlearn_h2pc_dimension_dict_scores_simtest["rf"], bnlearn_h2pc_dimension_dict_scores_simtest["rf_e"],
                               bnlearn_h2pc_dimension_dict_scores_simtest["lr"], bnlearn_h2pc_dimension_dict_scores_simtest["lr_l1"],
                               bnlearn_h2pc_dimension_dict_scores_simtest["lr_l2"], bnlearn_h2pc_dimension_dict_scores_simtest["lr_e"],
                               bnlearn_h2pc_dimension_dict_scores_simtest["nb"], bnlearn_h2pc_dimension_dict_scores_simtest["nb_g"],
                               bnlearn_h2pc_dimension_dict_scores_simtest["nb_m"], bnlearn_h2pc_dimension_dict_scores_simtest["nb_c"],
                               bnlearn_h2pc_dimension_dict_scores_simtest["svm"], bnlearn_h2pc_dimension_dict_scores_simtest["svm_po"],
                               bnlearn_h2pc_dimension_dict_scores_simtest["svm_r"], bnlearn_h2pc_dimension_dict_scores_simtest["knn"],
                               bnlearn_h2pc_dimension_dict_scores_simtest["knn_d"]]

    nt_dimension_means = [notears_dimension_dict_scores_simtest["dt"], notears_dimension_dict_scores_simtest["dt_e"],
                          notears_dimension_dict_scores_simtest["rf"], notears_dimension_dict_scores_simtest["rf_e"],
                          notears_dimension_dict_scores_simtest["lr"], notears_dimension_dict_scores_simtest["lr_l1"],
                          notears_dimension_dict_scores_simtest["lr_l2"], notears_dimension_dict_scores_simtest["lr_e"],
                          notears_dimension_dict_scores_simtest["nb"], notears_dimension_dict_scores_simtest["nb_g"],
                          notears_dimension_dict_scores_simtest["nb_m"], notears_dimension_dict_scores_simtest["nb_c"],
                          notears_dimension_dict_scores_simtest["svm"], notears_dimension_dict_scores_simtest["svm_po"],
                          notears_dimension_dict_scores_simtest["svm_r"], notears_dimension_dict_scores_simtest["knn"],
                          notears_dimension_dict_scores_simtest["knn_d"]]
    nt_l2_dimension_means = [notears_l2_dimension_dict_scores_simtest["dt"], notears_l2_dimension_dict_scores_simtest["dt_e"],
                             notears_l2_dimension_dict_scores_simtest["rf"], notears_l2_dimension_dict_scores_simtest["rf_e"],
                             notears_l2_dimension_dict_scores_simtest["lr"], notears_l2_dimension_dict_scores_simtest["lr_l1"],
                             notears_l2_dimension_dict_scores_simtest["lr_l2"], notears_l2_dimension_dict_scores_simtest["lr_e"],
                             notears_l2_dimension_dict_scores_simtest["nb"], notears_l2_dimension_dict_scores_simtest["nb_g"],
                             notears_l2_dimension_dict_scores_simtest["nb_m"], notears_l2_dimension_dict_scores_simtest["nb_c"],
                             notears_l2_dimension_dict_scores_simtest["svm"], notears_l2_dimension_dict_scores_simtest["svm_po"],
                             notears_l2_dimension_dict_scores_simtest["svm_r"], notears_l2_dimension_dict_scores_simtest["knn"],
                             notears_l2_dimension_dict_scores_simtest["knn_d"]]
    nt_p_dimension_means = [notears_poisson_dimension_dict_scores_simtest["dt"], notears_poisson_dimension_dict_scores_simtest["dt_e"],
                            notears_poisson_dimension_dict_scores_simtest["rf"], notears_poisson_dimension_dict_scores_simtest["rf_e"],
                            notears_poisson_dimension_dict_scores_simtest["lr"], notears_poisson_dimension_dict_scores_simtest["lr_l1"],
                            notears_poisson_dimension_dict_scores_simtest["lr_l2"],
                            notears_poisson_dimension_dict_scores_simtest["lr_e"],
                            notears_poisson_dimension_dict_scores_simtest["nb"], notears_poisson_dimension_dict_scores_simtest["nb_g"],
                            notears_poisson_dimension_dict_scores_simtest["nb_m"],
                            notears_poisson_dimension_dict_scores_simtest["nb_c"],
                            notears_poisson_dimension_dict_scores_simtest["svm"],
                            notears_poisson_dimension_dict_scores_simtest["svm_po"],
                            notears_poisson_dimension_dict_scores_simtest["svm_r"],
                            notears_poisson_dimension_dict_scores_simtest["knn"],
                            notears_poisson_dimension_dict_scores_simtest["knn_d"]]

    p_dimension_means = [pomegranate_exact_dimension_dict_scores_simtest["dt"], pomegranate_exact_dimension_dict_scores_simtest["dt_e"],
                         pomegranate_exact_dimension_dict_scores_simtest["rf"], pomegranate_exact_dimension_dict_scores_simtest["rf_e"],
                         pomegranate_exact_dimension_dict_scores_simtest["lr"],
                         pomegranate_exact_dimension_dict_scores_simtest["lr_l1"],
                         pomegranate_exact_dimension_dict_scores_simtest["lr_l2"],
                         pomegranate_exact_dimension_dict_scores_simtest["lr_e"], pomegranate_exact_dimension_dict_scores_simtest["nb"],
                         pomegranate_exact_dimension_dict_scores_simtest["nb_g"],
                         pomegranate_exact_dimension_dict_scores_simtest["nb_m"],
                         pomegranate_exact_dimension_dict_scores_simtest["nb_c"],
                         pomegranate_exact_dimension_dict_scores_simtest["svm"],
                         pomegranate_exact_dimension_dict_scores_simtest["svm_po"],
                         pomegranate_exact_dimension_dict_scores_simtest["svm_r"],
                         pomegranate_exact_dimension_dict_scores_simtest["knn"],
                         pomegranate_exact_dimension_dict_scores_simtest["knn_d"]]
    p_g_dimension_means = [pomegranate_greedy_dimension_dict_scores_simtest["dt"],
                           pomegranate_greedy_dimension_dict_scores_simtest["dt_e"],
                           pomegranate_greedy_dimension_dict_scores_simtest["rf"],
                           pomegranate_greedy_dimension_dict_scores_simtest["rf_e"],
                           pomegranate_greedy_dimension_dict_scores_simtest["lr"],
                           pomegranate_greedy_dimension_dict_scores_simtest["lr_l1"],
                           pomegranate_greedy_dimension_dict_scores_simtest["lr_l2"],
                           pomegranate_greedy_dimension_dict_scores_simtest["lr_e"],
                           pomegranate_greedy_dimension_dict_scores_simtest["nb"],
                           pomegranate_greedy_dimension_dict_scores_simtest["nb_g"],
                           pomegranate_greedy_dimension_dict_scores_simtest["nb_m"],
                           pomegranate_greedy_dimension_dict_scores_simtest["nb_c"],
                           pomegranate_greedy_dimension_dict_scores_simtest["svm"],
                           pomegranate_greedy_dimension_dict_scores_simtest["svm_po"],
                           pomegranate_greedy_dimension_dict_scores_simtest["svm_r"],
                           pomegranate_greedy_dimension_dict_scores_simtest["knn"],
                           pomegranate_greedy_dimension_dict_scores_simtest["knn_d"]]

    pgmpy_tree_dimension_means = [pgmpy_tree_dimension_dict_scores_simtest["dt"],
                                  pgmpy_tree_dimension_dict_scores_simtest["dt_e"],
                                  pgmpy_tree_dimension_dict_scores_simtest["rf"],
                                  pgmpy_tree_dimension_dict_scores_simtest["rf_e"],
                                  pgmpy_tree_dimension_dict_scores_simtest["lr"],
                                  pgmpy_tree_dimension_dict_scores_simtest["lr_l1"],
                                  pgmpy_tree_dimension_dict_scores_simtest["lr_l2"],
                                  pgmpy_tree_dimension_dict_scores_simtest["lr_e"],
                                  pgmpy_tree_dimension_dict_scores_simtest["nb"],
                                  pgmpy_tree_dimension_dict_scores_simtest["nb_g"],
                                  pgmpy_tree_dimension_dict_scores_simtest["nb_m"],
                                  pgmpy_tree_dimension_dict_scores_simtest["nb_c"],
                                  pgmpy_tree_dimension_dict_scores_simtest["svm"],
                                  pgmpy_tree_dimension_dict_scores_simtest["svm_po"],
                                  pgmpy_tree_dimension_dict_scores_simtest["svm_r"],
                                  pgmpy_tree_dimension_dict_scores_simtest["knn"],
                                  pgmpy_tree_dimension_dict_scores_simtest["knn_d"]]
    pgmpy_hc_dimension_means = [pgmpy_hc_dimension_dict_scores_simtest["dt"],
                                pgmpy_hc_dimension_dict_scores_simtest["dt_e"],
                                pgmpy_hc_dimension_dict_scores_simtest["rf"],
                                pgmpy_hc_dimension_dict_scores_simtest["rf_e"],
                                pgmpy_hc_dimension_dict_scores_simtest["lr"],
                                pgmpy_hc_dimension_dict_scores_simtest["lr_l1"],
                                pgmpy_hc_dimension_dict_scores_simtest["lr_l2"],
                                pgmpy_hc_dimension_dict_scores_simtest["lr_e"],
                                pgmpy_hc_dimension_dict_scores_simtest["nb"],
                                pgmpy_hc_dimension_dict_scores_simtest["nb_g"],
                                pgmpy_hc_dimension_dict_scores_simtest["nb_m"],
                                pgmpy_hc_dimension_dict_scores_simtest["nb_c"],
                                pgmpy_hc_dimension_dict_scores_simtest["svm"],
                                pgmpy_hc_dimension_dict_scores_simtest["svm_po"],
                                pgmpy_hc_dimension_dict_scores_simtest["svm_r"],
                                pgmpy_hc_dimension_dict_scores_simtest["knn"],
                                pgmpy_hc_dimension_dict_scores_simtest["knn_d"]]
    pgmpy_mmhc_dimension_means = [pgmpy_mmhc_dimension_dict_scores_simtest["dt"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["dt_e"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["rf"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["rf_e"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["lr"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["lr_l1"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["lr_l2"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["lr_e"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["nb"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["nb_g"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["nb_m"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["nb_c"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["svm"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["svm_po"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["svm_r"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["knn"],
                                  pgmpy_mmhc_dimension_dict_scores_simtest["knn_d"]]

    plt.rcParams["figure.figsize"] = [18, 18]
    plt.rcParams["figure.autolayout"] = True

    x_axis = np.arange(len(labels))
    w = 0.05  # the width of the bars

    plt.bar(x_axis + w, bn_dimension_means, width=0.05, label="BN_LEARN (HC)", color="lightsteelblue")
    plt.bar(x_axis + w * 2, nt_dimension_means, width=0.05, label="BN_LEARN (TABU)", color="cornflowerblue")
    plt.bar(x_axis + w * 3, bn_mmhc_dimension_means, width=0.05, label="BN_LEARN (MMHC)", color="blue")
    plt.bar(x_axis + w * 4, bn_rsmax2_dimension_means, width=0.05, label="BN_LEARN (RSMAX2)", color="mediumblue")
    plt.bar(x_axis + w * 5, bn_h2pc_dimension_means, width=0.05, label="BN_LEARN (H2PC)", color="navy")
    plt.bar(x_axis + w * 6, nt_dimension_means, width=0.05, label="NO_TEARS (logistic)", color="limegreen")
    plt.bar(x_axis + w * 7, nt_l2_dimension_means, width=0.05, label="NO_TEARS (l2)", color="forestgreen")
    plt.bar(x_axis + w * 8, nt_p_dimension_means, width=0.05, label="NO_TEARS (poisson)", color="darkgreen")
    plt.bar(x_axis + w * 9, p_dimension_means, width=0.05, label="POMEGRANATE (exact)", color="darkviolet")
    plt.bar(x_axis + w * 10, p_g_dimension_means, width=0.05, label="POMEGRANATE (greed)", color="rebeccapurple")
    plt.bar(x_axis + w * 11, pgmpy_mmhc_dimension_means, width=0.05, label="PGMPY (MMHC)", color="#FA8072")
    plt.bar(x_axis + w * 12, pgmpy_hc_dimension_means, width=0.05, label="PGMPY (HC)", color="#FF2400")
    plt.bar(x_axis + w * 13, pgmpy_tree_dimension_means, width=0.05, label="PGMPY (TREE)", color="#7C0A02")

    plt.xticks(x_axis, labels)
    plt.legend()
    plt.style.use("fivethirtyeight")
    plt.ylabel('Accuracy')
    plt.xlabel('ML Technique', labelpad=15)
    plt.title('Dimension Problem - Performance by library on ML technique')
    # plt.ylim(0.6, 1)
    # plt.tick_params(rotation=45)
    plt.savefig('pipeline_summary_benchmark_for_dimension_by_library_groupbar_simtest.png', bbox_inches='tight')
    plt.show()

write_real_to_figures()

def prediction_real_learned():
    print("#### SimCal Real/Learned-world Predictions ####")

    print("-- Exact (1-1) max(rank) output")
    real_linear_workflows = {'Decision Tree (gini)': real_linear_dt_scores, 'Decision Tree (entropy)': real_linear_dt_entropy_scores, 'Random Forest (gini)': real_linear_rf_scores, 'Random Forest (entropy)': real_linear_rf_entropy_scores,'Logistic Regression (none)': real_linear_lr_scores, 'Logistic Regression (l1)': real_linear_lr_l1_scores, 'Logistic Regression (l2)': real_linear_lr_l2_scores, 'Logistic Regression (elasticnet)': real_linear_lr_elastic_scores, 'Naive Bayes (bernoulli)': real_linear_gb_scores, 'Naive Bayes (multinomial)': real_linear_gb_multi_scores, 'Naive Bayes (gaussian)': real_linear_gb_gaussian_scores, 'Naive Bayes (complement)': real_linear_gb_complement_scores, 'Support Vector Machine (sigmoid)': real_linear_svm_scores, 'Support Vector Machine (polynomial)': real_linear_svm_poly_scores, 'Support Vector Machine (rbf)': real_linear_svm_rbf_scores, 'K Nearest Neighbor (uniform)': real_linear_knn_scores, 'K Nearest Neighbor (distance)': real_linear_knn_distance_scores}
    top_real_linear = max(real_linear_workflows, key=real_linear_workflows.get)
    print("Real world - Linear problem, Prediction: "+ top_real_linear + " (" + str(real_linear_workflows[top_real_linear]) + ")")
    sim_linear_workflows = {'BN Decision Tree (HC-gini)': bnlearn_linear_dict_scores["dt"], 'BN Decision Tree (HC-entropy)': bnlearn_linear_dict_scores["dt_e"],'BN Decision Tree (TABU-gini)': bnlearn_tabu_linear_dict_scores["dt"], 'BN Decision Tree (TABU-entropy)': bnlearn_tabu_linear_dict_scores["dt_e"],'BN Decision Tree (PC-gini)': bnlearn_pc_linear_dict_scores["dt"], 'BN Decision Tree (PC-entropy)': bnlearn_pc_linear_dict_scores["dt_e"],'BN Decision Tree (MMHC-gini)': bnlearn_mmhc_linear_dict_scores["dt"], 'BN Decision Tree (MMHC-entropy)': bnlearn_mmhc_linear_dict_scores["dt_e"],'BN Decision Tree (RSMAX2-gini)': bnlearn_rsmax2_linear_dict_scores["dt"], 'BN Decision Tree (RSMAX2-entropy)': bnlearn_rsmax2_linear_dict_scores["dt_e"],'BN Decision Tree (H2PC-gini)': bnlearn_h2pc_linear_dict_scores["dt"], 'BN Decision Tree (H2PC-entropy)': bnlearn_h2pc_linear_dict_scores["dt_e"],'NT Decision Tree (Logistic-gini)': notears_linear_dict_scores["dt"],'NT Decision Tree (Logistic-entropy)': notears_linear_dict_scores["dt_e"], 'NT Decision Tree (L2-gini)': notears_l2_linear_dict_scores["dt"],'NT Decision Tree (L2-entropy)': notears_l2_linear_dict_scores["dt_e"],'NT Decision Tree (Poisson-gini)': notears_poisson_linear_dict_scores["dt"],'NT Decision Tree (Poisson-entropy)': notears_poisson_linear_dict_scores["dt_e"],'POMEGRANATE Decision Tree (Exact-gini)': pomegranate_exact_linear_dict_scores["dt"],'POMEGRANATE Decision Tree (Exact-entropy)': pomegranate_exact_linear_dict_scores["dt_e"],'POMEGRANATE Decision Tree (Greedy-gini)': pomegranate_greedy_linear_dict_scores["dt"],'POMEGRANATE Decision Tree (Greedy-entropy)': pomegranate_greedy_linear_dict_scores["dt_e"],'PGMPY Decision Tree (HC-gini)': pgmpy_hc_linear_dict_scores["dt"],'PGMPY Decision Tree (HC-entropy)': pgmpy_hc_linear_dict_scores["dt_e"],'PGMPY Decision Tree (MMHC-gini)': pgmpy_mmhc_linear_dict_scores["dt"],'PGMPY Decision Tree (HC-entropy)': pgmpy_mmhc_linear_dict_scores["dt_e"],'PGMPY Decision Tree (TREE-gini)': pgmpy_tree_linear_dict_scores["dt"],'PGMPY Decision Tree (TREE-entropy)': pgmpy_tree_linear_dict_scores["dt_e"],'BN Random Forest (HC-gini)': bnlearn_linear_dict_scores["rf"], 'BN Random Forest (HC-entropy)': bnlearn_linear_dict_scores["rf_e"],'BN Random Forest (TABU-gini)': bnlearn_tabu_linear_dict_scores["rf"], 'BN Random Forest (TABU-entropy)': bnlearn_tabu_linear_dict_scores["rf_e"],'BN Random Forest (PC-gini)': bnlearn_pc_linear_dict_scores["rf"], 'BN Random Forest (PC-entropy)': bnlearn_pc_linear_dict_scores["rf_e"],'BN Random Forest (MMHC-gini)': bnlearn_mmhc_linear_dict_scores["rf"], 'BN Random Forest (MMHC-entropy)': bnlearn_mmhc_linear_dict_scores["rf_e"],'BN Random Forest (RSMAX2-gini)': bnlearn_rsmax2_linear_dict_scores["rf"], 'BN Random Forest (RSMAX2-entropy)': bnlearn_rsmax2_linear_dict_scores["rf_e"],'BN Random Forest (H2PC-gini)': bnlearn_h2pc_linear_dict_scores["rf"], 'BN Random Forest (H2PC-entropy)': bnlearn_h2pc_linear_dict_scores["rf_e"],'NT Random Forest (Logistic-gini)': notears_linear_dict_scores["rf"],'NT Random Forest (Logistic-entropy)': notears_linear_dict_scores["rf_e"],'NT Random Forest (L2-gini)': notears_l2_linear_dict_scores["rf"],'NT Random Forest (l2-entropy)': notears_l2_linear_dict_scores["rf_e"],'NT Random Forest (Poisson-gini)': notears_poisson_linear_dict_scores["rf"],'NT Random Forest (Poisson-entropy)': notears_poisson_linear_dict_scores["rf_e"],'POMEGRANATE Random Forest (Exact-gini)': pomegranate_exact_linear_dict_scores["rf"],'POMEGRANATE Random Forest (Exact-entropy)': pomegranate_exact_linear_dict_scores["rf_e"],'POMEGRANATE Random Forest (Greedy-gini)': pomegranate_greedy_linear_dict_scores["rf"],'POMEGRANATE Random Forest (Greedy-entropy)': pomegranate_greedy_linear_dict_scores["rf_e"],'PGMPY Random Forest (HC-gini)': pgmpy_hc_linear_dict_scores["rf"],'PGMPY Random Forest (HC-entropy)': pgmpy_hc_linear_dict_scores["rf_e"],'PGMPY Random Forest (MMHC-gini)': pgmpy_mmhc_linear_dict_scores["rf"],'PGMPY Random Forest (HC-entropy)': pgmpy_mmhc_linear_dict_scores["rf_e"],'PGMPY Random Forest (TREE-gini)': pgmpy_tree_linear_dict_scores["rf"],'PGMPY Random Forest (TREE-entropy)': pgmpy_tree_linear_dict_scores["rf_e"], 'BN Logistic Regression (HC-none)': bnlearn_linear_dict_scores["lr"],'BN Logistic Regression (HC-l1)': bnlearn_linear_dict_scores["lr_l1"],'BN Logistic Regression (HC-l2)': bnlearn_linear_dict_scores["lr_l2"],'BN Logistic Regression (HC-elastic)': bnlearn_linear_dict_scores["lr_e"], 'BN Logistic Regression (TABU-none)': bnlearn_tabu_linear_dict_scores["lr"],'BN Logistic Regression (TABU-l1)': bnlearn_tabu_linear_dict_scores["lr_l1"],'BN Logistic Regression (TABU-l2)': bnlearn_tabu_linear_dict_scores["lr_l2"],'BN Logistic Regression (TABU-elastic)': bnlearn_tabu_linear_dict_scores["lr_e"], 'BN Logistic Regression (PC-none)': bnlearn_pc_linear_dict_scores["lr"],'BN Logistic Regression (PC-l1)': bnlearn_pc_linear_dict_scores["lr_l1"],'BN Logistic Regression (PC-l2)': bnlearn_pc_linear_dict_scores["lr_l2"],'BN Logistic Regression (PC-elastic)': bnlearn_pc_linear_dict_scores["lr_e"], 'BN Logistic Regression (MMHC-none)': bnlearn_mmhc_linear_dict_scores["lr"],'BN Logistic Regression (MMHC-l1)': bnlearn_mmhc_linear_dict_scores["lr_l1"],'BN Logistic Regression (MMHC-l2)': bnlearn_mmhc_linear_dict_scores["lr_l2"],'BN Logistic Regression (MMHC-elastic)': bnlearn_mmhc_linear_dict_scores["lr_e"], 'BN Logistic Regression (RSMAX2-none)': bnlearn_rsmax2_linear_dict_scores["lr"],'BN Logistic Regression (RSMAX2-l1)': bnlearn_rsmax2_linear_dict_scores["lr_l1"],'BN Logistic Regression (RSMAX2-l2)': bnlearn_rsmax2_linear_dict_scores["lr_l2"],'BN Logistic Regression (RSMAX2-elastic)': bnlearn_rsmax2_linear_dict_scores["lr_e"], 'BN Logistic Regression (H2PC-none)': bnlearn_h2pc_linear_dict_scores["lr"],'BN Logistic Regression (H2PC-l1)': bnlearn_h2pc_linear_dict_scores["lr_l1"],'BN Logistic Regression (H2PC-l2)': bnlearn_h2pc_linear_dict_scores["lr_l2"],'BN Logistic Regression (H2PC-elastic)': bnlearn_h2pc_linear_dict_scores["lr_e"], 'POMEGRANATE Logistic Regression (Exact-none)': pomegranate_exact_linear_dict_scores["lr"],'POMEGRANATE Logistic Regression (Exact-l1)': pomegranate_exact_linear_dict_scores["lr_l1"],'POMEGRANATE Logistic Regression (Exact-l2)': pomegranate_exact_linear_dict_scores["lr_l2"],'POMEGRANATE Logistic Regression (Exact-elastic)': pomegranate_exact_linear_dict_scores["lr_e"],'POMEGRANATE Logistic Regression (Greedy-none)': pomegranate_greedy_linear_dict_scores["lr"],'POMEGRANATE Logistic Regression (Greedy-l1)': pomegranate_greedy_linear_dict_scores["lr_l1"],'POMEGRANATE Logistic Regression (Greedy-l2)': pomegranate_greedy_linear_dict_scores["lr_l2"],'POMEGRANATE Logistic Regression (Greedy-elastic)': pomegranate_greedy_linear_dict_scores["lr_e"],'PGMPY Logistic Regression (HC-none)': pgmpy_hc_linear_dict_scores["lr"],'PGMPY Logistic Regression (HC-l1)': pgmpy_hc_linear_dict_scores["lr_l1"],'PGMPY Logistic Regression (MMHC-l2)': pgmpy_mmhc_linear_dict_scores["lr_l2"],'PGMPY Logistic Regression (HC-elastic)': pgmpy_mmhc_linear_dict_scores["lr_e"],'PGMPY Logistic Regression (TREE-none)': pgmpy_tree_linear_dict_scores["lr"],'PGMPY Logistic Regression (TREE-l1)': pgmpy_tree_linear_dict_scores["lr_l1"],'PGMPY Logistic Regression (TREE-l2)': pgmpy_tree_linear_dict_scores["lr_l2"],'PGMPY Logistic Regression (TREE-elastic)': pgmpy_tree_linear_dict_scores["lr_e"], 'PGMPY Logistic Regression (MMHC-none)': pgmpy_mmhc_linear_dict_scores["lr"],'PGMPY Logistic Regression (MMHC-l1)': pgmpy_mmhc_linear_dict_scores["lr_l1"],'PGMPY Logistic Regression (MMHC-l2)': pgmpy_mmhc_linear_dict_scores["lr_l2"],'PGMPY Logistic Regression (MMHC-elastic)': pgmpy_mmhc_linear_dict_scores["lr_e"],'NT Logistic Regression (Logistic-none)': notears_linear_dict_scores["lr"], 'NT Logistic Regression (Logistic-l1)': notears_linear_dict_scores["lr_l1"], 'NT Logistic Regression (Logistic-l2)': notears_linear_dict_scores["lr_l2"], 'NT Logistic Regression (Logistic-elastic)': notears_linear_dict_scores["lr_e"],'NT Logistic Regression (L2-none)': notears_l2_linear_dict_scores["lr"], 'NT Logistic Regression (L2-l1)': notears_l2_linear_dict_scores["lr_l1"], 'NT Logistic Regression (L2-l2)': notears_l2_linear_dict_scores["lr_l2"], 'NT Logistic Regression (L2-elastic)': notears_l2_linear_dict_scores["lr_e"],'NT Logistic Regression (Poisson-none)': notears_poisson_linear_dict_scores["lr"], 'NT Logistic Regression (Poisson-l1)': notears_poisson_linear_dict_scores["lr_l1"], 'NT Logistic Regression (Poisson-l2)': notears_poisson_linear_dict_scores["lr_l2"], 'NT Logistic Regression (Poisson-elastic)': notears_poisson_linear_dict_scores["lr_e"], 'BN Naive Bayes (HC-bernoulli)': bnlearn_linear_dict_scores["nb"],'BN Naive Bayes (HC-gaussian)': bnlearn_linear_dict_scores["nb_g"],'BN Naive Bayes (HC-multinomial)': bnlearn_linear_dict_scores["nb_m"],'BN Naive Bayes (HC-complement)': bnlearn_linear_dict_scores["nb_c"],'BN Naive Bayes (TABU-bernoulli)': bnlearn_tabu_linear_dict_scores["nb"],'BN Naive Bayes (TABU-gaussian)': bnlearn_tabu_linear_dict_scores["nb_g"],'BN Naive Bayes (TABU-multinomial)': bnlearn_tabu_linear_dict_scores["nb_m"],'BN Naive Bayes (TABU-complement)': bnlearn_tabu_linear_dict_scores["nb_c"],'BN Naive Bayes (PC-bernoulli)': bnlearn_pc_linear_dict_scores["nb"],'BN Naive Bayes (PC-gaussian)': bnlearn_pc_linear_dict_scores["nb_g"],'BN Naive Bayes (PC-multinomial)': bnlearn_pc_linear_dict_scores["nb_m"],'BN Naive Bayes (PC-complement)': bnlearn_pc_linear_dict_scores["nb_c"], 'BN Naive Bayes (MMHC-bernoulli)': bnlearn_mmhc_linear_dict_scores["nb"],'BN Naive Bayes (MMHC-gaussian)': bnlearn_mmhc_linear_dict_scores["nb_g"],'BN Naive Bayes (MMHC-multinomial)': bnlearn_mmhc_linear_dict_scores["nb_m"],'BN Naive Bayes (MMHC-complement)': bnlearn_mmhc_linear_dict_scores["nb_c"],'BN Naive Bayes (RSMAX2-bernoulli)': bnlearn_rsmax2_linear_dict_scores["nb"],'BN Naive Bayes (RSMAX2-gaussian)': bnlearn_rsmax2_linear_dict_scores["nb_g"],'BN Naive Bayes (RSMAX2-multinomial)': bnlearn_rsmax2_linear_dict_scores["nb_m"],'BN Naive Bayes (RSMAX2-complement)': bnlearn_rsmax2_linear_dict_scores["nb_c"],'BN Naive Bayes (H2PC-bernoulli)': bnlearn_h2pc_linear_dict_scores["nb"],'BN Naive Bayes (H2PC-gaussian)': bnlearn_h2pc_linear_dict_scores["nb_g"],'BN Naive Bayes (H2PC-multinomial)': bnlearn_h2pc_linear_dict_scores["nb_m"],'BN Naive Bayes (H2PC-complement)': bnlearn_h2pc_linear_dict_scores["nb_c"],'NT Naive Bayes (Logistic-bernoulli)': notears_linear_dict_scores["nb"],'NT Naive Bayes (Logistic-gaussian)': notears_linear_dict_scores["nb_g"],'NT Naive Bayes (Logistic-multinomial)': notears_linear_dict_scores["nb_m"],'NT Naive Bayes (Logistic-complement)': notears_linear_dict_scores["nb_c"], 'NT Naive Bayes (L2-bernoulli)': notears_l2_linear_dict_scores["nb"],'NT Naive Bayes (L2-gaussian)': notears_l2_linear_dict_scores["nb_g"],'NT Naive Bayes (L2-multinomial)': notears_l2_linear_dict_scores["nb_m"],'NT Naive Bayes (L2-complement)': notears_l2_linear_dict_scores["nb_c"],'NT Naive Bayes (Poisson-bernoulli)': notears_poisson_linear_dict_scores["nb"],'NT Naive Bayes (Poisson-gaussian)': notears_poisson_linear_dict_scores["nb_g"],'NT Naive Bayes (Poisson-multinomial)': notears_poisson_linear_dict_scores["nb_m"],'NT Naive Bayes (Poisson-complement)': notears_poisson_linear_dict_scores["nb_c"],'POMEGRANATE Naive Bayes (Greedy-bernoulli)': pomegranate_greedy_linear_dict_scores["nb"],'POMEGRANATE Naive Bayes (Greedy-gaussian)': pomegranate_greedy_linear_dict_scores["nb_g"],'POMEGRANATE Naive Bayes (Greedy-multinomial)': pomegranate_greedy_linear_dict_scores["nb_m"],'POMEGRANATE Naive Bayes (Greedy-complement)': pomegranate_greedy_linear_dict_scores["nb_c"],'POMEGRANATE Naive Bayes (Exact-bernoulli)': pomegranate_exact_linear_dict_scores["nb"],'POMEGRANATE Naive Bayes (Exact-gaussian)': pomegranate_exact_linear_dict_scores["nb_g"],'POMEGRANATE Naive Bayes (Exact-multinomial)': pomegranate_exact_linear_dict_scores["nb_m"],'POMEGRANATE Naive Bayes (Exact-complement)': pomegranate_exact_linear_dict_scores["nb_c"], 'PGMPY Naive Bayes (HC-bernoulli)': pgmpy_hc_linear_dict_scores["nb"],'PGMPY Naive Bayes (HC-gaussian)': pgmpy_hc_linear_dict_scores["nb_g"],'PGMPY Naive Bayes (HC-multinomial)': pgmpy_hc_linear_dict_scores["nb_m"],'PGMPY Naive Bayes (HC-complement)': pgmpy_hc_linear_dict_scores["nb_c"], 'PGMPY Naive Bayes (MMHC-bernoulli)': pgmpy_mmhc_linear_dict_scores["nb"],'PGMPY Naive Bayes (MMHC-gaussian)': pgmpy_mmhc_linear_dict_scores["nb_g"],'PGMPY Naive Bayes (MMHC-multinomial)': pgmpy_mmhc_linear_dict_scores["nb_m"],'PGMPY Naive Bayes (MMHC-complement)': pgmpy_mmhc_linear_dict_scores["nb_c"], 'PGMPY Naive Bayes (TREE-bernoulli)': pgmpy_tree_linear_dict_scores["nb"],'PGMPY Naive Bayes (TREE-gaussian)': pgmpy_tree_linear_dict_scores["nb_g"],'PGMPY Naive Bayes (TREE-multinomial)': pgmpy_tree_linear_dict_scores["nb_m"],'PGMPY Naive Bayes (TREE-complement)': pgmpy_tree_linear_dict_scores["nb_c"], 'BN Support Vector Machine (HC-sigmoid)': bnlearn_linear_dict_scores["svm"], 'BN Support Vector Machine (HC-polynomial)': bnlearn_linear_dict_scores["svm_po"], 'BN Support Vector Machine (HC-rbf)': bnlearn_linear_dict_scores["svm_r"], 'BN Support Vector Machine (TABU-sigmoid)': bnlearn_tabu_linear_dict_scores["svm"], 'BN Support Vector Machine (TABU-polynomial)': bnlearn_tabu_linear_dict_scores["svm_po"], 'BN Support Vector Machine (TABU-rbf)': bnlearn_tabu_linear_dict_scores["svm_r"],'BN Support Vector Machine (PC-sigmoid)': bnlearn_pc_linear_dict_scores["svm"], 'BN Support Vector Machine (PC-polynomial)': bnlearn_pc_linear_dict_scores["svm_po"], 'BN Support Vector Machine (PC-rbf)': bnlearn_pc_linear_dict_scores["svm_r"],'BN Support Vector Machine (MMHC-sigmoid)': bnlearn_mmhc_linear_dict_scores["svm"], 'BN Support Vector Machine (MMHC-polynomial)': bnlearn_mmhc_linear_dict_scores["svm_po"], 'BN Support Vector Machine (MMHC-rbf)': bnlearn_mmhc_linear_dict_scores["svm_r"],'BN Support Vector Machine (RSMAX2-sigmoid)': bnlearn_rsmax2_linear_dict_scores["svm"], 'BN Support Vector Machine (RSMAX2-polynomial)': bnlearn_rsmax2_linear_dict_scores["svm_po"], 'BN Support Vector Machine (RSMAX2-rbf)': bnlearn_rsmax2_linear_dict_scores["svm_r"],'BN Support Vector Machine (H2PC-sigmoid)': bnlearn_h2pc_linear_dict_scores["svm"], 'BN Support Vector Machine (H2PC-polynomial)': bnlearn_h2pc_linear_dict_scores["svm_po"], 'BN Support Vector Machine (H2PC-rbf)': bnlearn_h2pc_linear_dict_scores["svm_r"],'NT Support Vector Machine (logistic-sigmoid)': notears_linear_dict_scores["svm"],'NT Support Vector Machine (logistic-polynomial)': notears_linear_dict_scores["svm_po"],'NT Support Vector Machine (logistic-rbf)': notears_linear_dict_scores["svm_r"],'NT Support Vector Machine (L2-sigmoid)': notears_l2_linear_dict_scores["svm"],'NT Support Vector Machine (L2-polynomial)': notears_l2_linear_dict_scores["svm_po"],'NT Support Vector Machine (L2-rbf)': notears_l2_linear_dict_scores["svm_r"],'NT Support Vector Machine (Poisson-sigmoid)': notears_poisson_linear_dict_scores["svm"],'NT Support Vector Machine (Poisson-polynomial)': notears_poisson_linear_dict_scores["svm_po"],'NT Support Vector Machine (Poisson-rbf)': notears_poisson_linear_dict_scores["svm_r"], 'Pomegranate Support Vector Machine (Exact-sigmoid)': pomegranate_exact_linear_dict_scores["svm"],'Pomegranate Support Vector Machine (Exact-polynomial)': pomegranate_exact_linear_dict_scores["svm_po"],'Pomegranate Support Vector Machine (Exact-rbf)': pomegranate_exact_linear_dict_scores["svm_r"], 'Pomegranate Support Vector Machine (Greedy-sigmoid)': pomegranate_greedy_linear_dict_scores["svm"],'Pomegranate Support Vector Machine (Greedy-polynomial)': pomegranate_greedy_linear_dict_scores["svm_po"],'Pomegranate Support Vector Machine (Greedy-rbf)': pomegranate_greedy_linear_dict_scores["svm_r"],  'PGMPY Support Vector Machine (HC-sigmoid)': pgmpy_hc_linear_dict_scores["svm"],'PGMPY Support Vector Machine (HC-polynomial)': pgmpy_hc_linear_dict_scores["svm_po"],'PGMPY Support Vector Machine (HC-rbf)': pgmpy_hc_linear_dict_scores["svm_r"], 'PGMPY Support Vector Machine (MMHC-sigmoid)': pgmpy_mmhc_linear_dict_scores["svm"],'PGMPY Support Vector Machine (MMHC-polynomial)': pgmpy_mmhc_linear_dict_scores["svm_po"],'PGMPY Support Vector Machine (MMHC-rbf)': pgmpy_mmhc_linear_dict_scores["svm_r"], 'PGMPY Support Vector Machine (TREE-sigmoid)': pgmpy_tree_linear_dict_scores["svm"],'PGMPY Support Vector Machine (TREE-polynomial)': pgmpy_tree_linear_dict_scores["svm_po"],'PGMPY Support Vector Machine (TREE-rbf)': pgmpy_tree_linear_dict_scores["svm_r"],'BN K Nearest Neighbor (HC-weight)': bnlearn_linear_dict_scores["knn"],'BN K Nearest Neighbor (HC-distance)': bnlearn_linear_dict_scores["knn_d"],'BN K Nearest Neighbor (TABU-weight)': bnlearn_tabu_linear_dict_scores["knn"],'BN K Nearest Neighbor (TABU-distance)': bnlearn_tabu_linear_dict_scores["knn_d"],'BN K Nearest Neighbor (PC-weight)': bnlearn_pc_linear_dict_scores["knn"],'BN K Nearest Neighbor (PC-distance)': bnlearn_pc_linear_dict_scores["knn_d"],'BN K Nearest Neighbor (MMHC-weight)': bnlearn_mmhc_linear_dict_scores["knn"],'BN K Nearest Neighbor (MMHC-distance)': bnlearn_mmhc_linear_dict_scores["knn_d"],'BN K Nearest Neighbor (RSMAX2-weight)': bnlearn_rsmax2_linear_dict_scores["knn"],'BN K Nearest Neighbor (RSMAX2-distance)': bnlearn_rsmax2_linear_dict_scores["knn_d"],'BN K Nearest Neighbor (H2PC-weight)': bnlearn_h2pc_linear_dict_scores["knn"],'BN K Nearest Neighbor (H2PC-distance)': bnlearn_h2pc_linear_dict_scores["knn_d"],'NT K Nearest Neighbor (Logistic-weight)': notears_linear_dict_scores["knn"], 'NT K Nearest Neighbor (Logistic-distance)': notears_linear_dict_scores["knn_d"],'NT K Nearest Neighbor (L2-weight)': notears_l2_linear_dict_scores["knn"], 'NT K Nearest Neighbor (L2-distance)': notears_l2_linear_dict_scores["knn_d"], 'NT K Nearest Neighbor (Poisson-weight)': notears_poisson_linear_dict_scores["knn"], 'NT K Nearest Neighbor (Poisson-distance)': notears_poisson_linear_dict_scores["knn_d"],  'POMEGRANATE K Nearest Neighbor (Exact-weight)': pomegranate_exact_linear_dict_scores["knn"], 'POMEGRANATE K Nearest Neighbor (Exact-distance)': pomegranate_exact_linear_dict_scores["knn_d"], 'POMEGRANATE K Nearest Neighbor (Greedy-weight)': pomegranate_greedy_linear_dict_scores["knn"], 'POMEGRANATE K Nearest Neighbor (Greedy-distance)': pomegranate_greedy_linear_dict_scores["knn_d"], 'PGMPY K Nearest Neighbor (HC-weight)': pgmpy_hc_linear_dict_scores["knn"], 'PGMPY K Nearest Neighbor (HC-distance)': pgmpy_hc_linear_dict_scores["knn_d"], 'PGMPY K Nearest Neighbor (MMHC-weight)': pgmpy_mmhc_linear_dict_scores["knn"], 'PGMPY K Nearest Neighbor (MMHC-distance)': pgmpy_mmhc_linear_dict_scores["knn_d"], 'PGMPY K Nearest Neighbor (TREE-weight)': pgmpy_tree_linear_dict_scores["knn"], 'PGMPY K Nearest Neighbor (TREE-distance)': pgmpy_tree_linear_dict_scores["knn_d"]}
    top_learned_linear = max(sim_linear_workflows, key=sim_linear_workflows.get)
    print("Learned world - Linear problem, Prediction: "+ top_learned_linear + " (" + str(sim_linear_workflows[top_learned_linear]) + ")")

    real_nonlinear_workflows = {'Decision Tree (gini)': real_nonlinear_dt_scores,
                             'Decision Tree (entropy)': real_nonlinear_dt_entropy_scores,
                             'Random Forest (gini)': real_nonlinear_rf_scores,
                             'Random Forest (entropy)': real_nonlinear_rf_entropy_scores,
                             'Logistic Regression (none)': real_nonlinear_lr_scores,
                             'Logistic Regression (l1)': real_nonlinear_lr_l1_scores,
                             'Logistic Regression (l2)': real_nonlinear_lr_l2_scores,
                             'Logistic Regression (elasticnet)': real_nonlinear_lr_elastic_scores,
                             'Naive Bayes (bernoulli)': real_nonlinear_gb_scores,
                             'Naive Bayes (multinomial)': real_nonlinear_gb_multi_scores,
                             'Naive Bayes (gaussian)': real_nonlinear_gb_gaussian_scores,
                             'Naive Bayes (complement)': real_nonlinear_gb_complement_scores,
                             'Support Vector Machine (sigmoid)': real_nonlinear_svm_scores,
                             'Support Vector Machine (polynomial)': real_nonlinear_svm_poly_scores,
                             'Support Vector Machine (rbf)': real_nonlinear_svm_rbf_scores,
                             'K Nearest Neighbor (uniform)': real_nonlinear_knn_scores,
                             'K Nearest Neighbor (distance)': real_nonlinear_knn_distance_scores}
    top_real_nonlinear = max(real_nonlinear_workflows, key=real_nonlinear_workflows.get)
    print("Real world - Nonlinear problem, Prediction: "+ top_real_nonlinear + " (" + str(real_nonlinear_workflows[top_real_nonlinear]) + ")")

    sim_nonlinear_workflows = {'BN Decision Tree (HC-gini)': bnlearn_nonlinear_dict_scores["dt"],
                            'BN Decision Tree (HC-entropy)': bnlearn_nonlinear_dict_scores["dt_e"],
                            'BN Decision Tree (TABU-gini)': bnlearn_tabu_nonlinear_dict_scores["dt"],
                            'BN Decision Tree (TABU-entropy)': bnlearn_tabu_nonlinear_dict_scores["dt_e"],
                            #'BN Decision Tree (PC-gini)': bnlearn_pc_nonlinear_dict_scores["dt"],
                            #'BN Decision Tree (PC-entropy)': bnlearn_pc_nonlinear_dict_scores["dt_e"],
                            'BN Decision Tree (MMHC-gini)': bnlearn_mmhc_nonlinear_dict_scores["dt"],
                            'BN Decision Tree (MMHC-entropy)': bnlearn_mmhc_nonlinear_dict_scores["dt_e"],
                            'BN Decision Tree (RSMAX2-gini)': bnlearn_rsmax2_nonlinear_dict_scores["dt"],
                            'BN Decision Tree (RSMAX2-entropy)': bnlearn_rsmax2_nonlinear_dict_scores["dt_e"],
                            'BN Decision Tree (H2PC-gini)': bnlearn_h2pc_nonlinear_dict_scores["dt"],
                            'BN Decision Tree (H2PC-entropy)': bnlearn_h2pc_nonlinear_dict_scores["dt_e"],
                            'NT Decision Tree (Logistic-gini)': notears_nonlinear_dict_scores["dt"],
                            'NT Decision Tree (Logistic-entropy)': notears_nonlinear_dict_scores["dt_e"],
                            'NT Decision Tree (L2-gini)': notears_l2_nonlinear_dict_scores["dt"],
                            'NT Decision Tree (L2-entropy)': notears_l2_nonlinear_dict_scores["dt_e"],
                            'NT Decision Tree (Poisson-gini)': notears_poisson_nonlinear_dict_scores["dt"],
                            'NT Decision Tree (Poisson-entropy)': notears_poisson_nonlinear_dict_scores["dt_e"],
                            'POMEGRANATE Decision Tree (Exact-gini)': pomegranate_exact_nonlinear_dict_scores["dt"],
                            'POMEGRANATE Decision Tree (Exact-entropy)': pomegranate_exact_nonlinear_dict_scores["dt_e"],
                            'POMEGRANATE Decision Tree (Greedy-gini)': pomegranate_greedy_nonlinear_dict_scores["dt"],
                            'POMEGRANATE Decision Tree (Greedy-entropy)': pomegranate_greedy_nonlinear_dict_scores["dt_e"],
                            'PGMPY Decision Tree (HC-gini)': pgmpy_hc_nonlinear_dict_scores["dt"],
                            'PGMPY Decision Tree (HC-entropy)': pgmpy_hc_nonlinear_dict_scores["dt_e"],
                            'PGMPY Decision Tree (MMHC-gini)': pgmpy_mmhc_nonlinear_dict_scores["dt"],
                            'PGMPY Decision Tree (HC-entropy)': pgmpy_mmhc_nonlinear_dict_scores["dt_e"],
                            'PGMPY Decision Tree (TREE-gini)': pgmpy_tree_nonlinear_dict_scores["dt"],
                            'PGMPY Decision Tree (TREE-entropy)': pgmpy_tree_nonlinear_dict_scores["dt_e"],
                            'BN Random Forest (HC-gini)': bnlearn_nonlinear_dict_scores["rf"],
                            'BN Random Forest (HC-entropy)': bnlearn_nonlinear_dict_scores["rf_e"],
                            'BN Random Forest (TABU-gini)': bnlearn_tabu_nonlinear_dict_scores["rf"],
                            'BN Random Forest (TABU-entropy)': bnlearn_tabu_nonlinear_dict_scores["rf_e"],
                            #'BN Random Forest (PC-gini)': bnlearn_pc_nonlinear_dict_scores["rf"],
                            #'BN Random Forest (PC-entropy)': bnlearn_pc_nonlinear_dict_scores["rf_e"],
                            'BN Random Forest (MMHC-gini)': bnlearn_mmhc_nonlinear_dict_scores["rf"],
                            'BN Random Forest (MMHC-entropy)': bnlearn_mmhc_nonlinear_dict_scores["rf_e"],
                            'BN Random Forest (RSMAX2-gini)': bnlearn_rsmax2_nonlinear_dict_scores["rf"],
                            'BN Random Forest (RSMAX2-entropy)': bnlearn_rsmax2_nonlinear_dict_scores["rf_e"],
                            'BN Random Forest (H2PC-gini)': bnlearn_h2pc_nonlinear_dict_scores["rf"],
                            'BN Random Forest (H2PC-entropy)': bnlearn_h2pc_nonlinear_dict_scores["rf_e"],
                            'NT Random Forest (Logistic-gini)': notears_nonlinear_dict_scores["rf"],
                            'NT Random Forest (Logistic-entropy)': notears_nonlinear_dict_scores["rf_e"],
                            'NT Random Forest (L2-gini)': notears_l2_nonlinear_dict_scores["rf"],
                            'NT Random Forest (l2-entropy)': notears_l2_nonlinear_dict_scores["rf_e"],
                            'NT Random Forest (Poisson-gini)': notears_poisson_nonlinear_dict_scores["rf"],
                            'NT Random Forest (Poisson-entropy)': notears_poisson_nonlinear_dict_scores["rf_e"],
                            'POMEGRANATE Random Forest (Exact-gini)': pomegranate_exact_nonlinear_dict_scores["rf"],
                            'POMEGRANATE Random Forest (Exact-entropy)': pomegranate_exact_nonlinear_dict_scores["rf_e"],
                            'POMEGRANATE Random Forest (Greedy-gini)': pomegranate_greedy_nonlinear_dict_scores["rf"],
                            'POMEGRANATE Random Forest (Greedy-entropy)': pomegranate_greedy_nonlinear_dict_scores["rf_e"],
                            'PGMPY Random Forest (HC-gini)': pgmpy_hc_nonlinear_dict_scores["rf"],
                            'PGMPY Random Forest (HC-entropy)': pgmpy_hc_nonlinear_dict_scores["rf_e"],
                            'PGMPY Random Forest (MMHC-gini)': pgmpy_mmhc_nonlinear_dict_scores["rf"],
                            'PGMPY Random Forest (HC-entropy)': pgmpy_mmhc_nonlinear_dict_scores["rf_e"],
                            'PGMPY Random Forest (TREE-gini)': pgmpy_tree_nonlinear_dict_scores["rf"],
                            'PGMPY Random Forest (TREE-entropy)': pgmpy_tree_nonlinear_dict_scores["rf_e"],
                            'BN Logistic Regression (HC-none)': bnlearn_nonlinear_dict_scores["lr"],
                            'BN Logistic Regression (HC-l1)': bnlearn_nonlinear_dict_scores["lr_l1"],
                            'BN Logistic Regression (HC-l2)': bnlearn_nonlinear_dict_scores["lr_l2"],
                            'BN Logistic Regression (HC-elastic)': bnlearn_nonlinear_dict_scores["lr_e"],
                            'BN Logistic Regression (TABU-none)': bnlearn_tabu_nonlinear_dict_scores["lr"],
                            'BN Logistic Regression (TABU-l1)': bnlearn_tabu_nonlinear_dict_scores["lr_l1"],
                            'BN Logistic Regression (TABU-l2)': bnlearn_tabu_nonlinear_dict_scores["lr_l2"],
                            'BN Logistic Regression (TABU-elastic)': bnlearn_tabu_nonlinear_dict_scores["lr_e"],
                            #'BN Logistic Regression (PC-none)': bnlearn_pc_nonlinear_dict_scores["lr"],
                            #'BN Logistic Regression (PC-l1)': bnlearn_pc_nonlinear_dict_scores["lr_l1"],
                            #'BN Logistic Regression (PC-l2)': bnlearn_pc_nonlinear_dict_scores["lr_l2"],
                            #'BN Logistic Regression (PC-elastic)': bnlearn_pc_nonlinear_dict_scores["lr_e"],
                            'BN Logistic Regression (MMHC-none)': bnlearn_mmhc_nonlinear_dict_scores["lr"],
                            'BN Logistic Regression (MMHC-l1)': bnlearn_mmhc_nonlinear_dict_scores["lr_l1"],
                            'BN Logistic Regression (MMHC-l2)': bnlearn_mmhc_nonlinear_dict_scores["lr_l2"],
                            'BN Logistic Regression (MMHC-elastic)': bnlearn_mmhc_nonlinear_dict_scores["lr_e"],
                            'BN Logistic Regression (RSMAX2-none)': bnlearn_rsmax2_nonlinear_dict_scores["lr"],
                            'BN Logistic Regression (RSMAX2-l1)': bnlearn_rsmax2_nonlinear_dict_scores["lr_l1"],
                            'BN Logistic Regression (RSMAX2-l2)': bnlearn_rsmax2_nonlinear_dict_scores["lr_l2"],
                            'BN Logistic Regression (RSMAX2-elastic)': bnlearn_rsmax2_nonlinear_dict_scores["lr_e"],
                            'BN Logistic Regression (H2PC-none)': bnlearn_h2pc_nonlinear_dict_scores["lr"],
                            'BN Logistic Regression (H2PC-l1)': bnlearn_h2pc_nonlinear_dict_scores["lr_l1"],
                            'BN Logistic Regression (H2PC-l2)': bnlearn_h2pc_nonlinear_dict_scores["lr_l2"],
                            'BN Logistic Regression (H2PC-elastic)': bnlearn_h2pc_nonlinear_dict_scores["lr_e"],
                            'POMEGRANATE Logistic Regression (Exact-none)': pomegranate_exact_nonlinear_dict_scores["lr"],
                            'POMEGRANATE Logistic Regression (Exact-l1)': pomegranate_exact_nonlinear_dict_scores["lr_l1"],
                            'POMEGRANATE Logistic Regression (Exact-l2)': pomegranate_exact_nonlinear_dict_scores["lr_l2"],
                            'POMEGRANATE Logistic Regression (Exact-elastic)': pomegranate_exact_nonlinear_dict_scores[
                                "lr_e"],
                            'POMEGRANATE Logistic Regression (Greedy-none)': pomegranate_greedy_nonlinear_dict_scores[
                                "lr"],
                            'POMEGRANATE Logistic Regression (Greedy-l1)': pomegranate_greedy_nonlinear_dict_scores[
                                "lr_l1"],
                            'POMEGRANATE Logistic Regression (Greedy-l2)': pomegranate_greedy_nonlinear_dict_scores[
                                "lr_l2"],
                            'POMEGRANATE Logistic Regression (Greedy-elastic)': pomegranate_greedy_nonlinear_dict_scores[
                                "lr_e"], 'PGMPY Logistic Regression (HC-none)': pgmpy_hc_nonlinear_dict_scores["lr"],
                            'PGMPY Logistic Regression (HC-l1)': pgmpy_hc_nonlinear_dict_scores["lr_l1"],
                            'PGMPY Logistic Regression (MMHC-l2)': pgmpy_mmhc_nonlinear_dict_scores["lr_l2"],
                            'PGMPY Logistic Regression (HC-elastic)': pgmpy_mmhc_nonlinear_dict_scores["lr_e"],
                            'PGMPY Logistic Regression (TREE-none)': pgmpy_tree_nonlinear_dict_scores["lr"],
                            'PGMPY Logistic Regression (TREE-l1)': pgmpy_tree_nonlinear_dict_scores["lr_l1"],
                            'PGMPY Logistic Regression (TREE-l2)': pgmpy_tree_nonlinear_dict_scores["lr_l2"],
                            'PGMPY Logistic Regression (TREE-elastic)': pgmpy_tree_nonlinear_dict_scores["lr_e"],
                            'PGMPY Logistic Regression (MMHC-none)': pgmpy_mmhc_nonlinear_dict_scores["lr"],
                            'PGMPY Logistic Regression (MMHC-l1)': pgmpy_mmhc_nonlinear_dict_scores["lr_l1"],
                            'PGMPY Logistic Regression (MMHC-l2)': pgmpy_mmhc_nonlinear_dict_scores["lr_l2"],
                            'PGMPY Logistic Regression (MMHC-elastic)': pgmpy_mmhc_nonlinear_dict_scores["lr_e"],
                            'NT Logistic Regression (Logistic-none)': notears_nonlinear_dict_scores["lr"],
                            'NT Logistic Regression (Logistic-l1)': notears_nonlinear_dict_scores["lr_l1"],
                            'NT Logistic Regression (Logistic-l2)': notears_nonlinear_dict_scores["lr_l2"],
                            'NT Logistic Regression (Logistic-elastic)': notears_nonlinear_dict_scores["lr_e"],
                            'NT Logistic Regression (L2-none)': notears_l2_nonlinear_dict_scores["lr"],
                            'NT Logistic Regression (L2-l1)': notears_l2_nonlinear_dict_scores["lr_l1"],
                            'NT Logistic Regression (L2-l2)': notears_l2_nonlinear_dict_scores["lr_l2"],
                            'NT Logistic Regression (L2-elastic)': notears_l2_nonlinear_dict_scores["lr_e"],
                            'NT Logistic Regression (Poisson-none)': notears_poisson_nonlinear_dict_scores["lr"],
                            'NT Logistic Regression (Poisson-l1)': notears_poisson_nonlinear_dict_scores["lr_l1"],
                            'NT Logistic Regression (Poisson-l2)': notears_poisson_nonlinear_dict_scores["lr_l2"],
                            'NT Logistic Regression (Poisson-elastic)': notears_poisson_nonlinear_dict_scores["lr_e"],
                            'BN Naive Bayes (HC-bernoulli)': bnlearn_nonlinear_dict_scores["nb"],
                            'BN Naive Bayes (HC-gaussian)': bnlearn_nonlinear_dict_scores["nb_g"],
                            'BN Naive Bayes (HC-multinomial)': bnlearn_nonlinear_dict_scores["nb_m"],
                            'BN Naive Bayes (HC-complement)': bnlearn_nonlinear_dict_scores["nb_c"],
                            'BN Naive Bayes (TABU-bernoulli)': bnlearn_tabu_nonlinear_dict_scores["nb"],
                            'BN Naive Bayes (TABU-gaussian)': bnlearn_tabu_nonlinear_dict_scores["nb_g"],
                            'BN Naive Bayes (TABU-multinomial)': bnlearn_tabu_nonlinear_dict_scores["nb_m"],
                            'BN Naive Bayes (TABU-complement)': bnlearn_tabu_nonlinear_dict_scores["nb_c"],
                            #'BN Naive Bayes (PC-bernoulli)': bnlearn_pc_nonlinear_dict_scores["nb"],
                            #'BN Naive Bayes (PC-gaussian)': bnlearn_pc_nonlinear_dict_scores["nb_g"],
                            #'BN Naive Bayes (PC-multinomial)': bnlearn_pc_nonlinear_dict_scores["nb_m"],
                            #'BN Naive Bayes (PC-complement)': bnlearn_pc_nonlinear_dict_scores["nb_c"],
                            'BN Naive Bayes (MMHC-bernoulli)': bnlearn_mmhc_nonlinear_dict_scores["nb"],
                            'BN Naive Bayes (MMHC-gaussian)': bnlearn_mmhc_nonlinear_dict_scores["nb_g"],
                            'BN Naive Bayes (MMHC-multinomial)': bnlearn_mmhc_nonlinear_dict_scores["nb_m"],
                            'BN Naive Bayes (MMHC-complement)': bnlearn_mmhc_nonlinear_dict_scores["nb_c"],
                            'BN Naive Bayes (RSMAX2-bernoulli)': bnlearn_rsmax2_nonlinear_dict_scores["nb"],
                            'BN Naive Bayes (RSMAX2-gaussian)': bnlearn_rsmax2_nonlinear_dict_scores["nb_g"],
                            'BN Naive Bayes (RSMAX2-multinomial)': bnlearn_rsmax2_nonlinear_dict_scores["nb_m"],
                            'BN Naive Bayes (RSMAX2-complement)': bnlearn_rsmax2_nonlinear_dict_scores["nb_c"],
                            'BN Naive Bayes (H2PC-bernoulli)': bnlearn_h2pc_nonlinear_dict_scores["nb"],
                            'BN Naive Bayes (H2PC-gaussian)': bnlearn_h2pc_nonlinear_dict_scores["nb_g"],
                            'BN Naive Bayes (H2PC-multinomial)': bnlearn_h2pc_nonlinear_dict_scores["nb_m"],
                            'BN Naive Bayes (H2PC-complement)': bnlearn_h2pc_nonlinear_dict_scores["nb_c"],
                            'NT Naive Bayes (Logistic-bernoulli)': notears_nonlinear_dict_scores["nb"],
                            'NT Naive Bayes (Logistic-gaussian)': notears_nonlinear_dict_scores["nb_g"],
                            'NT Naive Bayes (Logistic-multinomial)': notears_nonlinear_dict_scores["nb_m"],
                            'NT Naive Bayes (Logistic-complement)': notears_nonlinear_dict_scores["nb_c"],
                            'NT Naive Bayes (L2-bernoulli)': notears_l2_nonlinear_dict_scores["nb"],
                            'NT Naive Bayes (L2-gaussian)': notears_l2_nonlinear_dict_scores["nb_g"],
                            'NT Naive Bayes (L2-multinomial)': notears_l2_nonlinear_dict_scores["nb_m"],
                            'NT Naive Bayes (L2-complement)': notears_l2_nonlinear_dict_scores["nb_c"],
                            'NT Naive Bayes (Poisson-bernoulli)': notears_poisson_nonlinear_dict_scores["nb"],
                            'NT Naive Bayes (Poisson-gaussian)': notears_poisson_nonlinear_dict_scores["nb_g"],
                            'NT Naive Bayes (Poisson-multinomial)': notears_poisson_nonlinear_dict_scores["nb_m"],
                            'NT Naive Bayes (Poisson-complement)': notears_poisson_nonlinear_dict_scores["nb_c"],
                            'POMEGRANATE Naive Bayes (Greedy-bernoulli)': pomegranate_greedy_nonlinear_dict_scores["nb"],
                            'POMEGRANATE Naive Bayes (Greedy-gaussian)': pomegranate_greedy_nonlinear_dict_scores["nb_g"],
                            'POMEGRANATE Naive Bayes (Greedy-multinomial)': pomegranate_greedy_nonlinear_dict_scores[
                                "nb_m"],
                            'POMEGRANATE Naive Bayes (Greedy-complement)': pomegranate_greedy_nonlinear_dict_scores[
                                "nb_c"],
                            'POMEGRANATE Naive Bayes (Exact-bernoulli)': pomegranate_exact_nonlinear_dict_scores["nb"],
                            'POMEGRANATE Naive Bayes (Exact-gaussian)': pomegranate_exact_nonlinear_dict_scores["nb_g"],
                            'POMEGRANATE Naive Bayes (Exact-multinomial)': pomegranate_exact_nonlinear_dict_scores["nb_m"],
                            'POMEGRANATE Naive Bayes (Exact-complement)': pomegranate_exact_nonlinear_dict_scores["nb_c"],
                            'PGMPY Naive Bayes (HC-bernoulli)': pgmpy_hc_nonlinear_dict_scores["nb"],
                            'PGMPY Naive Bayes (HC-gaussian)': pgmpy_hc_nonlinear_dict_scores["nb_g"],
                            'PGMPY Naive Bayes (HC-multinomial)': pgmpy_hc_nonlinear_dict_scores["nb_m"],
                            'PGMPY Naive Bayes (HC-complement)': pgmpy_hc_nonlinear_dict_scores["nb_c"],
                            'PGMPY Naive Bayes (MMHC-bernoulli)': pgmpy_mmhc_nonlinear_dict_scores["nb"],
                            'PGMPY Naive Bayes (MMHC-gaussian)': pgmpy_mmhc_nonlinear_dict_scores["nb_g"],
                            'PGMPY Naive Bayes (MMHC-multinomial)': pgmpy_mmhc_nonlinear_dict_scores["nb_m"],
                            'PGMPY Naive Bayes (MMHC-complement)': pgmpy_mmhc_nonlinear_dict_scores["nb_c"],
                            'PGMPY Naive Bayes (TREE-bernoulli)': pgmpy_tree_nonlinear_dict_scores["nb"],
                            'PGMPY Naive Bayes (TREE-gaussian)': pgmpy_tree_nonlinear_dict_scores["nb_g"],
                            'PGMPY Naive Bayes (TREE-multinomial)': pgmpy_tree_nonlinear_dict_scores["nb_m"],
                            'PGMPY Naive Bayes (TREE-complement)': pgmpy_tree_nonlinear_dict_scores["nb_c"],
                            'BN Support Vector Machine (HC-sigmoid)': bnlearn_nonlinear_dict_scores["svm"],
                            'BN Support Vector Machine (HC-polynomial)': bnlearn_nonlinear_dict_scores["svm_po"],
                            'BN Support Vector Machine (HC-rbf)': bnlearn_nonlinear_dict_scores["svm_r"],
                            'BN Support Vector Machine (TABU-sigmoid)': bnlearn_tabu_nonlinear_dict_scores["svm"],
                            'BN Support Vector Machine (TABU-polynomial)': bnlearn_tabu_nonlinear_dict_scores["svm_po"],
                            'BN Support Vector Machine (TABU-rbf)': bnlearn_tabu_nonlinear_dict_scores["svm_r"],
                            #'BN Support Vector Machine (PC-sigmoid)': bnlearn_pc_nonlinear_dict_scores["svm"],
                            #'BN Support Vector Machine (PC-polynomial)': bnlearn_pc_nonlinear_dict_scores["svm_po"],
                            #'BN Support Vector Machine (PC-rbf)': bnlearn_pc_nonlinear_dict_scores["svm_r"],
                            'BN Support Vector Machine (MMHC-sigmoid)': bnlearn_mmhc_nonlinear_dict_scores["svm"],
                            'BN Support Vector Machine (MMHC-polynomial)': bnlearn_mmhc_nonlinear_dict_scores["svm_po"],
                            'BN Support Vector Machine (MMHC-rbf)': bnlearn_mmhc_nonlinear_dict_scores["svm_r"],
                            'BN Support Vector Machine (RSMAX2-sigmoid)': bnlearn_rsmax2_nonlinear_dict_scores["svm"],
                            'BN Support Vector Machine (RSMAX2-polynomial)': bnlearn_rsmax2_nonlinear_dict_scores[
                                "svm_po"],
                            'BN Support Vector Machine (RSMAX2-rbf)': bnlearn_rsmax2_nonlinear_dict_scores["svm_r"],
                            'BN Support Vector Machine (H2PC-sigmoid)': bnlearn_h2pc_nonlinear_dict_scores["svm"],
                            'BN Support Vector Machine (H2PC-polynomial)': bnlearn_h2pc_nonlinear_dict_scores["svm_po"],
                            'BN Support Vector Machine (H2PC-rbf)': bnlearn_h2pc_nonlinear_dict_scores["svm_r"],
                            'NT Support Vector Machine (logistic-sigmoid)': notears_nonlinear_dict_scores["svm"],
                            'NT Support Vector Machine (logistic-polynomial)': notears_nonlinear_dict_scores["svm_po"],
                            'NT Support Vector Machine (logistic-rbf)': notears_nonlinear_dict_scores["svm_r"],
                            'NT Support Vector Machine (L2-sigmoid)': notears_l2_nonlinear_dict_scores["svm"],
                            'NT Support Vector Machine (L2-polynomial)': notears_l2_nonlinear_dict_scores["svm_po"],
                            'NT Support Vector Machine (L2-rbf)': notears_l2_nonlinear_dict_scores["svm_r"],
                            'NT Support Vector Machine (Poisson-sigmoid)': notears_poisson_nonlinear_dict_scores["svm"],
                            'NT Support Vector Machine (Poisson-polynomial)': notears_poisson_nonlinear_dict_scores[
                                "svm_po"],
                            'NT Support Vector Machine (Poisson-rbf)': notears_poisson_nonlinear_dict_scores["svm_r"],
                            'Pomegranate Support Vector Machine (Exact-sigmoid)': pomegranate_exact_nonlinear_dict_scores[
                                "svm"], 'Pomegranate Support Vector Machine (Exact-polynomial)':
                                pomegranate_exact_nonlinear_dict_scores["svm_po"],
                            'Pomegranate Support Vector Machine (Exact-rbf)': pomegranate_exact_nonlinear_dict_scores[
                                "svm_r"], 'Pomegranate Support Vector Machine (Greedy-sigmoid)':
                                pomegranate_greedy_nonlinear_dict_scores["svm"],
                            'Pomegranate Support Vector Machine (Greedy-polynomial)':
                                pomegranate_greedy_nonlinear_dict_scores["svm_po"],
                            'Pomegranate Support Vector Machine (Greedy-rbf)': pomegranate_greedy_nonlinear_dict_scores[
                                "svm_r"],
                            'PGMPY Support Vector Machine (HC-sigmoid)': pgmpy_hc_nonlinear_dict_scores["svm"],
                            'PGMPY Support Vector Machine (HC-polynomial)': pgmpy_hc_nonlinear_dict_scores["svm_po"],
                            'PGMPY Support Vector Machine (HC-rbf)': pgmpy_hc_nonlinear_dict_scores["svm_r"],
                            'PGMPY Support Vector Machine (MMHC-sigmoid)': pgmpy_mmhc_nonlinear_dict_scores["svm"],
                            'PGMPY Support Vector Machine (MMHC-polynomial)': pgmpy_mmhc_nonlinear_dict_scores["svm_po"],
                            'PGMPY Support Vector Machine (MMHC-rbf)': pgmpy_mmhc_nonlinear_dict_scores["svm_r"],
                            'PGMPY Support Vector Machine (TREE-sigmoid)': pgmpy_tree_nonlinear_dict_scores["svm"],
                            'PGMPY Support Vector Machine (TREE-polynomial)': pgmpy_tree_nonlinear_dict_scores["svm_po"],
                            'PGMPY Support Vector Machine (TREE-rbf)': pgmpy_tree_nonlinear_dict_scores["svm_r"],
                            'BN K Nearest Neighbor (HC-weight)': bnlearn_nonlinear_dict_scores["knn"],
                            'BN K Nearest Neighbor (HC-distance)': bnlearn_nonlinear_dict_scores["knn_d"],
                            'BN K Nearest Neighbor (TABU-weight)': bnlearn_tabu_nonlinear_dict_scores["knn"],
                            'BN K Nearest Neighbor (TABU-distance)': bnlearn_tabu_nonlinear_dict_scores["knn_d"],
                            #'BN K Nearest Neighbor (PC-weight)': bnlearn_pc_nonlinear_dict_scores["knn"],
                            #'BN K Nearest Neighbor (PC-distance)': bnlearn_pc_nonlinear_dict_scores["knn_d"],
                            'BN K Nearest Neighbor (MMHC-weight)': bnlearn_mmhc_nonlinear_dict_scores["knn"],
                            'BN K Nearest Neighbor (MMHC-distance)': bnlearn_mmhc_nonlinear_dict_scores["knn_d"],
                            'BN K Nearest Neighbor (RSMAX2-weight)': bnlearn_rsmax2_nonlinear_dict_scores["knn"],
                            'BN K Nearest Neighbor (RSMAX2-distance)': bnlearn_rsmax2_nonlinear_dict_scores["knn_d"],
                            'BN K Nearest Neighbor (H2PC-weight)': bnlearn_h2pc_nonlinear_dict_scores["knn"],
                            'BN K Nearest Neighbor (H2PC-distance)': bnlearn_h2pc_nonlinear_dict_scores["knn_d"],
                            'NT K Nearest Neighbor (Logistic-weight)': notears_nonlinear_dict_scores["knn"],
                            'NT K Nearest Neighbor (Logistic-distance)': notears_nonlinear_dict_scores["knn_d"],
                            'NT K Nearest Neighbor (L2-weight)': notears_l2_nonlinear_dict_scores["knn"],
                            'NT K Nearest Neighbor (L2-distance)': notears_l2_nonlinear_dict_scores["knn_d"],
                            'NT K Nearest Neighbor (Poisson-weight)': notears_poisson_nonlinear_dict_scores["knn"],
                            'NT K Nearest Neighbor (Poisson-distance)': notears_poisson_nonlinear_dict_scores["knn_d"],
                            'POMEGRANATE K Nearest Neighbor (Exact-weight)': pomegranate_exact_nonlinear_dict_scores[
                                "knn"],
                            'POMEGRANATE K Nearest Neighbor (Exact-distance)': pomegranate_exact_nonlinear_dict_scores[
                                "knn_d"],
                            'POMEGRANATE K Nearest Neighbor (Greedy-weight)': pomegranate_greedy_nonlinear_dict_scores[
                                "knn"],
                            'POMEGRANATE K Nearest Neighbor (Greedy-distance)': pomegranate_greedy_nonlinear_dict_scores[
                                "knn_d"], 'PGMPY K Nearest Neighbor (HC-weight)': pgmpy_hc_nonlinear_dict_scores["knn"],
                            'PGMPY K Nearest Neighbor (HC-distance)': pgmpy_hc_nonlinear_dict_scores["knn_d"],
                            'PGMPY K Nearest Neighbor (MMHC-weight)': pgmpy_mmhc_nonlinear_dict_scores["knn"],
                            'PGMPY K Nearest Neighbor (MMHC-distance)': pgmpy_mmhc_nonlinear_dict_scores["knn_d"],
                            'PGMPY K Nearest Neighbor (TREE-weight)': pgmpy_tree_nonlinear_dict_scores["knn"],
                            'PGMPY K Nearest Neighbor (TREE-distance)': pgmpy_tree_nonlinear_dict_scores["knn_d"]}

    top_learned_nonlinear = max(sim_nonlinear_workflows, key=sim_nonlinear_workflows.get)
    print("Learned world - Nonlinear problem, Prediction: "+ top_learned_nonlinear + " (" + str(sim_nonlinear_workflows[top_learned_nonlinear]) + ")")

    real_sparse_workflows = {'Decision Tree (gini)': real_sparse_dt_scores,
                             'Decision Tree (entropy)': real_sparse_dt_entropy_scores,
                             'Random Forest (gini)': real_sparse_rf_scores,
                             'Random Forest (entropy)': real_sparse_rf_entropy_scores,
                             'Logistic Regression (none)': real_sparse_lr_scores,
                             'Logistic Regression (l1)': real_sparse_lr_l1_scores,
                             'Logistic Regression (l2)': real_sparse_lr_l2_scores,
                             'Logistic Regression (elasticnet)': real_sparse_lr_elastic_scores,
                             'Naive Bayes (bernoulli)': real_sparse_gb_scores,
                             'Naive Bayes (multinomial)': real_sparse_gb_multi_scores,
                             'Naive Bayes (gaussian)': real_sparse_gb_gaussian_scores,
                             'Naive Bayes (complement)': real_sparse_gb_complement_scores,
                             'Support Vector Machine (sigmoid)': real_sparse_svm_scores,
                             'Support Vector Machine (polynomial)': real_sparse_svm_poly_scores,
                             'Support Vector Machine (rbf)': real_sparse_svm_rbf_scores,
                             'K Nearest Neighbor (uniform)': real_sparse_knn_scores,
                             'K Nearest Neighbor (distance)': real_sparse_knn_distance_scores}
    top_real_sparse = max(real_sparse_workflows, key=real_sparse_workflows.get)
    print("Real world - Sparse problem, Prediction: "+ top_real_sparse + " (" + str(real_sparse_workflows[top_real_sparse]) + ")")

    sim_sparse_workflows = {'BN Decision Tree (HC-gini)': bnlearn_sparse_dict_scores["dt"],
                            'BN Decision Tree (HC-entropy)': bnlearn_sparse_dict_scores["dt_e"],
                            'BN Decision Tree (TABU-gini)': bnlearn_tabu_sparse_dict_scores["dt"],
                            'BN Decision Tree (TABU-entropy)': bnlearn_tabu_sparse_dict_scores["dt_e"],
                            #'BN Decision Tree (PC-gini)': bnlearn_pc_sparse_dict_scores["dt"],
                            #'BN Decision Tree (PC-entropy)': bnlearn_pc_sparse_dict_scores["dt_e"],
                            'BN Decision Tree (MMHC-gini)': bnlearn_mmhc_sparse_dict_scores["dt"],
                            'BN Decision Tree (MMHC-entropy)': bnlearn_mmhc_sparse_dict_scores["dt_e"],
                            'BN Decision Tree (RSMAX2-gini)': bnlearn_rsmax2_sparse_dict_scores["dt"],
                            'BN Decision Tree (RSMAX2-entropy)': bnlearn_rsmax2_sparse_dict_scores["dt_e"],
                            'BN Decision Tree (H2PC-gini)': bnlearn_h2pc_sparse_dict_scores["dt"],
                            'BN Decision Tree (H2PC-entropy)': bnlearn_h2pc_sparse_dict_scores["dt_e"],
                            'NT Decision Tree (Logistic-gini)': notears_sparse_dict_scores["dt"],
                            'NT Decision Tree (Logistic-entropy)': notears_sparse_dict_scores["dt_e"],
                            'NT Decision Tree (L2-gini)': notears_l2_sparse_dict_scores["dt"],
                            'NT Decision Tree (L2-entropy)': notears_l2_sparse_dict_scores["dt_e"],
                            'NT Decision Tree (Poisson-gini)': notears_poisson_sparse_dict_scores["dt"],
                            'NT Decision Tree (Poisson-entropy)': notears_poisson_sparse_dict_scores["dt_e"],
                            'POMEGRANATE Decision Tree (Exact-gini)': pomegranate_exact_sparse_dict_scores["dt"],
                            'POMEGRANATE Decision Tree (Exact-entropy)': pomegranate_exact_sparse_dict_scores["dt_e"],
                            'POMEGRANATE Decision Tree (Greedy-gini)': pomegranate_greedy_sparse_dict_scores["dt"],
                            'POMEGRANATE Decision Tree (Greedy-entropy)': pomegranate_greedy_sparse_dict_scores["dt_e"],
                            'PGMPY Decision Tree (HC-gini)': pgmpy_hc_sparse_dict_scores["dt"],
                            'PGMPY Decision Tree (HC-entropy)': pgmpy_hc_sparse_dict_scores["dt_e"],
                            'PGMPY Decision Tree (MMHC-gini)': pgmpy_mmhc_sparse_dict_scores["dt"],
                            'PGMPY Decision Tree (HC-entropy)': pgmpy_mmhc_sparse_dict_scores["dt_e"],
                            'PGMPY Decision Tree (TREE-gini)': pgmpy_tree_sparse_dict_scores["dt"],
                            'PGMPY Decision Tree (TREE-entropy)': pgmpy_tree_sparse_dict_scores["dt_e"],
                            'BN Random Forest (HC-gini)': bnlearn_sparse_dict_scores["rf"],
                            'BN Random Forest (HC-entropy)': bnlearn_sparse_dict_scores["rf_e"],
                            'BN Random Forest (TABU-gini)': bnlearn_tabu_sparse_dict_scores["rf"],
                            'BN Random Forest (TABU-entropy)': bnlearn_tabu_sparse_dict_scores["rf_e"],
                            #'BN Random Forest (PC-gini)': bnlearn_pc_sparse_dict_scores["rf"],
                            #'BN Random Forest (PC-entropy)': bnlearn_pc_sparse_dict_scores["rf_e"],
                            'BN Random Forest (MMHC-gini)': bnlearn_mmhc_sparse_dict_scores["rf"],
                            'BN Random Forest (MMHC-entropy)': bnlearn_mmhc_sparse_dict_scores["rf_e"],
                            'BN Random Forest (RSMAX2-gini)': bnlearn_rsmax2_sparse_dict_scores["rf"],
                            'BN Random Forest (RSMAX2-entropy)': bnlearn_rsmax2_sparse_dict_scores["rf_e"],
                            'BN Random Forest (H2PC-gini)': bnlearn_h2pc_sparse_dict_scores["rf"],
                            'BN Random Forest (H2PC-entropy)': bnlearn_h2pc_sparse_dict_scores["rf_e"],
                            'NT Random Forest (Logistic-gini)': notears_sparse_dict_scores["rf"],
                            'NT Random Forest (Logistic-entropy)': notears_sparse_dict_scores["rf_e"],
                            'NT Random Forest (L2-gini)': notears_l2_sparse_dict_scores["rf"],
                            'NT Random Forest (l2-entropy)': notears_l2_sparse_dict_scores["rf_e"],
                            'NT Random Forest (Poisson-gini)': notears_poisson_sparse_dict_scores["rf"],
                            'NT Random Forest (Poisson-entropy)': notears_poisson_sparse_dict_scores["rf_e"],
                            'POMEGRANATE Random Forest (Exact-gini)': pomegranate_exact_sparse_dict_scores["rf"],
                            'POMEGRANATE Random Forest (Exact-entropy)': pomegranate_exact_sparse_dict_scores["rf_e"],
                            'POMEGRANATE Random Forest (Greedy-gini)': pomegranate_greedy_sparse_dict_scores["rf"],
                            'POMEGRANATE Random Forest (Greedy-entropy)': pomegranate_greedy_sparse_dict_scores["rf_e"],
                            'PGMPY Random Forest (HC-gini)': pgmpy_hc_sparse_dict_scores["rf"],
                            'PGMPY Random Forest (HC-entropy)': pgmpy_hc_sparse_dict_scores["rf_e"],
                            'PGMPY Random Forest (MMHC-gini)': pgmpy_mmhc_sparse_dict_scores["rf"],
                            'PGMPY Random Forest (HC-entropy)': pgmpy_mmhc_sparse_dict_scores["rf_e"],
                            'PGMPY Random Forest (TREE-gini)': pgmpy_tree_sparse_dict_scores["rf"],
                            'PGMPY Random Forest (TREE-entropy)': pgmpy_tree_sparse_dict_scores["rf_e"],
                            'BN Logistic Regression (HC-none)': bnlearn_sparse_dict_scores["lr"],
                            'BN Logistic Regression (HC-l1)': bnlearn_sparse_dict_scores["lr_l1"],
                            'BN Logistic Regression (HC-l2)': bnlearn_sparse_dict_scores["lr_l2"],
                            'BN Logistic Regression (HC-elastic)': bnlearn_sparse_dict_scores["lr_e"],
                            'BN Logistic Regression (TABU-none)': bnlearn_tabu_sparse_dict_scores["lr"],
                            'BN Logistic Regression (TABU-l1)': bnlearn_tabu_sparse_dict_scores["lr_l1"],
                            'BN Logistic Regression (TABU-l2)': bnlearn_tabu_sparse_dict_scores["lr_l2"],
                            'BN Logistic Regression (TABU-elastic)': bnlearn_tabu_sparse_dict_scores["lr_e"],
                            #'BN Logistic Regression (PC-none)': bnlearn_pc_sparse_dict_scores["lr"],
                            #'BN Logistic Regression (PC-l1)': bnlearn_pc_sparse_dict_scores["lr_l1"],
                            #'BN Logistic Regression (PC-l2)': bnlearn_pc_sparse_dict_scores["lr_l2"],
                            #'BN Logistic Regression (PC-elastic)': bnlearn_pc_sparse_dict_scores["lr_e"],
                            'BN Logistic Regression (MMHC-none)': bnlearn_mmhc_sparse_dict_scores["lr"],
                            'BN Logistic Regression (MMHC-l1)': bnlearn_mmhc_sparse_dict_scores["lr_l1"],
                            'BN Logistic Regression (MMHC-l2)': bnlearn_mmhc_sparse_dict_scores["lr_l2"],
                            'BN Logistic Regression (MMHC-elastic)': bnlearn_mmhc_sparse_dict_scores["lr_e"],
                            'BN Logistic Regression (RSMAX2-none)': bnlearn_rsmax2_sparse_dict_scores["lr"],
                            'BN Logistic Regression (RSMAX2-l1)': bnlearn_rsmax2_sparse_dict_scores["lr_l1"],
                            'BN Logistic Regression (RSMAX2-l2)': bnlearn_rsmax2_sparse_dict_scores["lr_l2"],
                            'BN Logistic Regression (RSMAX2-elastic)': bnlearn_rsmax2_sparse_dict_scores["lr_e"],
                            'BN Logistic Regression (H2PC-none)': bnlearn_h2pc_sparse_dict_scores["lr"],
                            'BN Logistic Regression (H2PC-l1)': bnlearn_h2pc_sparse_dict_scores["lr_l1"],
                            'BN Logistic Regression (H2PC-l2)': bnlearn_h2pc_sparse_dict_scores["lr_l2"],
                            'BN Logistic Regression (H2PC-elastic)': bnlearn_h2pc_sparse_dict_scores["lr_e"],
                            'POMEGRANATE Logistic Regression (Exact-none)': pomegranate_exact_sparse_dict_scores["lr"],
                            'POMEGRANATE Logistic Regression (Exact-l1)': pomegranate_exact_sparse_dict_scores["lr_l1"],
                            'POMEGRANATE Logistic Regression (Exact-l2)': pomegranate_exact_sparse_dict_scores["lr_l2"],
                            'POMEGRANATE Logistic Regression (Exact-elastic)': pomegranate_exact_sparse_dict_scores[
                                "lr_e"],
                            'POMEGRANATE Logistic Regression (Greedy-none)': pomegranate_greedy_sparse_dict_scores[
                                "lr"],
                            'POMEGRANATE Logistic Regression (Greedy-l1)': pomegranate_greedy_sparse_dict_scores[
                                "lr_l1"],
                            'POMEGRANATE Logistic Regression (Greedy-l2)': pomegranate_greedy_sparse_dict_scores[
                                "lr_l2"],
                            'POMEGRANATE Logistic Regression (Greedy-elastic)': pomegranate_greedy_sparse_dict_scores[
                                "lr_e"], 'PGMPY Logistic Regression (HC-none)': pgmpy_hc_sparse_dict_scores["lr"],
                            'PGMPY Logistic Regression (HC-l1)': pgmpy_hc_sparse_dict_scores["lr_l1"],
                            'PGMPY Logistic Regression (MMHC-l2)': pgmpy_mmhc_sparse_dict_scores["lr_l2"],
                            'PGMPY Logistic Regression (HC-elastic)': pgmpy_mmhc_sparse_dict_scores["lr_e"],
                            'PGMPY Logistic Regression (TREE-none)': pgmpy_tree_sparse_dict_scores["lr"],
                            'PGMPY Logistic Regression (TREE-l1)': pgmpy_tree_sparse_dict_scores["lr_l1"],
                            'PGMPY Logistic Regression (TREE-l2)': pgmpy_tree_sparse_dict_scores["lr_l2"],
                            'PGMPY Logistic Regression (TREE-elastic)': pgmpy_tree_sparse_dict_scores["lr_e"],
                            'PGMPY Logistic Regression (MMHC-none)': pgmpy_mmhc_sparse_dict_scores["lr"],
                            'PGMPY Logistic Regression (MMHC-l1)': pgmpy_mmhc_sparse_dict_scores["lr_l1"],
                            'PGMPY Logistic Regression (MMHC-l2)': pgmpy_mmhc_sparse_dict_scores["lr_l2"],
                            'PGMPY Logistic Regression (MMHC-elastic)': pgmpy_mmhc_sparse_dict_scores["lr_e"],
                            'NT Logistic Regression (Logistic-none)': notears_sparse_dict_scores["lr"],
                            'NT Logistic Regression (Logistic-l1)': notears_sparse_dict_scores["lr_l1"],
                            'NT Logistic Regression (Logistic-l2)': notears_sparse_dict_scores["lr_l2"],
                            'NT Logistic Regression (Logistic-elastic)': notears_sparse_dict_scores["lr_e"],
                            'NT Logistic Regression (L2-none)': notears_l2_sparse_dict_scores["lr"],
                            'NT Logistic Regression (L2-l1)': notears_l2_sparse_dict_scores["lr_l1"],
                            'NT Logistic Regression (L2-l2)': notears_l2_sparse_dict_scores["lr_l2"],
                            'NT Logistic Regression (L2-elastic)': notears_l2_sparse_dict_scores["lr_e"],
                            'NT Logistic Regression (Poisson-none)': notears_poisson_sparse_dict_scores["lr"],
                            'NT Logistic Regression (Poisson-l1)': notears_poisson_sparse_dict_scores["lr_l1"],
                            'NT Logistic Regression (Poisson-l2)': notears_poisson_sparse_dict_scores["lr_l2"],
                            'NT Logistic Regression (Poisson-elastic)': notears_poisson_sparse_dict_scores["lr_e"],
                            'BN Naive Bayes (HC-bernoulli)': bnlearn_sparse_dict_scores["nb"],
                            'BN Naive Bayes (HC-gaussian)': bnlearn_sparse_dict_scores["nb_g"],
                            'BN Naive Bayes (HC-multinomial)': bnlearn_sparse_dict_scores["nb_m"],
                            'BN Naive Bayes (HC-complement)': bnlearn_sparse_dict_scores["nb_c"],
                            'BN Naive Bayes (TABU-bernoulli)': bnlearn_tabu_sparse_dict_scores["nb"],
                            'BN Naive Bayes (TABU-gaussian)': bnlearn_tabu_sparse_dict_scores["nb_g"],
                            'BN Naive Bayes (TABU-multinomial)': bnlearn_tabu_sparse_dict_scores["nb_m"],
                            'BN Naive Bayes (TABU-complement)': bnlearn_tabu_sparse_dict_scores["nb_c"],
                            #'BN Naive Bayes (PC-bernoulli)': bnlearn_pc_sparse_dict_scores["nb"],
                            #'BN Naive Bayes (PC-gaussian)': bnlearn_pc_sparse_dict_scores["nb_g"],
                            #'BN Naive Bayes (PC-multinomial)': bnlearn_pc_sparse_dict_scores["nb_m"],
                            #'BN Naive Bayes (PC-complement)': bnlearn_pc_sparse_dict_scores["nb_c"],
                            'BN Naive Bayes (MMHC-bernoulli)': bnlearn_mmhc_sparse_dict_scores["nb"],
                            'BN Naive Bayes (MMHC-gaussian)': bnlearn_mmhc_sparse_dict_scores["nb_g"],
                            'BN Naive Bayes (MMHC-multinomial)': bnlearn_mmhc_sparse_dict_scores["nb_m"],
                            'BN Naive Bayes (MMHC-complement)': bnlearn_mmhc_sparse_dict_scores["nb_c"],
                            'BN Naive Bayes (RSMAX2-bernoulli)': bnlearn_rsmax2_sparse_dict_scores["nb"],
                            'BN Naive Bayes (RSMAX2-gaussian)': bnlearn_rsmax2_sparse_dict_scores["nb_g"],
                            'BN Naive Bayes (RSMAX2-multinomial)': bnlearn_rsmax2_sparse_dict_scores["nb_m"],
                            'BN Naive Bayes (RSMAX2-complement)': bnlearn_rsmax2_sparse_dict_scores["nb_c"],
                            'BN Naive Bayes (H2PC-bernoulli)': bnlearn_h2pc_sparse_dict_scores["nb"],
                            'BN Naive Bayes (H2PC-gaussian)': bnlearn_h2pc_sparse_dict_scores["nb_g"],
                            'BN Naive Bayes (H2PC-multinomial)': bnlearn_h2pc_sparse_dict_scores["nb_m"],
                            'BN Naive Bayes (H2PC-complement)': bnlearn_h2pc_sparse_dict_scores["nb_c"],
                            'NT Naive Bayes (Logistic-bernoulli)': notears_sparse_dict_scores["nb"],
                            'NT Naive Bayes (Logistic-gaussian)': notears_sparse_dict_scores["nb_g"],
                            'NT Naive Bayes (Logistic-multinomial)': notears_sparse_dict_scores["nb_m"],
                            'NT Naive Bayes (Logistic-complement)': notears_sparse_dict_scores["nb_c"],
                            'NT Naive Bayes (L2-bernoulli)': notears_l2_sparse_dict_scores["nb"],
                            'NT Naive Bayes (L2-gaussian)': notears_l2_sparse_dict_scores["nb_g"],
                            'NT Naive Bayes (L2-multinomial)': notears_l2_sparse_dict_scores["nb_m"],
                            'NT Naive Bayes (L2-complement)': notears_l2_sparse_dict_scores["nb_c"],
                            'NT Naive Bayes (Poisson-bernoulli)': notears_poisson_sparse_dict_scores["nb"],
                            'NT Naive Bayes (Poisson-gaussian)': notears_poisson_sparse_dict_scores["nb_g"],
                            'NT Naive Bayes (Poisson-multinomial)': notears_poisson_sparse_dict_scores["nb_m"],
                            'NT Naive Bayes (Poisson-complement)': notears_poisson_sparse_dict_scores["nb_c"],
                            'POMEGRANATE Naive Bayes (Greedy-bernoulli)': pomegranate_greedy_sparse_dict_scores["nb"],
                            'POMEGRANATE Naive Bayes (Greedy-gaussian)': pomegranate_greedy_sparse_dict_scores["nb_g"],
                            'POMEGRANATE Naive Bayes (Greedy-multinomial)': pomegranate_greedy_sparse_dict_scores[
                                "nb_m"],
                            'POMEGRANATE Naive Bayes (Greedy-complement)': pomegranate_greedy_sparse_dict_scores[
                                "nb_c"],
                            'POMEGRANATE Naive Bayes (Exact-bernoulli)': pomegranate_exact_sparse_dict_scores["nb"],
                            'POMEGRANATE Naive Bayes (Exact-gaussian)': pomegranate_exact_sparse_dict_scores["nb_g"],
                            'POMEGRANATE Naive Bayes (Exact-multinomial)': pomegranate_exact_sparse_dict_scores["nb_m"],
                            'POMEGRANATE Naive Bayes (Exact-complement)': pomegranate_exact_sparse_dict_scores["nb_c"],
                            'PGMPY Naive Bayes (HC-bernoulli)': pgmpy_hc_sparse_dict_scores["nb"],
                            'PGMPY Naive Bayes (HC-gaussian)': pgmpy_hc_sparse_dict_scores["nb_g"],
                            'PGMPY Naive Bayes (HC-multinomial)': pgmpy_hc_sparse_dict_scores["nb_m"],
                            'PGMPY Naive Bayes (HC-complement)': pgmpy_hc_sparse_dict_scores["nb_c"],
                            'PGMPY Naive Bayes (MMHC-bernoulli)': pgmpy_mmhc_sparse_dict_scores["nb"],
                            'PGMPY Naive Bayes (MMHC-gaussian)': pgmpy_mmhc_sparse_dict_scores["nb_g"],
                            'PGMPY Naive Bayes (MMHC-multinomial)': pgmpy_mmhc_sparse_dict_scores["nb_m"],
                            'PGMPY Naive Bayes (MMHC-complement)': pgmpy_mmhc_sparse_dict_scores["nb_c"],
                            'PGMPY Naive Bayes (TREE-bernoulli)': pgmpy_tree_sparse_dict_scores["nb"],
                            'PGMPY Naive Bayes (TREE-gaussian)': pgmpy_tree_sparse_dict_scores["nb_g"],
                            'PGMPY Naive Bayes (TREE-multinomial)': pgmpy_tree_sparse_dict_scores["nb_m"],
                            'PGMPY Naive Bayes (TREE-complement)': pgmpy_tree_sparse_dict_scores["nb_c"],
                            'BN Support Vector Machine (HC-sigmoid)': bnlearn_sparse_dict_scores["svm"],
                            'BN Support Vector Machine (HC-polynomial)': bnlearn_sparse_dict_scores["svm_po"],
                            'BN Support Vector Machine (HC-rbf)': bnlearn_sparse_dict_scores["svm_r"],
                            'BN Support Vector Machine (TABU-sigmoid)': bnlearn_tabu_sparse_dict_scores["svm"],
                            'BN Support Vector Machine (TABU-polynomial)': bnlearn_tabu_sparse_dict_scores["svm_po"],
                            'BN Support Vector Machine (TABU-rbf)': bnlearn_tabu_sparse_dict_scores["svm_r"],
                            #'BN Support Vector Machine (PC-sigmoid)': bnlearn_pc_sparse_dict_scores["svm"],
                            #'BN Support Vector Machine (PC-polynomial)': bnlearn_pc_sparse_dict_scores["svm_po"],
                            #'BN Support Vector Machine (PC-rbf)': bnlearn_pc_sparse_dict_scores["svm_r"],
                            'BN Support Vector Machine (MMHC-sigmoid)': bnlearn_mmhc_sparse_dict_scores["svm"],
                            'BN Support Vector Machine (MMHC-polynomial)': bnlearn_mmhc_sparse_dict_scores["svm_po"],
                            'BN Support Vector Machine (MMHC-rbf)': bnlearn_mmhc_sparse_dict_scores["svm_r"],
                            'BN Support Vector Machine (RSMAX2-sigmoid)': bnlearn_rsmax2_sparse_dict_scores["svm"],
                            'BN Support Vector Machine (RSMAX2-polynomial)': bnlearn_rsmax2_sparse_dict_scores[
                                "svm_po"],
                            'BN Support Vector Machine (RSMAX2-rbf)': bnlearn_rsmax2_sparse_dict_scores["svm_r"],
                            'BN Support Vector Machine (H2PC-sigmoid)': bnlearn_h2pc_sparse_dict_scores["svm"],
                            'BN Support Vector Machine (H2PC-polynomial)': bnlearn_h2pc_sparse_dict_scores["svm_po"],
                            'BN Support Vector Machine (H2PC-rbf)': bnlearn_h2pc_sparse_dict_scores["svm_r"],
                            'NT Support Vector Machine (logistic-sigmoid)': notears_sparse_dict_scores["svm"],
                            'NT Support Vector Machine (logistic-polynomial)': notears_sparse_dict_scores["svm_po"],
                            'NT Support Vector Machine (logistic-rbf)': notears_sparse_dict_scores["svm_r"],
                            'NT Support Vector Machine (L2-sigmoid)': notears_l2_sparse_dict_scores["svm"],
                            'NT Support Vector Machine (L2-polynomial)': notears_l2_sparse_dict_scores["svm_po"],
                            'NT Support Vector Machine (L2-rbf)': notears_l2_sparse_dict_scores["svm_r"],
                            'NT Support Vector Machine (Poisson-sigmoid)': notears_poisson_sparse_dict_scores["svm"],
                            'NT Support Vector Machine (Poisson-polynomial)': notears_poisson_sparse_dict_scores[
                                "svm_po"],
                            'NT Support Vector Machine (Poisson-rbf)': notears_poisson_sparse_dict_scores["svm_r"],
                            'Pomegranate Support Vector Machine (Exact-sigmoid)': pomegranate_exact_sparse_dict_scores[
                                "svm"], 'Pomegranate Support Vector Machine (Exact-polynomial)':
                                pomegranate_exact_sparse_dict_scores["svm_po"],
                            'Pomegranate Support Vector Machine (Exact-rbf)': pomegranate_exact_sparse_dict_scores[
                                "svm_r"], 'Pomegranate Support Vector Machine (Greedy-sigmoid)':
                                pomegranate_greedy_sparse_dict_scores["svm"],
                            'Pomegranate Support Vector Machine (Greedy-polynomial)':
                                pomegranate_greedy_sparse_dict_scores["svm_po"],
                            'Pomegranate Support Vector Machine (Greedy-rbf)': pomegranate_greedy_sparse_dict_scores[
                                "svm_r"],
                            'PGMPY Support Vector Machine (HC-sigmoid)': pgmpy_hc_sparse_dict_scores["svm"],
                            'PGMPY Support Vector Machine (HC-polynomial)': pgmpy_hc_sparse_dict_scores["svm_po"],
                            'PGMPY Support Vector Machine (HC-rbf)': pgmpy_hc_sparse_dict_scores["svm_r"],
                            'PGMPY Support Vector Machine (MMHC-sigmoid)': pgmpy_mmhc_sparse_dict_scores["svm"],
                            'PGMPY Support Vector Machine (MMHC-polynomial)': pgmpy_mmhc_sparse_dict_scores["svm_po"],
                            'PGMPY Support Vector Machine (MMHC-rbf)': pgmpy_mmhc_sparse_dict_scores["svm_r"],
                            'PGMPY Support Vector Machine (TREE-sigmoid)': pgmpy_tree_sparse_dict_scores["svm"],
                            'PGMPY Support Vector Machine (TREE-polynomial)': pgmpy_tree_sparse_dict_scores["svm_po"],
                            'PGMPY Support Vector Machine (TREE-rbf)': pgmpy_tree_sparse_dict_scores["svm_r"],
                            'BN K Nearest Neighbor (HC-weight)': bnlearn_sparse_dict_scores["knn"],
                            'BN K Nearest Neighbor (HC-distance)': bnlearn_sparse_dict_scores["knn_d"],
                            'BN K Nearest Neighbor (TABU-weight)': bnlearn_tabu_sparse_dict_scores["knn"],
                            'BN K Nearest Neighbor (TABU-distance)': bnlearn_tabu_sparse_dict_scores["knn_d"],
                            #'BN K Nearest Neighbor (PC-weight)': bnlearn_pc_sparse_dict_scores["knn"],
                            #'BN K Nearest Neighbor (PC-distance)': bnlearn_pc_sparse_dict_scores["knn_d"],
                            'BN K Nearest Neighbor (MMHC-weight)': bnlearn_mmhc_sparse_dict_scores["knn"],
                            'BN K Nearest Neighbor (MMHC-distance)': bnlearn_mmhc_sparse_dict_scores["knn_d"],
                            'BN K Nearest Neighbor (RSMAX2-weight)': bnlearn_rsmax2_sparse_dict_scores["knn"],
                            'BN K Nearest Neighbor (RSMAX2-distance)': bnlearn_rsmax2_sparse_dict_scores["knn_d"],
                            'BN K Nearest Neighbor (H2PC-weight)': bnlearn_h2pc_sparse_dict_scores["knn"],
                            'BN K Nearest Neighbor (H2PC-distance)': bnlearn_h2pc_sparse_dict_scores["knn_d"],
                            'NT K Nearest Neighbor (Logistic-weight)': notears_sparse_dict_scores["knn"],
                            'NT K Nearest Neighbor (Logistic-distance)': notears_sparse_dict_scores["knn_d"],
                            'NT K Nearest Neighbor (L2-weight)': notears_l2_sparse_dict_scores["knn"],
                            'NT K Nearest Neighbor (L2-distance)': notears_l2_sparse_dict_scores["knn_d"],
                            'NT K Nearest Neighbor (Poisson-weight)': notears_poisson_sparse_dict_scores["knn"],
                            'NT K Nearest Neighbor (Poisson-distance)': notears_poisson_sparse_dict_scores["knn_d"],
                            'POMEGRANATE K Nearest Neighbor (Exact-weight)': pomegranate_exact_sparse_dict_scores[
                                "knn"],
                            'POMEGRANATE K Nearest Neighbor (Exact-distance)': pomegranate_exact_sparse_dict_scores[
                                "knn_d"],
                            'POMEGRANATE K Nearest Neighbor (Greedy-weight)': pomegranate_greedy_sparse_dict_scores[
                                "knn"],
                            'POMEGRANATE K Nearest Neighbor (Greedy-distance)': pomegranate_greedy_sparse_dict_scores[
                                "knn_d"], 'PGMPY K Nearest Neighbor (HC-weight)': pgmpy_hc_sparse_dict_scores["knn"],
                            'PGMPY K Nearest Neighbor (HC-distance)': pgmpy_hc_sparse_dict_scores["knn_d"],
                            'PGMPY K Nearest Neighbor (MMHC-weight)': pgmpy_mmhc_sparse_dict_scores["knn"],
                            'PGMPY K Nearest Neighbor (MMHC-distance)': pgmpy_mmhc_sparse_dict_scores["knn_d"],
                            'PGMPY K Nearest Neighbor (TREE-weight)': pgmpy_tree_sparse_dict_scores["knn"],
                            'PGMPY K Nearest Neighbor (TREE-distance)': pgmpy_tree_sparse_dict_scores["knn_d"]}

    top_learned_sparse = max(sim_sparse_workflows, key=sim_sparse_workflows.get)
    print("Learned world - Sparse problem, Prediction: "+ top_learned_sparse + " (" + str(sim_sparse_workflows[top_learned_sparse]) + ")")

    real_dimension_workflows = {'Decision Tree (gini)': real_dimension_dt_scores,
                             'Decision Tree (entropy)': real_dimension_dt_entropy_scores,
                             'Random Forest (gini)': real_dimension_rf_scores,
                             'Random Forest (entropy)': real_dimension_rf_entropy_scores,
                             'Logistic Regression (none)': real_dimension_lr_scores,
                             'Logistic Regression (l1)': real_dimension_lr_l1_scores,
                             'Logistic Regression (l2)': real_dimension_lr_l2_scores,
                             'Logistic Regression (elasticnet)': real_dimension_lr_elastic_scores,
                             'Naive Bayes (bernoulli)': real_dimension_gb_scores,
                             'Naive Bayes (multinomial)': real_dimension_gb_multi_scores,
                             'Naive Bayes (gaussian)': real_dimension_gb_gaussian_scores,
                             'Naive Bayes (complement)': real_dimension_gb_complement_scores,
                             'Support Vector Machine (sigmoid)': real_dimension_svm_scores,
                             'Support Vector Machine (polynomial)': real_dimension_svm_poly_scores,
                             'Support Vector Machine (rbf)': real_dimension_svm_rbf_scores,
                             'K Nearest Neighbor (uniform)': real_dimension_knn_scores,
                             'K Nearest Neighbor (distance)': real_dimension_knn_distance_scores}
    top_real_dimension = max(real_dimension_workflows, key=real_dimension_workflows.get)
    print("Real world - Dimensional problem, Prediction: "+ top_real_dimension + " (" + str(real_dimension_workflows[top_real_dimension]) + ")")

    sim_dimension_workflows = {'BN Decision Tree (HC-gini)': bnlearn_dimension_dict_scores["dt"],
                            'BN Decision Tree (HC-entropy)': bnlearn_dimension_dict_scores["dt_e"],
                            'BN Decision Tree (TABU-gini)': bnlearn_tabu_dimension_dict_scores["dt"],
                            'BN Decision Tree (TABU-entropy)': bnlearn_tabu_dimension_dict_scores["dt_e"],
                            #'BN Decision Tree (PC-gini)': bnlearn_pc_dimension_dict_scores["dt"],
                            #'BN Decision Tree (PC-entropy)': bnlearn_pc_dimension_dict_scores["dt_e"],
                            'BN Decision Tree (MMHC-gini)': bnlearn_mmhc_dimension_dict_scores["dt"],
                            'BN Decision Tree (MMHC-entropy)': bnlearn_mmhc_dimension_dict_scores["dt_e"],
                            'BN Decision Tree (RSMAX2-gini)': bnlearn_rsmax2_dimension_dict_scores["dt"],
                            'BN Decision Tree (RSMAX2-entropy)': bnlearn_rsmax2_dimension_dict_scores["dt_e"],
                            'BN Decision Tree (H2PC-gini)': bnlearn_h2pc_dimension_dict_scores["dt"],
                            'BN Decision Tree (H2PC-entropy)': bnlearn_h2pc_dimension_dict_scores["dt_e"],
                            'NT Decision Tree (Logistic-gini)': notears_dimension_dict_scores["dt"],
                            'NT Decision Tree (Logistic-entropy)': notears_dimension_dict_scores["dt_e"],
                            'NT Decision Tree (L2-gini)': notears_l2_dimension_dict_scores["dt"],
                            'NT Decision Tree (L2-entropy)': notears_l2_dimension_dict_scores["dt_e"],
                            'NT Decision Tree (Poisson-gini)': notears_poisson_dimension_dict_scores["dt"],
                            'NT Decision Tree (Poisson-entropy)': notears_poisson_dimension_dict_scores["dt_e"],
                            'POMEGRANATE Decision Tree (Exact-gini)': pomegranate_exact_dimension_dict_scores["dt"],
                            'POMEGRANATE Decision Tree (Exact-entropy)': pomegranate_exact_dimension_dict_scores["dt_e"],
                            'POMEGRANATE Decision Tree (Greedy-gini)': pomegranate_greedy_dimension_dict_scores["dt"],
                            'POMEGRANATE Decision Tree (Greedy-entropy)': pomegranate_greedy_dimension_dict_scores["dt_e"],
                            'PGMPY Decision Tree (HC-gini)': pgmpy_hc_dimension_dict_scores["dt"],
                            'PGMPY Decision Tree (HC-entropy)': pgmpy_hc_dimension_dict_scores["dt_e"],
                            'PGMPY Decision Tree (MMHC-gini)': pgmpy_mmhc_dimension_dict_scores["dt"],
                            'PGMPY Decision Tree (HC-entropy)': pgmpy_mmhc_dimension_dict_scores["dt_e"],
                            'PGMPY Decision Tree (TREE-gini)': pgmpy_tree_dimension_dict_scores["dt"],
                            'PGMPY Decision Tree (TREE-entropy)': pgmpy_tree_dimension_dict_scores["dt_e"],
                            'BN Random Forest (HC-gini)': bnlearn_dimension_dict_scores["rf"],
                            'BN Random Forest (HC-entropy)': bnlearn_dimension_dict_scores["rf_e"],
                            'BN Random Forest (TABU-gini)': bnlearn_tabu_dimension_dict_scores["rf"],
                            'BN Random Forest (TABU-entropy)': bnlearn_tabu_dimension_dict_scores["rf_e"],
                            #'BN Random Forest (PC-gini)': bnlearn_pc_dimension_dict_scores["rf"],
                            #'BN Random Forest (PC-entropy)': bnlearn_pc_dimension_dict_scores["rf_e"],
                            'BN Random Forest (MMHC-gini)': bnlearn_mmhc_dimension_dict_scores["rf"],
                            'BN Random Forest (MMHC-entropy)': bnlearn_mmhc_dimension_dict_scores["rf_e"],
                            'BN Random Forest (RSMAX2-gini)': bnlearn_rsmax2_dimension_dict_scores["rf"],
                            'BN Random Forest (RSMAX2-entropy)': bnlearn_rsmax2_dimension_dict_scores["rf_e"],
                            'BN Random Forest (H2PC-gini)': bnlearn_h2pc_dimension_dict_scores["rf"],
                            'BN Random Forest (H2PC-entropy)': bnlearn_h2pc_dimension_dict_scores["rf_e"],
                            'NT Random Forest (Logistic-gini)': notears_dimension_dict_scores["rf"],
                            'NT Random Forest (Logistic-entropy)': notears_dimension_dict_scores["rf_e"],
                            'NT Random Forest (L2-gini)': notears_l2_dimension_dict_scores["rf"],
                            'NT Random Forest (l2-entropy)': notears_l2_dimension_dict_scores["rf_e"],
                            'NT Random Forest (Poisson-gini)': notears_poisson_dimension_dict_scores["rf"],
                            'NT Random Forest (Poisson-entropy)': notears_poisson_dimension_dict_scores["rf_e"],
                            'POMEGRANATE Random Forest (Exact-gini)': pomegranate_exact_dimension_dict_scores["rf"],
                            'POMEGRANATE Random Forest (Exact-entropy)': pomegranate_exact_dimension_dict_scores["rf_e"],
                            'POMEGRANATE Random Forest (Greedy-gini)': pomegranate_greedy_dimension_dict_scores["rf"],
                            'POMEGRANATE Random Forest (Greedy-entropy)': pomegranate_greedy_dimension_dict_scores["rf_e"],
                            'PGMPY Random Forest (HC-gini)': pgmpy_hc_dimension_dict_scores["rf"],
                            'PGMPY Random Forest (HC-entropy)': pgmpy_hc_dimension_dict_scores["rf_e"],
                            'PGMPY Random Forest (MMHC-gini)': pgmpy_mmhc_dimension_dict_scores["rf"],
                            'PGMPY Random Forest (HC-entropy)': pgmpy_mmhc_dimension_dict_scores["rf_e"],
                            'PGMPY Random Forest (TREE-gini)': pgmpy_tree_dimension_dict_scores["rf"],
                            'PGMPY Random Forest (TREE-entropy)': pgmpy_tree_dimension_dict_scores["rf_e"],
                            'BN Logistic Regression (HC-none)': bnlearn_dimension_dict_scores["lr"],
                            'BN Logistic Regression (HC-l1)': bnlearn_dimension_dict_scores["lr_l1"],
                            'BN Logistic Regression (HC-l2)': bnlearn_dimension_dict_scores["lr_l2"],
                            'BN Logistic Regression (HC-elastic)': bnlearn_dimension_dict_scores["lr_e"],
                            'BN Logistic Regression (TABU-none)': bnlearn_tabu_dimension_dict_scores["lr"],
                            'BN Logistic Regression (TABU-l1)': bnlearn_tabu_dimension_dict_scores["lr_l1"],
                            'BN Logistic Regression (TABU-l2)': bnlearn_tabu_dimension_dict_scores["lr_l2"],
                            'BN Logistic Regression (TABU-elastic)': bnlearn_tabu_dimension_dict_scores["lr_e"],
                            #'BN Logistic Regression (PC-none)': bnlearn_pc_dimension_dict_scores["lr"],
                            #'BN Logistic Regression (PC-l1)': bnlearn_pc_dimension_dict_scores["lr_l1"],
                            #'BN Logistic Regression (PC-l2)': bnlearn_pc_dimension_dict_scores["lr_l2"],
                            #'BN Logistic Regression (PC-elastic)': bnlearn_pc_dimension_dict_scores["lr_e"],
                            'BN Logistic Regression (MMHC-none)': bnlearn_mmhc_dimension_dict_scores["lr"],
                            'BN Logistic Regression (MMHC-l1)': bnlearn_mmhc_dimension_dict_scores["lr_l1"],
                            'BN Logistic Regression (MMHC-l2)': bnlearn_mmhc_dimension_dict_scores["lr_l2"],
                            'BN Logistic Regression (MMHC-elastic)': bnlearn_mmhc_dimension_dict_scores["lr_e"],
                            'BN Logistic Regression (RSMAX2-none)': bnlearn_rsmax2_dimension_dict_scores["lr"],
                            'BN Logistic Regression (RSMAX2-l1)': bnlearn_rsmax2_dimension_dict_scores["lr_l1"],
                            'BN Logistic Regression (RSMAX2-l2)': bnlearn_rsmax2_dimension_dict_scores["lr_l2"],
                            'BN Logistic Regression (RSMAX2-elastic)': bnlearn_rsmax2_dimension_dict_scores["lr_e"],
                            'BN Logistic Regression (H2PC-none)': bnlearn_h2pc_dimension_dict_scores["lr"],
                            'BN Logistic Regression (H2PC-l1)': bnlearn_h2pc_dimension_dict_scores["lr_l1"],
                            'BN Logistic Regression (H2PC-l2)': bnlearn_h2pc_dimension_dict_scores["lr_l2"],
                            'BN Logistic Regression (H2PC-elastic)': bnlearn_h2pc_dimension_dict_scores["lr_e"],
                            'POMEGRANATE Logistic Regression (Exact-none)': pomegranate_exact_dimension_dict_scores["lr"],
                            'POMEGRANATE Logistic Regression (Exact-l1)': pomegranate_exact_dimension_dict_scores["lr_l1"],
                            'POMEGRANATE Logistic Regression (Exact-l2)': pomegranate_exact_dimension_dict_scores["lr_l2"],
                            'POMEGRANATE Logistic Regression (Exact-elastic)': pomegranate_exact_dimension_dict_scores[
                                "lr_e"],
                            'POMEGRANATE Logistic Regression (Greedy-none)': pomegranate_greedy_dimension_dict_scores[
                                "lr"],
                            'POMEGRANATE Logistic Regression (Greedy-l1)': pomegranate_greedy_dimension_dict_scores[
                                "lr_l1"],
                            'POMEGRANATE Logistic Regression (Greedy-l2)': pomegranate_greedy_dimension_dict_scores[
                                "lr_l2"],
                            'POMEGRANATE Logistic Regression (Greedy-elastic)': pomegranate_greedy_dimension_dict_scores[
                                "lr_e"], 'PGMPY Logistic Regression (HC-none)': pgmpy_hc_dimension_dict_scores["lr"],
                            'PGMPY Logistic Regression (HC-l1)': pgmpy_hc_dimension_dict_scores["lr_l1"],
                            'PGMPY Logistic Regression (MMHC-l2)': pgmpy_mmhc_dimension_dict_scores["lr_l2"],
                            'PGMPY Logistic Regression (HC-elastic)': pgmpy_mmhc_dimension_dict_scores["lr_e"],
                            'PGMPY Logistic Regression (TREE-none)': pgmpy_tree_dimension_dict_scores["lr"],
                            'PGMPY Logistic Regression (TREE-l1)': pgmpy_tree_dimension_dict_scores["lr_l1"],
                            'PGMPY Logistic Regression (TREE-l2)': pgmpy_tree_dimension_dict_scores["lr_l2"],
                            'PGMPY Logistic Regression (TREE-elastic)': pgmpy_tree_dimension_dict_scores["lr_e"],
                            'PGMPY Logistic Regression (MMHC-none)': pgmpy_mmhc_dimension_dict_scores["lr"],
                            'PGMPY Logistic Regression (MMHC-l1)': pgmpy_mmhc_dimension_dict_scores["lr_l1"],
                            'PGMPY Logistic Regression (MMHC-l2)': pgmpy_mmhc_dimension_dict_scores["lr_l2"],
                            'PGMPY Logistic Regression (MMHC-elastic)': pgmpy_mmhc_dimension_dict_scores["lr_e"],
                            'NT Logistic Regression (Logistic-none)': notears_dimension_dict_scores["lr"],
                            'NT Logistic Regression (Logistic-l1)': notears_dimension_dict_scores["lr_l1"],
                            'NT Logistic Regression (Logistic-l2)': notears_dimension_dict_scores["lr_l2"],
                            'NT Logistic Regression (Logistic-elastic)': notears_dimension_dict_scores["lr_e"],
                            'NT Logistic Regression (L2-none)': notears_l2_dimension_dict_scores["lr"],
                            'NT Logistic Regression (L2-l1)': notears_l2_dimension_dict_scores["lr_l1"],
                            'NT Logistic Regression (L2-l2)': notears_l2_dimension_dict_scores["lr_l2"],
                            'NT Logistic Regression (L2-elastic)': notears_l2_dimension_dict_scores["lr_e"],
                            'NT Logistic Regression (Poisson-none)': notears_poisson_dimension_dict_scores["lr"],
                            'NT Logistic Regression (Poisson-l1)': notears_poisson_dimension_dict_scores["lr_l1"],
                            'NT Logistic Regression (Poisson-l2)': notears_poisson_dimension_dict_scores["lr_l2"],
                            'NT Logistic Regression (Poisson-elastic)': notears_poisson_dimension_dict_scores["lr_e"],
                            'BN Naive Bayes (HC-bernoulli)': bnlearn_dimension_dict_scores["nb"],
                            'BN Naive Bayes (HC-gaussian)': bnlearn_dimension_dict_scores["nb_g"],
                            'BN Naive Bayes (HC-multinomial)': bnlearn_dimension_dict_scores["nb_m"],
                            'BN Naive Bayes (HC-complement)': bnlearn_dimension_dict_scores["nb_c"],
                            'BN Naive Bayes (TABU-bernoulli)': bnlearn_tabu_dimension_dict_scores["nb"],
                            'BN Naive Bayes (TABU-gaussian)': bnlearn_tabu_dimension_dict_scores["nb_g"],
                            'BN Naive Bayes (TABU-multinomial)': bnlearn_tabu_dimension_dict_scores["nb_m"],
                            'BN Naive Bayes (TABU-complement)': bnlearn_tabu_dimension_dict_scores["nb_c"],
                            #'BN Naive Bayes (PC-bernoulli)': bnlearn_pc_dimension_dict_scores["nb"],
                            #'BN Naive Bayes (PC-gaussian)': bnlearn_pc_dimension_dict_scores["nb_g"],
                            #'BN Naive Bayes (PC-multinomial)': bnlearn_pc_dimension_dict_scores["nb_m"],
                            #'BN Naive Bayes (PC-complement)': bnlearn_pc_dimension_dict_scores["nb_c"],
                            'BN Naive Bayes (MMHC-bernoulli)': bnlearn_mmhc_dimension_dict_scores["nb"],
                            'BN Naive Bayes (MMHC-gaussian)': bnlearn_mmhc_dimension_dict_scores["nb_g"],
                            'BN Naive Bayes (MMHC-multinomial)': bnlearn_mmhc_dimension_dict_scores["nb_m"],
                            'BN Naive Bayes (MMHC-complement)': bnlearn_mmhc_dimension_dict_scores["nb_c"],
                            'BN Naive Bayes (RSMAX2-bernoulli)': bnlearn_rsmax2_dimension_dict_scores["nb"],
                            'BN Naive Bayes (RSMAX2-gaussian)': bnlearn_rsmax2_dimension_dict_scores["nb_g"],
                            'BN Naive Bayes (RSMAX2-multinomial)': bnlearn_rsmax2_dimension_dict_scores["nb_m"],
                            'BN Naive Bayes (RSMAX2-complement)': bnlearn_rsmax2_dimension_dict_scores["nb_c"],
                            'BN Naive Bayes (H2PC-bernoulli)': bnlearn_h2pc_dimension_dict_scores["nb"],
                            'BN Naive Bayes (H2PC-gaussian)': bnlearn_h2pc_dimension_dict_scores["nb_g"],
                            'BN Naive Bayes (H2PC-multinomial)': bnlearn_h2pc_dimension_dict_scores["nb_m"],
                            'BN Naive Bayes (H2PC-complement)': bnlearn_h2pc_dimension_dict_scores["nb_c"],
                            'NT Naive Bayes (Logistic-bernoulli)': notears_dimension_dict_scores["nb"],
                            'NT Naive Bayes (Logistic-gaussian)': notears_dimension_dict_scores["nb_g"],
                            'NT Naive Bayes (Logistic-multinomial)': notears_dimension_dict_scores["nb_m"],
                            'NT Naive Bayes (Logistic-complement)': notears_dimension_dict_scores["nb_c"],
                            'NT Naive Bayes (L2-bernoulli)': notears_l2_dimension_dict_scores["nb"],
                            'NT Naive Bayes (L2-gaussian)': notears_l2_dimension_dict_scores["nb_g"],
                            'NT Naive Bayes (L2-multinomial)': notears_l2_dimension_dict_scores["nb_m"],
                            'NT Naive Bayes (L2-complement)': notears_l2_dimension_dict_scores["nb_c"],
                            'NT Naive Bayes (Poisson-bernoulli)': notears_poisson_dimension_dict_scores["nb"],
                            'NT Naive Bayes (Poisson-gaussian)': notears_poisson_dimension_dict_scores["nb_g"],
                            'NT Naive Bayes (Poisson-multinomial)': notears_poisson_dimension_dict_scores["nb_m"],
                            'NT Naive Bayes (Poisson-complement)': notears_poisson_dimension_dict_scores["nb_c"],
                            'POMEGRANATE Naive Bayes (Greedy-bernoulli)': pomegranate_greedy_dimension_dict_scores["nb"],
                            'POMEGRANATE Naive Bayes (Greedy-gaussian)': pomegranate_greedy_dimension_dict_scores["nb_g"],
                            'POMEGRANATE Naive Bayes (Greedy-multinomial)': pomegranate_greedy_dimension_dict_scores[
                                "nb_m"],
                            'POMEGRANATE Naive Bayes (Greedy-complement)': pomegranate_greedy_dimension_dict_scores[
                                "nb_c"],
                            'POMEGRANATE Naive Bayes (Exact-bernoulli)': pomegranate_exact_dimension_dict_scores["nb"],
                            'POMEGRANATE Naive Bayes (Exact-gaussian)': pomegranate_exact_dimension_dict_scores["nb_g"],
                            'POMEGRANATE Naive Bayes (Exact-multinomial)': pomegranate_exact_dimension_dict_scores["nb_m"],
                            'POMEGRANATE Naive Bayes (Exact-complement)': pomegranate_exact_dimension_dict_scores["nb_c"],
                            'PGMPY Naive Bayes (HC-bernoulli)': pgmpy_hc_dimension_dict_scores["nb"],
                            'PGMPY Naive Bayes (HC-gaussian)': pgmpy_hc_dimension_dict_scores["nb_g"],
                            'PGMPY Naive Bayes (HC-multinomial)': pgmpy_hc_dimension_dict_scores["nb_m"],
                            'PGMPY Naive Bayes (HC-complement)': pgmpy_hc_dimension_dict_scores["nb_c"],
                            'PGMPY Naive Bayes (MMHC-bernoulli)': pgmpy_mmhc_dimension_dict_scores["nb"],
                            'PGMPY Naive Bayes (MMHC-gaussian)': pgmpy_mmhc_dimension_dict_scores["nb_g"],
                            'PGMPY Naive Bayes (MMHC-multinomial)': pgmpy_mmhc_dimension_dict_scores["nb_m"],
                            'PGMPY Naive Bayes (MMHC-complement)': pgmpy_mmhc_dimension_dict_scores["nb_c"],
                            'PGMPY Naive Bayes (TREE-bernoulli)': pgmpy_tree_dimension_dict_scores["nb"],
                            'PGMPY Naive Bayes (TREE-gaussian)': pgmpy_tree_dimension_dict_scores["nb_g"],
                            'PGMPY Naive Bayes (TREE-multinomial)': pgmpy_tree_dimension_dict_scores["nb_m"],
                            'PGMPY Naive Bayes (TREE-complement)': pgmpy_tree_dimension_dict_scores["nb_c"],
                            'BN Support Vector Machine (HC-sigmoid)': bnlearn_dimension_dict_scores["svm"],
                            'BN Support Vector Machine (HC-polynomial)': bnlearn_dimension_dict_scores["svm_po"],
                            'BN Support Vector Machine (HC-rbf)': bnlearn_dimension_dict_scores["svm_r"],
                            'BN Support Vector Machine (TABU-sigmoid)': bnlearn_tabu_dimension_dict_scores["svm"],
                            'BN Support Vector Machine (TABU-polynomial)': bnlearn_tabu_dimension_dict_scores["svm_po"],
                            'BN Support Vector Machine (TABU-rbf)': bnlearn_tabu_dimension_dict_scores["svm_r"],
                            #'BN Support Vector Machine (PC-sigmoid)': bnlearn_pc_dimension_dict_scores["svm"],
                            #'BN Support Vector Machine (PC-polynomial)': bnlearn_pc_dimension_dict_scores["svm_po"],
                            #'BN Support Vector Machine (PC-rbf)': bnlearn_pc_dimension_dict_scores["svm_r"],
                            'BN Support Vector Machine (MMHC-sigmoid)': bnlearn_mmhc_dimension_dict_scores["svm"],
                            'BN Support Vector Machine (MMHC-polynomial)': bnlearn_mmhc_dimension_dict_scores["svm_po"],
                            'BN Support Vector Machine (MMHC-rbf)': bnlearn_mmhc_dimension_dict_scores["svm_r"],
                            'BN Support Vector Machine (RSMAX2-sigmoid)': bnlearn_rsmax2_dimension_dict_scores["svm"],
                            'BN Support Vector Machine (RSMAX2-polynomial)': bnlearn_rsmax2_dimension_dict_scores[
                                "svm_po"],
                            'BN Support Vector Machine (RSMAX2-rbf)': bnlearn_rsmax2_dimension_dict_scores["svm_r"],
                            'BN Support Vector Machine (H2PC-sigmoid)': bnlearn_h2pc_dimension_dict_scores["svm"],
                            'BN Support Vector Machine (H2PC-polynomial)': bnlearn_h2pc_dimension_dict_scores["svm_po"],
                            'BN Support Vector Machine (H2PC-rbf)': bnlearn_h2pc_dimension_dict_scores["svm_r"],
                            'NT Support Vector Machine (logistic-sigmoid)': notears_dimension_dict_scores["svm"],
                            'NT Support Vector Machine (logistic-polynomial)': notears_dimension_dict_scores["svm_po"],
                            'NT Support Vector Machine (logistic-rbf)': notears_dimension_dict_scores["svm_r"],
                            'NT Support Vector Machine (L2-sigmoid)': notears_l2_dimension_dict_scores["svm"],
                            'NT Support Vector Machine (L2-polynomial)': notears_l2_dimension_dict_scores["svm_po"],
                            'NT Support Vector Machine (L2-rbf)': notears_l2_dimension_dict_scores["svm_r"],
                            'NT Support Vector Machine (Poisson-sigmoid)': notears_poisson_dimension_dict_scores["svm"],
                            'NT Support Vector Machine (Poisson-polynomial)': notears_poisson_dimension_dict_scores[
                                "svm_po"],
                            'NT Support Vector Machine (Poisson-rbf)': notears_poisson_dimension_dict_scores["svm_r"],
                            'Pomegranate Support Vector Machine (Exact-sigmoid)': pomegranate_exact_dimension_dict_scores[
                                "svm"], 'Pomegranate Support Vector Machine (Exact-polynomial)':
                                pomegranate_exact_dimension_dict_scores["svm_po"],
                            'Pomegranate Support Vector Machine (Exact-rbf)': pomegranate_exact_dimension_dict_scores[
                                "svm_r"], 'Pomegranate Support Vector Machine (Greedy-sigmoid)':
                                pomegranate_greedy_dimension_dict_scores["svm"],
                            'Pomegranate Support Vector Machine (Greedy-polynomial)':
                                pomegranate_greedy_dimension_dict_scores["svm_po"],
                            'Pomegranate Support Vector Machine (Greedy-rbf)': pomegranate_greedy_dimension_dict_scores[
                                "svm_r"],
                            'PGMPY Support Vector Machine (HC-sigmoid)': pgmpy_hc_dimension_dict_scores["svm"],
                            'PGMPY Support Vector Machine (HC-polynomial)': pgmpy_hc_dimension_dict_scores["svm_po"],
                            'PGMPY Support Vector Machine (HC-rbf)': pgmpy_hc_dimension_dict_scores["svm_r"],
                            'PGMPY Support Vector Machine (MMHC-sigmoid)': pgmpy_mmhc_dimension_dict_scores["svm"],
                            'PGMPY Support Vector Machine (MMHC-polynomial)': pgmpy_mmhc_dimension_dict_scores["svm_po"],
                            'PGMPY Support Vector Machine (MMHC-rbf)': pgmpy_mmhc_dimension_dict_scores["svm_r"],
                            'PGMPY Support Vector Machine (TREE-sigmoid)': pgmpy_tree_dimension_dict_scores["svm"],
                            'PGMPY Support Vector Machine (TREE-polynomial)': pgmpy_tree_dimension_dict_scores["svm_po"],
                            'PGMPY Support Vector Machine (TREE-rbf)': pgmpy_tree_dimension_dict_scores["svm_r"],
                            'BN K Nearest Neighbor (HC-weight)': bnlearn_dimension_dict_scores["knn"],
                            'BN K Nearest Neighbor (HC-distance)': bnlearn_dimension_dict_scores["knn_d"],
                            'BN K Nearest Neighbor (TABU-weight)': bnlearn_tabu_dimension_dict_scores["knn"],
                            'BN K Nearest Neighbor (TABU-distance)': bnlearn_tabu_dimension_dict_scores["knn_d"],
                            #'BN K Nearest Neighbor (PC-weight)': bnlearn_pc_dimension_dict_scores["knn"],
                            #'BN K Nearest Neighbor (PC-distance)': bnlearn_pc_dimension_dict_scores["knn_d"],
                            'BN K Nearest Neighbor (MMHC-weight)': bnlearn_mmhc_dimension_dict_scores["knn"],
                            'BN K Nearest Neighbor (MMHC-distance)': bnlearn_mmhc_dimension_dict_scores["knn_d"],
                            'BN K Nearest Neighbor (RSMAX2-weight)': bnlearn_rsmax2_dimension_dict_scores["knn"],
                            'BN K Nearest Neighbor (RSMAX2-distance)': bnlearn_rsmax2_dimension_dict_scores["knn_d"],
                            'BN K Nearest Neighbor (H2PC-weight)': bnlearn_h2pc_dimension_dict_scores["knn"],
                            'BN K Nearest Neighbor (H2PC-distance)': bnlearn_h2pc_dimension_dict_scores["knn_d"],
                            'NT K Nearest Neighbor (Logistic-weight)': notears_dimension_dict_scores["knn"],
                            'NT K Nearest Neighbor (Logistic-distance)': notears_dimension_dict_scores["knn_d"],
                            'NT K Nearest Neighbor (L2-weight)': notears_l2_dimension_dict_scores["knn"],
                            'NT K Nearest Neighbor (L2-distance)': notears_l2_dimension_dict_scores["knn_d"],
                            'NT K Nearest Neighbor (Poisson-weight)': notears_poisson_dimension_dict_scores["knn"],
                            'NT K Nearest Neighbor (Poisson-distance)': notears_poisson_dimension_dict_scores["knn_d"],
                            'POMEGRANATE K Nearest Neighbor (Exact-weight)': pomegranate_exact_dimension_dict_scores[
                                "knn"],
                            'POMEGRANATE K Nearest Neighbor (Exact-distance)': pomegranate_exact_dimension_dict_scores[
                                "knn_d"],
                            'POMEGRANATE K Nearest Neighbor (Greedy-weight)': pomegranate_greedy_dimension_dict_scores[
                                "knn"],
                            'POMEGRANATE K Nearest Neighbor (Greedy-distance)': pomegranate_greedy_dimension_dict_scores[
                                "knn_d"], 'PGMPY K Nearest Neighbor (HC-weight)': pgmpy_hc_dimension_dict_scores["knn"],
                            'PGMPY K Nearest Neighbor (HC-distance)': pgmpy_hc_dimension_dict_scores["knn_d"],
                            'PGMPY K Nearest Neighbor (MMHC-weight)': pgmpy_mmhc_dimension_dict_scores["knn"],
                            'PGMPY K Nearest Neighbor (MMHC-distance)': pgmpy_mmhc_dimension_dict_scores["knn_d"],
                            'PGMPY K Nearest Neighbor (TREE-weight)': pgmpy_tree_dimension_dict_scores["knn"],
                            'PGMPY K Nearest Neighbor (TREE-distance)': pgmpy_tree_dimension_dict_scores["knn_d"]}

    top_learned_dimension = max(sim_dimension_workflows, key=sim_dimension_workflows.get)
    print("Learned world - Dimensional problem, Prediction: "+ top_learned_dimension + " (" + str(sim_dimension_workflows[top_learned_dimension]) + ")")

real_experiment_summary = pd.read_csv("real_experiments_summary.csv")
real_experiment_summary

learned_experiment_summary = pd.read_csv("simulation_experiments_summary.csv")
learned_experiment_summary

prediction_real_learned()
