import pandas as pd
import numpy as np
from pgmpy.estimators import PC
from pgmpy.sampling import BayesianModelInference
from pgmpy.sampling import _return_samples
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.sampling import GibbsSampling
from pgmpy.models import BayesianNetwork
from notears.nonlinear import notears_nonlinear, NotearsMLP
from statsmodels.tools.eval_measures import bic
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.estimators import TreeSearch
from pgmpy.estimators import MmhcEstimator
from pgmpy.estimators import ExhaustiveSearch


def pgmpy_setup_pc(train_data, training_n):
    print("this is the training data shape before structure learning", train_data.shape)
    model_learn = PC(train_data)
    model = model_learn.estimate()
    #print("Dag",model)
    construct = BayesianModel(model)
    estimator = BayesianEstimator(construct, train_data)
    cpds = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=1000)
    for cpd in cpds:
        construct.add_cpds(cpd)
    construct.check_model()
    pgmpy_sampling_train = construct.simulate(n_samples=int(1000))
    print("this is the output shape after structure learning", pgmpy_sampling_train.shape)
    #print(pgmpy_sampling_train)
    np.savetxt('V_est_train.csv', pgmpy_sampling_train, delimiter=',')
    return pgmpy_sampling_train

def pgmpy_setup_hc(train_data, training_n):
    print("this is the training data shape before structure learning", train_data.shape)
    est = HillClimbSearch(train_data)
    best_model = est.estimate(scoring_method=BicScore(train_data))
    #print("Dag",best_model)
    construct = BayesianModel(best_model)
    estimator = BayesianEstimator(construct, train_data)
    cpds = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=1000)
    for cpd in cpds:
        construct.add_cpds(cpd)
    construct.check_model()
    pgmpy_sampling_train = construct.simulate(n_samples=int(1000))
    print("this is the output shape after structure learning", pgmpy_sampling_train.shape)
    #print(pgmpy_sampling_train)
    np.savetxt('V_est_train.csv', pgmpy_sampling_train, delimiter=',')
    return pgmpy_sampling_train

def pgmpy_setup_tree(train_data, training_n):
    print("this is the training data shape before structure learning", train_data.shape)
    est = TreeSearch(train_data)
    best_model = est.estimate(estimator_type='chow-liu')
    #print("Dag",best_model)
    construct = BayesianModel(best_model)
    estimator = BayesianEstimator(construct, train_data)
    cpds = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=1000)
    for cpd in cpds:
        construct.add_cpds(cpd)
    construct.check_model()
    pgmpy_sampling_train = construct.simulate(n_samples=int(1000))
    print("this is the output shape after structure learning", pgmpy_sampling_train.shape)
    #print(pgmpy_sampling_train)
    np.savetxt('V_est_train.csv', pgmpy_sampling_train, delimiter=',')
    return pgmpy_sampling_train

def pgmpy_setup_mmhc(train_data, training_n):
    print("this is the training data shape before structure learning", train_data.shape)
    est = MmhcEstimator(train_data)
    best_model = est.estimate()
    #print("Dag",best_model)
    construct = BayesianModel(best_model)
    estimator = BayesianEstimator(construct, train_data)
    cpds = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=1000)
    for cpd in cpds:
        construct.add_cpds(cpd)
    construct.check_model()
    pgmpy_sampling_train = construct.simulate(n_samples=int(1000))
    print("this is the output shape after structure learning", pgmpy_sampling_train.shape)
    #print(pgmpy_sampling_train)
    np.savetxt('V_est_train.csv', pgmpy_sampling_train, delimiter=',')
    return pgmpy_sampling_train

#def pgmpy_setup_exhaustive(train_data, training_n):
#    print("this is the training data shape before structure learning", train_data.shape)
#    est = ExhaustiveSearch(train_data)
#    best_model = est.estimate()
#    print("Dag",best_model)
#    construct = BayesianModel(best_model)
#    estimator = BayesianEstimator(construct, train_data)
#    cpds = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=1000)
#    for cpd in cpds:
#        construct.add_cpds(cpd)
#    construct.check_model()
#    pgmpy_sampling_train = construct.simulate(n_samples=int(1000))
#    print("this is the output shape after structure learning", pgmpy_sampling_train.shape)
#    print(pgmpy_sampling_train)
#    np.savetxt('V_est_train.csv', pgmpy_sampling_train, delimiter=',')

