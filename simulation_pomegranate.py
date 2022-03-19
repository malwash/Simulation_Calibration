import notears
import numpy as np
from pomegranate import BayesianNetwork

from notears.nonlinear import notears_nonlinear, NotearsMLP
from statsmodels.tools.eval_measures import bic


def pomegranate_setup(train_data, training_n):
    model = BayesianNetwork.from_samples(train_data, state_names=train_data.columns.values, algorithm='exact')
    #print(model.structure)
#    model.plot()
    nt_sampling_train = model.sample(1000)
    #print(nt_sampling_train)
    np.savetxt('X_est_train.csv', nt_sampling_train, delimiter=',')
    #np.savetxt('W_est_test.csv', nt_sampling_test, delimiter=',')
    return nt_sampling_train

def pomegranate_setup_b(train_data, training_n):
    model = BayesianNetwork.from_samples(train_data, state_names=train_data.columns.values, algorithm='greedy')
    #print(model.structure)
#    model.plot()
    nt_sampling_train = model.sample(1000)
    #print(nt_sampling_train)
    np.savetxt('X_est_train.csv', nt_sampling_train, delimiter=',')
    #np.savetxt('W_est_test.csv', nt_sampling_test, delimiter=',')
    return nt_sampling_train

def pomegranate_setup_c(train_data, training_n):
    model = BayesianNetwork.from_samples(train_data, state_names=train_data.columns.values, algorithm='chow-liu’')
    #print(model.structure)
#    model.plot()
    nt_sampling_train = model.sample(1000)
    #print(nt_sampling_train)
    np.savetxt('X_est_train.csv', nt_sampling_train, delimiter=',')
    #np.savetxt('W_est_test.csv', nt_sampling_test, delimiter=',')
    return nt_sampling_train
