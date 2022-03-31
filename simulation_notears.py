import notears
import numpy as np
from notears import utils
from notears.linear import notears_linear
from notears.nonlinear import notears_nonlinear, NotearsMLP


def notears_setup(train_data, training_n, test_n):
    nt_sampling = notears_linear(train_data[0:100], lambda1=0.01, loss_type='logistic')
    nt_sampling_train = utils.simulate_linear_sem(nt_sampling, training_n, 'logistic')
    nt_test_train = utils.simulate_linear_sem(nt_sampling, test_n, 'logistic')
    np.savetxt('W_est_train.csv', nt_sampling_train, delimiter=',')
    return nt_sampling_train, nt_test_train

def notears_setup_b(train_data, training_n, test_n):
    nt_sampling = notears_linear(train_data[0:100], lambda1=0.01, loss_type='l2')
    nt_sampling_train = utils.simulate_linear_sem(nt_sampling, training_n, 'logistic')
    nt_test_train = utils.simulate_linear_sem(nt_sampling, test_n, 'logistic')
    np.savetxt('W_est_train.csv', nt_sampling_train, delimiter=',')
    return nt_sampling_train, nt_test_train

def notears_setup_c(train_data, training_n, test_n):
    nt_sampling = notears_linear(train_data[0:100], lambda1=0.01, loss_type='poisson')
    nt_sampling_train = utils.simulate_linear_sem(nt_sampling, training_n, 'logistic')
    nt_test_train = utils.simulate_linear_sem(nt_sampling, test_n, 'logistic')
    np.savetxt('W_est_train.csv', nt_sampling_train, delimiter=',')
    return nt_sampling_train, nt_test_train

def notears_nonlinear_setup(train_data, training_n):
    n, d, s0, graph_type, sem_type = 100, 5, 10, 'ER', 'mim'
    B_true = utils.simulate_dag(d, s0, graph_type)

    X = utils.simulate_nonlinear_sem(B_true, n, sem_type)

    model = NotearsMLP(dims=[d, 10, 2], bias=True) #input, weight, bias
    nt_nonlinear_sampling = notears_nonlinear(model.float(), X, lambda1=0.01, lambda2=0.01)
    np.savetxt('K_est_train.csv', nt_nonlinear_sampling, delimiter=',')
