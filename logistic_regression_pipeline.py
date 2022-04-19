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
    learners = ["notears","pgmpy","pomegranate"]
    notears_loss = ["logistic", "l2", "poisson"]
    pgmpy_algorithms = ["hc","tree", "mmhc"]
    pomegranate_algorithms = ["exact", "greedy"]
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
    #for loop with drawn parameters from dict - list comprehension
    #for algorithm in ...
    #for model in ..
    real_results, learned_results = run_all()
    print(real_results)
    print(learned_results)
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

write_results_to_csv()

def write_results_to_figures():
    # Group by figure
    labels = ['DT_G', 'DT_E', 'RF_G', 'RF_E', 'LR', 'LR_L1', 'LR_L2', 'LR_E', 'NB_B', 'NB_G', 'NB_M', 'NB_C', 'SVM_S',
              'SVM_P', 'SVM_R', 'KNN_W', 'KNN_D']
    #bn_means = [round(mean(bnlearn_linear_dict_scores["dt"]),2), round(mean(bnlearn_linear_dict_scores["dt_e"]),2), round(mean(bnlearn_linear_dict_scores["rf"]),2), round(mean(bnlearn_linear_dict_scores["rf_e"]),2), round(mean(bnlearn_linear_dict_scores["lr"]),2), round(mean(bnlearn_linear_dict_scores["lr_l1"]),2), round(mean(bnlearn_linear_dict_scores["lr_l2"]),2), round(mean(bnlearn_linear_dict_scores["lr_e"]),2), round(mean(bnlearn_linear_dict_scores["nb"]),2), round(mean(bnlearn_linear_dict_scores["nb_g"]),2), round(mean(bnlearn_linear_dict_scores["nb_m"]),2), round(mean(bnlearn_linear_dict_scores["nb_c"]),2), round(mean(bnlearn_linear_dict_scores["svm"]),2), round(mean(bnlearn_linear_dict_scores["svm_po"]),2), round(mean(bnlearn_linear_dict_scores["svm_r"]),2), round(mean(bnlearn_linear_dict_scores["knn"]),2), round(mean(bnlearn_linear_dict_scores["knn_d"]),2)]
    #nt_means = [round(mean(notears_linear_dict_scores["dt"]),2), round(mean(notears_linear_dict_scores["dt_e"]),2), round(mean(notears_linear_dict_scores["rf"]),2), round(mean(notears_linear_dict_scores["rf_e"]),2), round(mean(notears_linear_dict_scores["lr"]),2), round(mean(notears_linear_dict_scores["lr_l1"]),2), round(mean(notears_linear_dict_scores["lr_l2"]),2), round(mean(notears_linear_dict_scores["lr_e"]),2), round(mean(notears_linear_dict_scores["nb"]),2), round(mean(notears_linear_dict_scores["nb_g"]),2), round(mean(notears_linear_dict_scores["nb_m"]),2), round(mean(notears_linear_dict_scores["nb_c"]),2), round(mean(notears_linear_dict_scores["svm"]),2), round(mean(notears_linear_dict_scores["svm_po"]),2), round(mean(notears_linear_dict_scores["svm_r"]),2), round(mean(notears_linear_dict_scores["knn"]),2), round(mean(notears_linear_dict_scores["knn_d"]),2)]

    x = np.arange(len(labels))  # the label locations
    width = 2  # the width of the bars

    fig, ax = plt.subplots()
    #rects1 = ax.bar(x - width / 2, bn_means, width, label='NO')
    #rects2 = ax.bar(x + width / 2, nt_means, width, label='NO_TEARS')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Linear Problem - Performance by library on ML technique')
    ax.set_xticks(x, labels)
    ax.legend()

    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)

    fig.set_size_inches(15, 15)
    fig.tight_layout()
    plt.savefig('pipeline_summary_benchmark_by_library_bargraph.png')
    plt.show()

#write_real_to_figures()

def prediction_real_learned():
    #List with values and loop

    print("#### SimCal Real/Learned-world Predictions ####")

    print("-- Exact (1-1) max(rank) output")
    #real_linear_workflows = {'Decision Tree (gini)': max(real_linear_dt_scores), 'Decision Tree (entropy)': max(real_linear_dt_entropy_scores), 'Random Forest (gini)': max(real_linear_rf_scores), 'Random Forest (entropy)': max(real_linear_rf_entropy_scores),'Logistic Regression (none)': max(real_linear_lr_scores), 'Logistic Regression (l1)': max(real_linear_lr_l1_scores), 'Logistic Regression (l2)': max(real_linear_lr_l2_scores), 'Logistic Regression (elasticnet)': max(real_linear_lr_elastic_scores), 'Naive Bayes (bernoulli)': max(real_linear_gb_scores), 'Naive Bayes (multinomial)': max(real_linear_gb_multi_scores), 'Naive Bayes (gaussian)': max(real_linear_gb_gaussian_scores), 'Naive Bayes (complement)': max(real_linear_gb_complement_scores), 'Support Vector Machine (sigmoid)': max(real_linear_svm_scores), 'Support Vector Machine (polynomial)': max(real_linear_svm_poly_scores), 'Support Vector Machine (rbf)': max(real_linear_svm_rbf_scores), 'K Nearest Neighbor (uniform)': max(real_linear_knn_scores), 'K Nearest Neighbor (distance)': max(real_linear_knn_distance_scores)}
    #top_real_linear = max(real_linear_workflows, key=real_linear_workflows.get)
    #print("Real world - Linear problem, Prediction: ", top_real_linear)
    #sim_linear_workflows = {'BN Decision Tree (gini)': max(bnlearn_linear_dict_scores["dt"]), 'BN Decision Tree (entropy)': max(bnlearn_linear_dict_scores["dt_e"]),'NT Decision Tree (gini)': max(notears_linear_dict_scores["dt"]),'NT Decision Tree (entropy)': max(notears_linear_dict_scores["dt_e"]), 'BN Random Forest (gini)': max(bnlearn_linear_dict_scores["rf"]), 'BN Random Forest (entropy)': max(bnlearn_linear_dict_scores["rf_e"]),'NT Random Forest (gini)': max(notears_linear_dict_scores["rf"]),'NT Random Forest (entropy)': max(notears_linear_dict_scores["rf_e"]),'BN Logistic Regression (none)': max(bnlearn_linear_dict_scores["lr"]),'BN Logistic Regression (l1)': max(bnlearn_linear_dict_scores["lr_l1"]),'BN Logistic Regression (l2)': max(bnlearn_linear_dict_scores["lr_l2"]),'BN Logistic Regression (elastic)': max(bnlearn_linear_dict_scores["lr_e"]), 'NT Logistic Regression (none)': max(notears_linear_dict_scores["lr"]),  'NT Logistic Regression (l1)': max(notears_linear_dict_scores["lr_l1"]), 'NT Logistic Regression (l2)': max(notears_linear_dict_scores["lr_l2"]), 'NT Logistic Regression (elastic)': max(notears_linear_dict_scores["lr_e"]),'BN Naive Bayes (bernoulli)': max(bnlearn_linear_dict_scores["nb"]),'BN Naive Bayes (gaussian)': max(bnlearn_linear_dict_scores["nb_g"]),'BN Naive Bayes (multinomial)': max(bnlearn_linear_dict_scores["nb_m"]),'BN Naive Bayes (complement)': max(bnlearn_linear_dict_scores["nb_c"]), 'NT Naive Bayes (bernoulli)': max(notears_linear_dict_scores["nb"]),'NT Naive Bayes (gaussian)': max(notears_linear_dict_scores["nb_g"]),'NT Naive Bayes (multinomial)': max(notears_linear_dict_scores["nb_m"]),'NT Naive Bayes (complement)': max(notears_linear_dict_scores["nb_c"]), 'BN Support Vector Machine (sigmoid)': max(bnlearn_linear_dict_scores["svm"]), 'BN Support Vector Machine (polynomial)': max(bnlearn_linear_dict_scores["svm_po"]), 'BN Support Vector Machine (rbf)': max(bnlearn_linear_dict_scores["svm_r"]), 'NT Support Vector Machine (sigmoid)': max(notears_linear_dict_scores["svm"]),'NT Support Vector Machine (polynomial)': max(notears_linear_dict_scores["svm_po"]),'NT Support Vector Machine (rbf)': max(notears_linear_dict_scores["svm_r"]), 'BN K Nearest Neighbor (weight)': max(bnlearn_linear_dict_scores["knn"]),'BN K Nearest Neighbor (distance)': max(bnlearn_linear_dict_scores["knn_d"]),'NT K Nearest Neighbor (weight)': max(notears_linear_dict_scores["knn"]), 'NT K Nearest Neighbor (distance)': max(notears_linear_dict_scores["knn_d"])}
    #top_learned_linear = max(sim_linear_workflows, key=sim_linear_workflows.get)
    #print("Learned world - Linear problem, Prediction: ", top_learned_linear)

    #real_nonlinear_workflows = {'Decision Tree (gini)': max(real_nonlinear_dt_scores),
    #                         'Decision Tree (entropy)': max(real_nonlinear_dt_entropy_scores),
    #                         'Random Forest (gini)': max(real_nonlinear_rf_scores),
    #                         'Random Forest (entropy)': max(real_nonlinear_rf_entropy_scores),
    #                         'Logistic Regression (none)': max(real_nonlinear_lr_scores),
    #                         'Logistic Regression (l1)': max(real_nonlinear_lr_l1_scores),
    #                         'Logistic Regression (l2)': max(real_nonlinear_lr_l2_scores),
    #                         'Logistic Regression (elasticnet)': max(real_nonlinear_lr_elastic_scores),
    #                         'Naive Bayes (bernoulli)': max(real_nonlinear_gb_scores),
    #                         'Naive Bayes (multinomial)': max(real_nonlinear_gb_multi_scores),
    #                         'Naive Bayes (gaussian)': max(real_nonlinear_gb_gaussian_scores),
    #                         'Naive Bayes (complement)': max(real_nonlinear_gb_complement_scores),
    #                         'Support Vector Machine (sigmoid)': max(real_nonlinear_svm_scores),
    #                         'Support Vector Machine (polynomial)': max(real_nonlinear_svm_poly_scores),
    #                         'Support Vector Machine (rbf)': max(real_nonlinear_svm_rbf_scores),
    #                         'K Nearest Neighbor (uniform)': max(real_nonlinear_knn_scores),
    #                         'K Nearest Neighbor (distance)': max(real_nonlinear_knn_distance_scores)}
    #top_real_nonlinear = max(real_nonlinear_workflows, key=real_nonlinear_workflows.get)
    #print("Real world - Nonlinear problem, Prediction: ", top_real_nonlinear)
    #sim_nonlinear_workflows = {'BN Decision Tree (gini)': max(bnlearn_nonlinear_dict_scores["dt"]),
    #                        'BN Decision Tree (entropy)': max(bnlearn_nonlinear_dict_scores["dt_e"]),
    #                        'NT Decision Tree (gini)': max(notears_nonlinear_dict_scores["dt"]),
    #                        'NT Decision Tree (entropy)': max(notears_nonlinear_dict_scores["dt_e"]),
    #                        'BN Random Forest (gini)': max(bnlearn_nonlinear_dict_scores["rf"]),
    #                        'BN Random Forest (entropy)': max(bnlearn_nonlinear_dict_scores["rf_e"]),
    #                        'NT Random Forest (gini)': max(notears_nonlinear_dict_scores["rf"]),
    #                        'NT Random Forest (entropy)': max(notears_nonlinear_dict_scores["rf_e"]),
    #                        'BN Logistic Regression (none)': max(bnlearn_nonlinear_dict_scores["lr"]),
    #                        'BN Logistic Regression (l1)': max(bnlearn_nonlinear_dict_scores["lr_l1"]),
    #                        'BN Logistic Regression (l2)': max(bnlearn_nonlinear_dict_scores["lr_l2"]),
    #                        'BN Logistic Regression (elastic)': max(bnlearn_nonlinear_dict_scores["lr_e"]),
    #                        'NT Logistic Regression (none)': max(notears_nonlinear_dict_scores["lr"]),
    #                        'NT Logistic Regression (l1)': max(notears_nonlinear_dict_scores["lr_l1"]),
    #                        'NT Logistic Regression (l2)': max(notears_nonlinear_dict_scores["lr_l2"]),
    #                        'NT Logistic Regression (elastic)': max(notears_nonlinear_dict_scores["lr_e"]),
    #                        'BN Naive Bayes (bernoulli)': max(bnlearn_nonlinear_dict_scores["nb"]),
    #                        'BN Naive Bayes (gaussian)': max(bnlearn_nonlinear_dict_scores["nb_g"]),
    #                        'BN Naive Bayes (multinomial)': max(bnlearn_nonlinear_dict_scores["nb_m"]),
    #                        'BN Naive Bayes (complement)': max(bnlearn_nonlinear_dict_scores["nb_c"]),
    #                        'NT Naive Bayes (bernoulli)': max(notears_nonlinear_dict_scores["nb"]),
    #                        'NT Naive Bayes (gaussian)': max(notears_nonlinear_dict_scores["nb_g"]),
    #                        'NT Naive Bayes (multinomial)': max(notears_nonlinear_dict_scores["nb_m"]),
    #                        'NT Naive Bayes (complement)': max(notears_nonlinear_dict_scores["nb_c"]),
    #                        'BN Support Vector Machine (sigmoid)': max(bnlearn_nonlinear_dict_scores["svm"]),
    #                        'BN Support Vector Machine (polynomial)': max(bnlearn_nonlinear_dict_scores["svm_po"]),
    #                        'BN Support Vector Machine (rbf)': max(bnlearn_nonlinear_dict_scores["svm_r"]),
    #                        'NT Support Vector Machine (sigmoid)': max(notears_nonlinear_dict_scores["svm"]),
    #                        'NT Support Vector Machine (polynomial)': max(notears_nonlinear_dict_scores["svm_po"]),
    #                        'NT Support Vector Machine (rbf)': max(notears_nonlinear_dict_scores["svm_r"]),
    #                        'BN K Nearest Neighbor (weight)': max(bnlearn_nonlinear_dict_scores["knn"]),
    #                        'BN K Nearest Neighbor (distance)': max(bnlearn_nonlinear_dict_scores["knn_d"]),
    #                        'NT K Nearest Neighbor (weight)': max(notears_nonlinear_dict_scores["knn"]),
    #                        'NT K Nearest Neighbor (distance)': max(notears_nonlinear_dict_scores["knn_d"])}
    #top_learned_nonlinear = max(sim_nonlinear_workflows, key=sim_nonlinear_workflows.get)
    #print("Learned world - Nonlinear problem, Prediction: ", top_learned_nonlinear)

    #real_sparse_workflows = {'Decision Tree (gini)': max(real_sparse_dt_scores),
    #                         'Decision Tree (entropy)': max(real_sparse_dt_entropy_scores),
    #                         'Random Forest (gini)': max(real_sparse_rf_scores),
    #                         'Random Forest (entropy)': max(real_sparse_rf_entropy_scores),
    #                         'Logistic Regression (none)': max(real_sparse_lr_scores),
    #                         'Logistic Regression (l1)': max(real_sparse_lr_l1_scores),
    #                         'Logistic Regression (l2)': max(real_sparse_lr_l2_scores),
    #                         'Logistic Regression (elasticnet)': max(real_sparse_lr_elastic_scores),
    #                         'Naive Bayes (bernoulli)': max(real_sparse_gb_scores),
    #                         'Naive Bayes (multinomial)': max(real_sparse_gb_multi_scores),
    #                         'Naive Bayes (gaussian)': max(real_sparse_gb_gaussian_scores),
    #                         'Naive Bayes (complement)': max(real_sparse_gb_complement_scores),
    #                         'Support Vector Machine (sigmoid)': max(real_sparse_svm_scores),
    #                         'Support Vector Machine (polynomial)': max(real_sparse_svm_poly_scores),
    #                         'Support Vector Machine (rbf)': max(real_sparse_svm_rbf_scores),
    #                         'K Nearest Neighbor (uniform)': max(real_sparse_knn_scores),
    #                         'K Nearest Neighbor (distance)': max(real_sparse_knn_distance_scores)}
    #top_real_sparse = max(real_sparse_workflows, key=real_sparse_workflows.get)
    #print("Real world - Sparse problem, Prediction: ", top_real_sparse)
    #sim_sparse_workflows = {'BN Decision Tree (gini)': max(bnlearn_sparse_dict_scores["dt"]),
    #                        'BN Decision Tree (entropy)': max(bnlearn_sparse_dict_scores["dt_e"]),
    #                        'NT Decision Tree (gini)': max(notears_sparse_dict_scores["dt"]),
    #                        'NT Decision Tree (entropy)': max(notears_sparse_dict_scores["dt_e"]),
    #                        'BN Random Forest (gini)': max(bnlearn_sparse_dict_scores["rf"]),
    #                        'BN Random Forest (entropy)': max(bnlearn_sparse_dict_scores["rf_e"]),
    #                        'NT Random Forest (gini)': max(notears_sparse_dict_scores["rf"]),
    #                        'NT Random Forest (entropy)': max(notears_sparse_dict_scores["rf_e"]),
    #                        'BN Logistic Regression (none)': max(bnlearn_sparse_dict_scores["lr"]),
    #                        'BN Logistic Regression (l1)': max(bnlearn_sparse_dict_scores["lr_l1"]),
    #                        'BN Logistic Regression (l2)': max(bnlearn_sparse_dict_scores["lr_l2"]),
    #                        'BN Logistic Regression (elastic)': max(bnlearn_sparse_dict_scores["lr_e"]),
    #                        'NT Logistic Regression (none)': max(notears_sparse_dict_scores["lr"]),
    #                        'NT Logistic Regression (l1)': max(notears_sparse_dict_scores["lr_l1"]),
    #                        'NT Logistic Regression (l2)': max(notears_sparse_dict_scores["lr_l2"]),
    #                        'NT Logistic Regression (elastic)': max(notears_sparse_dict_scores["lr_e"]),
    #                        'BN Naive Bayes (bernoulli)': max(bnlearn_sparse_dict_scores["nb"]),
    #                        'BN Naive Bayes (gaussian)': max(bnlearn_sparse_dict_scores["nb_g"]),
    #                        'BN Naive Bayes (multinomial)': max(bnlearn_sparse_dict_scores["nb_m"]),
    #                        'BN Naive Bayes (complement)': max(bnlearn_sparse_dict_scores["nb_c"]),
    #                        'NT Naive Bayes (bernoulli)': max(notears_sparse_dict_scores["nb"]),
    #                        'NT Naive Bayes (gaussian)': max(notears_sparse_dict_scores["nb_g"]),
    #                        'NT Naive Bayes (multinomial)': max(notears_sparse_dict_scores["nb_m"]),
    #                        'NT Naive Bayes (complement)': max(notears_sparse_dict_scores["nb_c"]),
    #                        'BN Support Vector Machine (sigmoid)': max(bnlearn_sparse_dict_scores["svm"]),
    #                        'BN Support Vector Machine (polynomial)': max(bnlearn_sparse_dict_scores["svm_po"]),
    #                        'BN Support Vector Machine (rbf)': max(bnlearn_sparse_dict_scores["svm_r"]),
    #                        'NT Support Vector Machine (sigmoid)': max(notears_sparse_dict_scores["svm"]),
    #                        'NT Support Vector Machine (polynomial)': max(notears_sparse_dict_scores["svm_po"]),
    #                        'NT Support Vector Machine (rbf)': max(notears_sparse_dict_scores["svm_r"]),
    #                        'BN K Nearest Neighbor (weight)': max(bnlearn_sparse_dict_scores["knn"]),
    #                        'BN K Nearest Neighbor (distance)': max(bnlearn_sparse_dict_scores["knn_d"]),
    #                        'NT K Nearest Neighbor (weight)': max(notears_sparse_dict_scores["knn"]),
    #                        'NT K Nearest Neighbor (distance)': max(notears_sparse_dict_scores["knn_d"])}
    #top_learned_sparse = max(sim_sparse_workflows, key=sim_sparse_workflows.get)
    #print("Learned world - Sparse problem, Prediction: ", top_learned_sparse)

    #real_dimension_workflows = {'Decision Tree (gini)': max(real_dimension_dt_scores),
    #                         'Decision Tree (entropy)': max(real_dimension_dt_entropy_scores),
    #                         'Random Forest (gini)': max(real_dimension_rf_scores),
    #                         'Random Forest (entropy)': max(real_dimension_rf_entropy_scores),
    #                         'Logistic Regression (none)': max(real_dimension_lr_scores),
    #                         'Logistic Regression (l1)': max(real_dimension_lr_l1_scores),
    #                         'Logistic Regression (l2)': max(real_dimension_lr_l2_scores),
    #                         'Logistic Regression (elasticnet)': max(real_dimension_lr_elastic_scores),
    #                         'Naive Bayes (bernoulli)': max(real_dimension_gb_scores),
    #                         'Naive Bayes (multinomial)': max(real_dimension_gb_multi_scores),
    #                         'Naive Bayes (gaussian)': max(real_dimension_gb_gaussian_scores),
    #                         'Naive Bayes (complement)': max(real_dimension_gb_complement_scores),
    #                         'Support Vector Machine (sigmoid)': max(real_dimension_svm_scores),
    #                         'Support Vector Machine (polynomial)': max(real_dimension_svm_poly_scores),
    #                         'Support Vector Machine (rbf)': max(real_dimension_svm_rbf_scores),
    #                         'K Nearest Neighbor (uniform)': max(real_dimension_knn_scores),
    #                         'K Nearest Neighbor (distance)': max(real_dimension_knn_distance_scores)}
    #top_real_dimension = max(real_dimension_workflows, key=real_dimension_workflows.get)
    #print("Real world - Dimensional problem, Prediction: ", top_real_dimension)
    #sim_dimension_workflows = {'BN Decision Tree (gini)': max(bnlearn_dimension_dict_scores["dt"]),
    #                        'BN Decision Tree (entropy)': max(bnlearn_dimension_dict_scores["dt_e"]),
    #                        'NT Decision Tree (gini)': max(notears_dimension_dict_scores["dt"]),
    #                        'NT Decision Tree (entropy)': max(notears_dimension_dict_scores["dt_e"]),
    #                        'BN Random Forest (gini)': max(bnlearn_dimension_dict_scores["rf"]),
    #                        'BN Random Forest (entropy)': max(bnlearn_dimension_dict_scores["rf_e"]),
    #                        'NT Random Forest (gini)': max(notears_dimension_dict_scores["rf"]),
    #                        'NT Random Forest (entropy)': max(notears_dimension_dict_scores["rf_e"]),
    #                        'BN Logistic Regression (none)': max(bnlearn_dimension_dict_scores["lr"]),
    #                        'BN Logistic Regression (l1)': max(bnlearn_dimension_dict_scores["lr_l1"]),
    #                        'BN Logistic Regression (l2)': max(bnlearn_dimension_dict_scores["lr_l2"]),
    #                        'BN Logistic Regression (elastic)': max(bnlearn_dimension_dict_scores["lr_e"]),
    #                        'NT Logistic Regression (none)': max(notears_dimension_dict_scores["lr"]),
    #                        'NT Logistic Regression (l1)': max(notears_dimension_dict_scores["lr_l1"]),
    #                        'NT Logistic Regression (l2)': max(notears_dimension_dict_scores["lr_l2"]),
    #                        'NT Logistic Regression (elastic)': max(notears_dimension_dict_scores["lr_e"]),
    #                        'BN Naive Bayes (bernoulli)': max(bnlearn_dimension_dict_scores["nb"]),
    #                        'BN Naive Bayes (gaussian)': max(bnlearn_dimension_dict_scores["nb_g"]),
    #                        'BN Naive Bayes (multinomial)': max(bnlearn_dimension_dict_scores["nb_m"]),
    #                        'BN Naive Bayes (complement)': max(bnlearn_dimension_dict_scores["nb_c"]),
    #                        'NT Naive Bayes (bernoulli)': max(notears_dimension_dict_scores["nb"]),
    #                        'NT Naive Bayes (gaussian)': max(notears_dimension_dict_scores["nb_g"]),
    #                        'NT Naive Bayes (multinomial)': max(notears_dimension_dict_scores["nb_m"]),
    #                        'NT Naive Bayes (complement)': max(notears_dimension_dict_scores["nb_c"]),
    #                        'BN Support Vector Machine (sigmoid)': max(bnlearn_dimension_dict_scores["svm"]),
    #                        'BN Support Vector Machine (polynomial)': max(bnlearn_dimension_dict_scores["svm_po"]),
    #                        'BN Support Vector Machine (rbf)': max(bnlearn_dimension_dict_scores["svm_r"]),
    #                        'NT Support Vector Machine (sigmoid)': max(notears_dimension_dict_scores["svm"]),
    #                        'NT Support Vector Machine (polynomial)': max(notears_dimension_dict_scores["svm_po"]),
    #                        'NT Support Vector Machine (rbf)': max(notears_dimension_dict_scores["svm_r"]),
    #                        'BN K Nearest Neighbor (weight)': max(bnlearn_dimension_dict_scores["knn"]),
    #                        'BN K Nearest Neighbor (distance)': max(bnlearn_dimension_dict_scores["knn_d"]),
    #                        'NT K Nearest Neighbor (weight)': max(notears_dimension_dict_scores["knn"]),
    #                        'NT K Nearest Neighbor (distance)': max(notears_dimension_dict_scores["knn_d"])}
    #top_learned_dimension = max(sim_dimension_workflows, key=sim_dimension_workflows.get)
    #print("Learned world - Dimensional problem, Prediction: ", top_learned_dimension)

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
