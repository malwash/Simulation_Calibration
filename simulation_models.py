from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

def decision_tree(x_train, y_train, x_test, y_test):
    clf = DecisionTreeClassifier(criterion='gini')
    clf = clf.fit(x_train, y_train)
    #y_pred = clf.predict(x_test)
    #return metrics.accuracy_score(y_test,y_pred)
    return cross_val_score(clf, x_test, y_test, cv=10)

def decision_tree_entropy(x_train, y_train, x_test, y_test):
    clf = DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(x_train, y_train)
    #y_pred = clf.predict(x_test)
    #return metrics.accuracy_score(y_test,y_pred)
    return cross_val_score(clf, x_test, y_test, cv=10)

def random_forest(x_train, y_train, x_test, y_test):
    rf = RandomForestClassifier(criterion='gini')
    rf = rf.fit(x_train, y_train)
    #y_pred = rf.predict(x_test)
    #return metrics.accuracy_score(y_test,y_pred)
    return cross_val_score(rf, x_test, y_test, cv=10)

def random_forest_entropy(x_train, y_train, x_test, y_test):
    rf = RandomForestClassifier(criterion='entropy')
    rf = rf.fit(x_train, y_train)
    #y_pred = rf.predict(x_test)
    #return metrics.accuracy_score(y_test,y_pred)
    return cross_val_score(rf, x_test, y_test, cv=10)

def logistic_regression(x_train, y_train, x_test, y_test):
    lr = LogisticRegression(penalty='none')
    lr = lr.fit(x_train, y_train)
    #y_pred = lr.predict(x_test)
    return cross_val_score(lr, x_test, y_test, cv=10)

def logistic_regression_l1(x_train, y_train, x_test, y_test):
    lr = LogisticRegression(penalty='l1', solver='liblinear', l1_ratio=1)
    lr = lr.fit(x_train, y_train)
    #y_pred = lr.predict(x_test)
    return cross_val_score(lr, x_test, y_test, cv=10)

def logistic_regression_l2(x_train, y_train, x_test, y_test):
    lr = LogisticRegression(penalty='l2')
    lr = lr.fit(x_train, y_train)
    #y_pred = lr.predict(x_test)
    return cross_val_score(lr, x_test, y_test, cv=10)

def logistic_regression_elastic(x_train, y_train, x_test, y_test):
    lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
    lr = lr.fit(x_train, y_train)
    #y_pred = lr.predict(x_test)
    return cross_val_score(lr, x_test, y_test, cv=10)

def naive_bayes(x_train, y_train, x_test, y_test):
    gnb = BernoulliNB()
    gnb = gnb.fit(x_train, y_train)
    #y_pred = gnb.predict(x_test)
    return cross_val_score(gnb, x_test, y_test, cv=10)

def naive_bayes_gaussian(x_train, y_train, x_test, y_test):
    gnb = GaussianNB()
    gnb = gnb.fit(x_train, y_train)
    #y_pred = gnb.predict(x_test)
    return cross_val_score(gnb, x_test, y_test, cv=10)

def naive_bayes_multinomial(x_train, y_train, x_test, y_test):
    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(x_train)
    X_test_minmax = min_max_scaler.transform(x_test)
    gnb = MultinomialNB()
    gnb = gnb.fit(X_train_minmax, y_train)
    #y_pred = gnb.predict(X_test_minmax)
    return cross_val_score(gnb, X_test_minmax, y_test, cv=10)

def naive_bayes_complement(x_train, y_train, x_test, y_test):
    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(x_train)
    X_test_minmax = min_max_scaler.transform(x_test)
    gnb = ComplementNB() #ValueError: Negative values in data passed to ComplementNB (input X)
    gnb = gnb.fit(X_train_minmax, y_train)
    #y_pred = gnb.predict(X_test_minmax)
    return cross_val_score(gnb, X_test_minmax, y_test, cv=10)

def support_vector_machines(x_train, y_train, x_test, y_test):
    clf = svm.SVC(kernel="sigmoid")
    clf = clf.fit(x_train, y_train)
    #y_pred = clf.predict(x_test)
    return cross_val_score(clf, x_test, y_test, cv=10)

def support_vector_machines_linear(x_train, y_train, x_test, y_test):
    clf = svm.SVC(kernel="linear")
    clf = clf.fit(x_train, y_train)
    #y_pred = clf.predict(x_test)
    return cross_val_score(clf, x_test, y_test, cv=10)

def support_vector_machines_poly(x_train, y_train, x_test, y_test):
    clf = svm.SVC(kernel="poly")
    clf = clf.fit(x_train, y_train)
    #y_pred = clf.predict(x_test)
    return cross_val_score(clf, x_test, y_test, cv=10)

def support_vector_machines_rbf(x_train, y_train, x_test, y_test):
    clf = svm.SVC(kernel="rbf")
    clf = clf.fit(x_train, y_train)
    #y_pred = clf.predict(x_test)
    return cross_val_score(clf, x_test, y_test, cv=10)

def support_vector_machines_precomputed(x_train, y_train, x_test, y_test):
    clf = svm.SVC(kernel="precomputed")
    clf = clf.fit(x_train, y_train)
    #y_pred = clf.predict(x_test)
    return cross_val_score(clf, x_test, y_test, cv=10)

def k_nearest_neighbor(x_train, y_train, x_test, y_test):
    clf = KNeighborsClassifier(weights='uniform')
    clf = clf.fit(x_train, y_train)
    #y_pred = clf.predict(x_test)
    return cross_val_score(clf, x_test, y_test, cv=10)

def k_nearest_neighbor_distance(x_train, y_train, x_test, y_test):
    clf = KNeighborsClassifier(weights='distance')
    clf = clf.fit(x_train, y_train)
    #y_pred = clf.predict(x_test)
    #return metrics.accuracy_score(y_test, y_pred)
    return cross_val_score(clf, x_test, y_test, cv=10)

#def decision_tree():
#    clf = DecisionTreeClassifier(random_state=0)
#    clf = clf.fit(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4])
#    y_pred = clf.predict(x_test)
#    print("(Double-Simulated) Accuracy of bnlearn on Test set (DecisionTree)", metrics.accuracy_score(y_test,y_pred))
#
#    clf = clf.fit(no_tears_sample_train.iloc[:,0:4], no_tears_sample_train.iloc[:,4])
#    y_pred = clf.predict(x_test)
#    print("(Double-Simulated) Accuracy of no_tears on Test set (DecisionTree)", metrics.accuracy_score(y_test,y_pred))

#decision_tree()

#def random_forest():
#    rf = RandomForestClassifier(random_state=0)
#    rf = rf.fit(bn_learn_sample_train.iloc[:,0:4], bn_learn_sample_train.iloc[:,4])
#    y_pred = rf.predict(x_test)
#    print('(Double-Simulated) Accuracy of bn_learn on Test set (RandomForestClassifier):', metrics.accuracy_score(y_test,y_pred), '%.')
#
#    rf = rf.fit(no_tears_sample_train.iloc[:, 0:4], no_tears_sample_train.iloc[:, 4])
#    y_pred = rf.predict(x_test)
#    print('(Double-Simulated) Accuracy of no_tears on Test set (RandomForestClassifier):', metrics.accuracy_score(y_test,y_pred), '%.')

#random_forest()
