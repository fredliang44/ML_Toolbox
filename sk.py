"""Proccess Data"""
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

"""Traning"""
from neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, SVR

""" Visualize"""
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

""" Classifiers """
classifiers = [
    ("GaussianNB", GaussianNB())
    ("MultinomialNB", MultinomialNB())
    ("LogisticRegression", LogisticRegression())
    ("SVC", SVC())
    ("SVR", SVR())
    ("SGD", SGDClassifier()),
    ("ASGD", SGDClassifier(average=True)),
    ("Perceptron", Perceptron()),
    ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                         C=1.0)),
    ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0)),
    ("KNeighbors", KNeighborsClassifier())
    ("MLP", MLPClassifier())
    ("DecisionTree", DecisionTreeClassifier())
    ("AdaBoost", AdaBoostClassifier())

]


class Process(object):
    """docstring for Process."""

    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.random_state = 0.4
        self.test_size = 0.2

    """ Split Dataset"""

    def split(self):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, random_state=self.random_state, test_size=self.test_size)

    """ Scale Part """

    def scale(self):
        scaler = preprocessing.MinMaxScaler().fit(self.data)
        return scaler.transform(self.data)

    """ Fill NaN in datasets """

    def fillnan(self):
        imp = Imputer(missing_values='NaN', strategy='mean')
        imp.fit(self.data)
        return imp.transform(self.data)


class Model(object):
    """docstring for Process."""

    def __init__(self, clf=GaussianNB()):
        self.clf = clf
        self.scale, self.grid_search self.mod_search = False, False, False
        self.X, self.y = None, None
        self.x_label = "x_label"
        self.y_label = "y_label"

    """ Fit model """

    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.mod_search:
            for name, clf in classifiers:
                print("training %s" % name)
                rng = np.random.RandomState(42)
                yy = []
                for i in heldout:
                    yy_ = []
                    for r in range(rounds):
                        X_train, X_test, y_train, y_test = \
                            train_test_split(self.X, selfy, test_size=i, random_state=rng)
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        yy_.append(1 - np.mean(y_pred == y_test))
                    yy.append(np.mean(yy_))
                plt.plot(xx, yy, label=name)
            plt.legend(loc="upper right")
            plt.xlabel("Proportion train")
            plt.ylabel("Test Error Rate")
            plt.show()
        elif self.grid_search:
            self.grid.fit(X, y)
        else:
            self.clf.fit(X, y)

    """ Predict """

    def predict(self, x):
        x_t = x
        return self.clf.predict(x_t)[:, 1]

    """ Predict Prob """

    def predict_prob(self, x):
        x_t = x
        return self.clf.predict_proba(x_t)[:, 1]

    """ Get auc of model """

    def auc(self, x, y_true):
        y_score = self.predict_prob(x)
        return roc_auc_score(y_true, y_score)

    """ Get f1 of model """

    def f1(self, x):
        pass
    """ Show classifier graph """

    def show(self, X_test, y_test):

        x_min = 0
        x_max = 1
        y_min = 0
        y_max = 1

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        h = .01  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

        # Plot also the test points
        grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii] == 0]
        bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii] == 0]
        grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii] == 1]
        bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii] == 1]

        plt.scatter(grade_sig, bumpy_sig, color="b", label="cust")
        plt.scatter(grade_bkg, bumpy_bkg, color="r", label="main")
        plt.legend()
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        plt.show()
