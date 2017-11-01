"""Proccess Data"""
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

"""Traning"""
from neighbors import KNeighborsClassifier, neighbors.KNeighborsRegressor
from sklearn.ensemble import AdaBoostClassifier, ensemble.AdaBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, SVR


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

    def fillnan(self):
        imp = Imputer(missing_values='NaN', strategy='mean')
        imp.fit(self.data)
        return imp.transform(self.data)


class Model(object):
    """docstring for Process."""

    def __init__(self, clf=GaussianNB):
        self.clf = clf
        self.scale = False
        self.grid_search = False
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict_prob(self, x):
        x_t = x
        return self.clf.predict_proba(x_t)[:, 1]

    def auc(self, x, y_true):
        y_score = self.predict_prob(x)
        return roc_auc_score(y_true, y_score)
