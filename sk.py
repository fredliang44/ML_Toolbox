from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Processing
from sklearn import preprocessing


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

    def scal(self):
        scaler = preprocessing.MinMaxScaler().fit(self.data)
        return scaler.transform(self.data)
