from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def train_models(X_train, y_train):
    models = {}

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    models['naive_bayes'] = nb

    svm = LinearSVC()
    svm.fit(X_train, y_train)
    models['svm'] = svm

    lr = LogisticRegression(max_iter=300, C=3)
    lr.fit(X_train, y_train)
    models['logistic_regression'] = lr

    return models