from imports import *


def train_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    


# clf = LinearSVC()

# clf.fit(X_train, y_train)


# distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
# rscv = RandomizedSearchCV(clf, distributions, random_state=0)
# random_search = rscv.fit(X_train,y_train)

# model_linearSVC = random_search.best_estimator_

# model = StackingClassifier([('linearSVC', model_linearSVC),
#                             ('randomForest', model_randomForest)],
#                              final_estimator=model_linearSVC)


# model.fit(X_train, y_train)

# model.score(X_test, y_test)