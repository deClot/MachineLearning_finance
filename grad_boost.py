import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from common import print_prediction

def model_Ada (X_train,y_train,X_test, y_test,file_out, n_estimators, depth=1):
    abdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth, \
                                                     random_state=228),
                          algorithm="SAMME",
                          n_estimators=n_estimators,
                          random_state=228)
    abdt.fit(X_train, y_train)

    predictions = abdt.predict(X_train)
    predictions_test = abdt.predict(X_test)

    print_prediction(predictions, y_train, predictions_test, y_test, file_out)
    
    return abdt

from sklearn.ensemble import GradientBoostingClassifier

def model_GradBoost (X_train,y_train,X_test, y_test,file_out, n_estimators, depth=1):
    gb = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=depth,
                                 random_state=228)
    gb.fit(X_train, y_train)

    predictions = gb.predict(X_train)
    predictions_test = gb.predict(X_test)

    print_prediction(predictions, y_train, predictions_test, y_test, file_out)
    
    return gb

from catboost import CatBoostClassifier

def model_CatBoost(X_train,y_train,X_test, y_test,file_out, iterations, \
                   learn_rate = 0.3, depth=2):
    cb = CatBoostClassifier(iterations=iterations, learning_rate=learn_rate,
                         depth=depth, loss_function='MultiClassOneVsAll',
                         classes_count=2, random_seed=228)
    cb.fit(X_train, y_train, verbose=False)

    predictions = cb.predict(X_train)
    predictions_test = cb.predict(X_test)

    print_prediction(predictions, y_train, predictions_test, y_test, file_out)
    
    return cb
