import numpy as np

from sklearn import tree
from common import print_prediction

def model_tree_fit(X_train,y_train,X_test,y_test,file_out):
#    print ('\n\t' + 'TRAIN\t |\tTEST')
    dtc = tree.DecisionTreeClassifier(random_state=228)
    dtc = dtc.fit(X_train, y_train)

    predictions = dtc.predict(X_train)
    predictions_test = dtc.predict(X_test)
    print_prediction(predictions, y_train, predictions_test, y_test,file_out)
    return dtc

from sklearn.ensemble import RandomForestClassifier

def model_RandomForest_fit (file_out, X_train,y_train,X_test,y_test, \
                            n_estimators=1000, max_depth=None, \
                            flag_graph=0):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=228)
    rf = rf.fit(X_train, y_train)

    predictions = rf.predict(X_train)
    predictions_test = rf.predict(X_test)
    if flag_graph == 0:
        print_prediction(predictions, y_train, predictions_test, y_test, file_out)
    
    return rf

import matplotlib.pyplot as plt

def RandomForest_graph(n_estimators, X_train, y_train, X_test, y_test, name):
    x_axis = np.linspace(1, n_estimators, num=n_estimators)
    print (x_axis.shape)

    plt.figure(figsize=(10, 7))

    y_prec = np.zeros(x_axis.shape)
    y_rec  =  np.zeros(x_axis.shape)
    print (y_prec.shape)
    for n in range(n_estimators):
        print (n+1)
        rf = model_RandomForest_fit(X_train,y_train,X_test,y_test, \
                                        n_estimators=n+1, max_depth=None)

        y_prec[n] = precision_score(y_test, rf.predict(X_test))
        y_rec[n] = recall_score(y_test, rf.predict(X_test))
        
        print ('finished')

    print (y_prec)
    plt.plot(x_axis, y_prec, color='b', lw=3, alpha=0.7, label='Precision')
    plt.plot(x_axis, y_rec, color='r', lw=3, alpha=0.7, label='Recall')
    plt.title('Random Forest')
    plt.xlabel('number of estimators')
    plt.ylabel('Metric, %')
    plt.legend(loc='upper right')
    plt.grid(True)

    path = 'Graphs/RandomForest_form_'
    plt.savefig(path + name + str(n_estimators) + '.png')
