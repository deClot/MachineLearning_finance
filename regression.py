from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from common import print_prediction

import matplotlib.pyplot as plt
import numpy as np

def model_logistic_regression (X_train, y_train, X_dev,y_dev, X_test,y_test,\
                               file_out, flag_degree = True, degree = 2, \
                               C = 1,class_weight=None,out=False):
    if flag_degree == True:
        poly = PolynomialFeatures(degree=degree, interaction_only = False)
        X_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
    else:
        X_poly = X_train
        X_test_poly = X_test

    lr =LogisticRegression(C=C, n_jobs=-1, class_weight=None,\
                           penalty = 'l2', random_state=228)
    lr.fit(X_poly, y_train)

    predictions = lr.predict(X_poly)
    #predictions2 = lr.predict(X_train)
    predictions_test = lr.predict(X_test_poly)

    if X_dev == None:
        predictions_dev = None
        dev = False
    else:
        predictions_dev  = dtc.predict(X_dev)
      
    print_prediction(predictions, y_train, \
                     predictions_dev, y_dev, \
                     predictions_test, y_test,\
                     file_out, dev , out=out)

    return lr,predictions_test

def Logic_regression_graph(X_train, y_train, X_test, y_test,file_out,name,\
                           X_dev= None,y_dev=None,\
                           flag_degree = False, degree = 1, C=1):
    x_axis = np.linspace(1, degree, num=degree)
    #x_axis = np.logspace(-C, 0, base=10,num=C+1)
    
    plt.figure(figsize=(10, 7))

    y_acc      = np.zeros(x_axis.shape)
    y_rec_pos = np.zeros(x_axis.shape)
    y_rec_neg  =  np.zeros(x_axis.shape)
    y_mean = np.zeros(x_axis.shape)
    
    i=0
    for n in x_axis:
        print (n)
        logic,predictions_test = model_logistic_regression (X_train, y_train, X_dev,y_dev,\
                                           X_test,y_test,\
                                           file_out,\
                                           flag_degree = flag_degree,degree = degree, \
                                           C = C)

        all = precision_recall_fscore_support(y_test,predictions_test)
        prec_all, rec_all,f1_all, __ = all
        
        y_rec_pos[i]  = rec_all[0]
        y_rec_neg[i]  = rec_all[1]
        y_mean[i]    = (y_rec_pos[i]+y_rec_neg[i])/2
        
        y_acc[i]      = accuracy_score(y_test,predictions_test)
        i=i+1
        print ('finished')

    '''
    plt.semilogx(x_axis, y_rec_pos, color='b', lw=3, alpha=0.7, label='Recall_positive')
    plt.semilogx(x_axis, y_rec_neg, color='r', lw=3, alpha=0.7, label='Recall_negative')
    plt.semilogx(x_axis, y_mean, color='black', lw=3, alpha=0.7, label='Recall_mean')
    plt.semilogx(x_axis, y_acc, color='g', lw=3, alpha=0.7, label='Accuracy')
    '''
    
    plt.plot(x_axis, y_rec_pos, color='b', lw=3, alpha=0.7, label='Recall_positive')
    plt.plot(x_axis, y_rec_neg, color='r', lw=3, alpha=0.7, label='Recall_negative')
    plt.plot(x_axis, y_mean, color='black', lw=3, alpha=0.7, label='Recall_mean')
    plt.plot(x_axis, y_acc, color='g', lw=3, alpha=0.7, label='Accuracy')
    
    plt.title('Logic Regression')
    plt.xlabel('degree')
    plt.ylabel('Metric')
    plt.legend(loc='upper right')
    plt.grid(True)

    path = 'Graphs/LogicRegr_for_'
    plt.savefig(path +'degree='+str(degree)+'C=_'+str(C)+name+ '.png')
