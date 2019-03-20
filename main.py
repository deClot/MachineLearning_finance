import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

import preparing_data
import pca_func
import common
from trees_classifiers import model_RandomForest_fit, model_tree_fit
from nn_classifier import get_nn_model, loss_graph, fit_nn_model
from grad_boost import model_Ada, model_GradBoost, model_CatBoost

#  -------  Read data from CSV --------
train_data = pd.read_csv('finance_train.csv')
test_data = pd.read_csv('finance_test.csv')

#  ------- Cutting outliers for real features  --------
train_data, test_data = preparing_data.features_cut(train_data,test_data,\
                                                    plot=False)
#  ------- Separate data on X,y  --------
X,y,X_test,y_test,y_ini,y_test_ini,y_pos,y_test_pos = \
    preparing_data.X_y_data(train_data,test_data,\
                            y_lab = 'negative')

#  ------- Norm X - for real and catecorical features  --------
X_norm, X_test_norm = preparing_data.norm_data(X,X_test,y_pos,y_test_pos,\
                                               real='standart',\
                                               categ='target',\
                                               all = True)
#  ------- Used it for cros-val  --------
#X_train, X_dev, y_train, y_dev = train_test_split(X_norm,y, random_state = 228)

#  ------- Used it for study influence of sampling  --------
#X_replaced, y_replaced  = RandomOverSampler(random_state=228).fit_resample(X_train, y_train)
X_replaced, y_replaced = SMOTE(random_state=228, k_neighbors=5).fit_resample(X_train, y_train)
#X_replaced, y_replaced = ADASYN(random_state=228, n_neighbors=4).fit_resample(X_norm, y_train)X_train = X_replaced

X_train = X_replaced
y_train = y_replaced

#  ------- Used it for study influence of pca  --------
X_pca, X_test_pca = pca_func.pca(X_train,X_test, 55)
X_train = X_pca
X_test = X_test_pca


def main_body (X_train,y_train,X_dev, y_dev):
    file_out = open('result.res', 'w')
    '''
    file_out.write('----Tree----\n')
    dtc = model_tree_fit(X_train,y_train,X_dev, y_dev,file_out)
    
    file_out.write('--Random Forest--\n')
    rf = model_RandomForest_fit(file_out,X_train,y_train,X_dev,y_dev, \
                            n_estimators=100, max_depth=None,flag_graph=0)
    #RandomForest_graph(100,X_train,y_train,X_dev,y_dev, 'dev')

    file_out.write('--Neural Network--\n')
    nn = fit_nn_model(100, X_train,y_train, X_dev, y_dev,file_out, batch_size = 32)
    '''
    file_out.write('----AdaBoost----\n')
    abdt = model_Ada(X_train,y_train,X_dev, y_dev,file_out,n_estimators=64,depth=1)
    
    file_out.write('----GradBoost----\n')
    gb  = model_GradBoost(X_train,y_train,X_dev, y_dev,file_out,\
                          n_estimators=64,depth=1)
    
    
    file_out.write('----CatBoost----\n')
    cb = model_CatBoost(X_train,y_train,X_test, y_test,file_out, iterations=232, \
                        learn_rate = 0.5, depth=5)
    #preparing_data.draw_objects(X_dev,y_dev,cb)
    

    
    file_out.close()
####### X_test_norm, y_test   #X_test_pca, y_test
####### X_train_dev, y_train_dev
####### X_dev, y_dev
main_body(X_train,y_train, X_dev, y_dev)

