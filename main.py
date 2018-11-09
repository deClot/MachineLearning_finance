import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix

import preparing_data
from trees_classifiers import model_RandomForest_fit, model_tree_fit
from nn_classifier import get_nn_model, loss_graph, acc_graph

X,y,X_test,y_test = preparing_data.load_data(y_lab = 'negative')

X_norm, X_test_norm = preparing_data.preprocessing_data(X,X_test)

X_train,X_dev, y_train, y_dev = train_test_split(X_norm,y, random_state = 228)
#X_train = X
#y_train = y
#preparing_data.print_count_data(y_train,y_dev)
#preparing_data.print_count_data(y_train,y_test)

#dtc = model_tree_fit(X_train,y_train,X_dev, y_dev)

#rf = model_RandomForest_fit(X_train,y_train,X_dev,y_dev, \
#                                        n_estimators=150, max_depth=None)
#RandomForest_graph(100,X_train,y_train,X_dev,y_dev, 'dev')

nn_model = get_nn_model(X_train.shape[1])
logs = nn_model.fit(X_train,y_train, epochs=100, batch_size=32,\
                    validation_data=(X_dev, y_dev), shuffle=True)

#loss_graph(epochs, logs.history.get('loss'), logs.history.get('val_loss'))
#acc_graph(epochs, logs.history.get('acc'), logs.history.get('val_acc'))
