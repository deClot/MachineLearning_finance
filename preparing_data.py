import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def print_count_data(y,y_test):
    print ('\t\tTRAIN DATA\tTEST DATA')
    print ('Negative\t', np.sum(y==0),'  ',
           round(100*np.sum(y==0)/y.shape[0],2),'%'
           '\t',np.sum(y_test==0),'  ',
           round(100*np.sum(y_test==0)/y_test.shape[0],2),'%'
           '\nPositive\t', np.sum(y==1),' ',
           round(100*np.sum(y==1)/y.shape[0],2),'%'
           '\t',np.sum(y_test==1),' ',
           round(100*np.sum(y_test==1)/y_test.shape[0],2),'%')


def load_data (y_lab = 'positive'):
    train_data = pd.read_csv('finance_train.csv')
    test_data = pd.read_csv('finance_test.csv')
    
    #print(train_data.head(5))
    print (train_data['Time'].value_counts().shape)

  #  from pandas.plotting import scatter_matrix
  #  scatter_matrix(train_data, alpha=0.05, figsize=(10, 10));
 
    X = train_data.drop(['Financial Distress'], axis='columns').values
    y = train_data['Financial Distress'].values

    X_test = test_data.drop(['Financial Distress'], axis='columns').values
    y_test = test_data['Financial Distress'].values

#    print (X[:,:3])
    if y_lab =='negative':
        y = np.where(y < -0.5, 1, 0)
        y_test = np.where(y_test < -0.5, 1, 0)
    else:
        y = np.where(y >= -0.5, 1, 0)
        y_test = np.where(y_test >= -0.5, 1, 0)
        
    
    print_count_data(y, y_test)

    print (train_data.head())
    
    return X,y,X_test,y_test

def preprocessing_data(X_train,X_test):
    #rint (train_data['Time'].value_counts())

    X_train_categ = np.hstack((X_train[:,:2],X_train[:,81:82]))
    X_test_categ  = np.hstack((X_test[:,:2],X_test[:,81:82]))

    #print (X_train_categ)
    #print (X_train_categ.shape)
 
    X_train_real = np.hstack((X_train[:,2:81],X_train[:,82:]))
    X_test_real  = np.hstack((X_test[:,2:81],X_test[:,82:]))

    ss = StandardScaler()
    X_train_real = ss.fit_transform(X_train_real)
    X_test_real  = ss.transform(X_test_real)

    ######
    #             DO SOMETHING WITH CATEG

    #############################

    X_train_ready = np.hstack((X_train_categ[:,:2],X_train_real[:,:79],\
                               X_train_categ[:,2:],X_train_real[:,79:] ))
    X_test_ready = np.hstack((X_test_categ[:,:2],X_test_real[:,:79],\
                               X_test_categ[:,2:],X_test_real[:,79:] ))

    return X_train_ready, X_test_ready
    
