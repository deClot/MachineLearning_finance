import pandas as pd
import numpy as np

def features_cut (train_data, test_data, plot = False):
    '''Cut real features - set boundary for some real features and implement changing feature region'''
    dic = {'x8': (50, -3), 'x12': (120,-57), 'x15': (2000,-1), 'x16': (10,-22),\
           'x17':(2230,0), 'x19': (2.6,0), 'x22': (300,0), 'x25': (16000,-4000),
           'x27':(1000,-1800), 'x31':(5200000,0), 'x32':(300,0),'x35':(80000,0),\
           'x38':(16,0), 'x39':(13,-1), 'x44':(500,-57),'x47':(31,0),\
           'x48':(22000,0), 'x52':(8,-21),'x57':(120,-1), 'x59':(2,-8)}
    for key in dic.keys():
        (max, min) = dic[key]
        train_data, test_data = feature_changing(train_data,\
                                                 test_data,\
                                                 key, max, min)
        if plot == True:
            plot_real(train_data, key, 'train_without_outliners_')
            plot_real(test_data, key, 'test_without_outliners_')

    return train_data, test_data

def feature_changing (train_data, test_data, name, value1, value2):
    '''Change region for real features. Change ouliers on mean value'''
    mean_fare = train_data.loc[train_data[name]<value1].loc[train_data[name]>value2][name].mean()
    
    for k,v in train_data[name].items():
        if float(v) > value1 or v<value2:
            train_data[name] = train_data[name].replace({k:v}, mean_fare)

    for k,v in test_data[name].items():
        if float(v) > value1 or v<value2:
            test_data[name] = test_data[name].replace({k:v}, mean_fare)

    return train_data, test_data

import matplotlib.pyplot as plt

def plot_real(data, name, comment):
    plt.figure(figsize=(8, 5))
    plt.hist(data[name], 20, edgecolor='white')
    plt.xlabel(name)
    plt.grid(True)

    path = 'Graphs/Real_feature__'
    plt.savefig(path +comment+str(name)+'.png')

def print_count_data(y,y_test):
    '''Сonsole output info exsiting class ratio, in %'''
    print ('\t\tTRAIN DATA\tTEST DATA')
    print ('Negative\t', np.sum(y==0),'  ',
           round(100*np.sum(y==0)/y.shape[0],2),'%'
           '\t',np.sum(y_test==0),'  ',
           round(100*np.sum(y_test==0)/y_test.shape[0],2),'%'
           '\nPositive\t', np.sum(y==1),' ',
           round(100*np.sum(y==1)/y.shape[0],2),'%'
           '\t',np.sum(y_test==1),' ',
           round(100*np.sum(y_test==1)/y_test.shape[0],2),'%')


def X_y_data (train_data, test_data, y_lab = 'positive'):
    '''y_lab - marker for class, which will be positive. y_lab = 'negative' - y=0 will be changed to y=1'''
    X = train_data.drop(['Financial Distress'], axis='columns').values
    y_ini = train_data['Financial Distress'].values

    X_test = test_data.drop(['Financial Distress'], axis='columns').values
    y_test_ini = test_data['Financial Distress'].values

    #check changing of labels
    if y_lab =='negative':
        y_pos = np.where(y_ini >= -0.5, 1, 0)
        y = np.where(y_ini < -0.5, 1, 0)
        y_test_pos = np.where(y_test_ini >= -0.5, 1, 0)
        y_test = np.where(y_test_ini < -0.5, 1, 0)
    else:
        y_pos = np.where(y_ini >= -0.5, 1, 0)
        y      = np.where(y_ini >= -0.5, 1, 0)
        y_test_pos = np.where(y_test_ini >= -0.5, 1, 0)
        y_test = np.where(y_test_ini >= -0.5, 1, 0)
    
    print_count_data(y, y_test)
    print (train_data.head())

    return X,y,X_test,y_test,y_ini,y_test_ini,y_pos,y_test_pos

from sklearn.preprocessing import StandardScaler, OneHotEncoder
import category_encoders as ce
from sklearn import preprocessing

def norm_data(X_train,X_test,y_train,y_test,real=None,categ=None, all = True):
    '''Preprocessing features'''
    #  -------------   Split data on real and categ   -----------------
    X_train_categ = np.hstack((X_train[:,:2],X_train[:,81:82]))
    X_test_categ  = np.hstack((X_test[:,:2],X_test[:,81:82]))
   
    X_train_real = np.hstack((X_train[:,2:81],X_train[:,82:]))
    X_test_real  = np.hstack((X_test[:,2:81],X_test[:,82:]))

    #  -------  Check flag that we want to use all data for encoding --------
    if all == True:
        X_all_categ = np.append(X_train_categ,X_test_categ, axis = 0)
        #print (X.shape, X_train_categ.shape, X_test_categ.shape)
        y_all = np.append(y_train, y_test, axis = 0)
        #print (y_all.shape, y_train.shape, y_test.shape)
    else:
        X_all_categ = X_train_categ
        y_all = y_train

    #  -------  Norm of real data on mean and deviation --------
    if real == 'standart':
        ss = StandardScaler()
        X_train_real_res = ss.fit_transform(X_train_real)
        X_test_real_res  = ss.transform(X_test_real)
    elif real == 'normal':
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_real_res = min_max_scaler.fit_transform(X_train_real)
        X_test_real_res  = min_max_scaler.transform(X_test_real)
    else:
        X_train_real_res = X_train_real
        X_test_real_res  = X_test_real
        
    
    #  -------  Encoding of categorical features  -----------
    if categ == 'target':
        encoder = ce.TargetEncoder(cols=[0,1,2], return_df = False)
        encoder.fit(X_all_categ, y_all)
        
        X_train_categ_res = encoder.transform(X_train_categ)
        X_test_categ_res  = encoder.transform(X_test_categ)
    elif categ == 'onehot':
        encoder = ce.OneHotEncoder(cols=[0,1,2], return_df = False)
        encoder.fit(X_all_categ, y_all)
        
        X_train_categ_res = encoder.transform(X_train_categ)
        X_test_categ_res  = encoder.transform(X_test_categ)
    elif categ == 'helmert':
        encoder = ce.HelmertEncoder(cols=[0,1,2], return_df = False)
        encoder.fit(X_all_categ, y_all)
        
        X_train_categ_res = encoder.transform(X_train_categ)
        X_test_categ_res  = encoder.transform(X_test_categ)
    elif categ == 'hash':
        encoder = ce.HashingEncoder(cols=[0,1,2], return_df = False)
        encoder.fit(X_all_categ, y_all)
        
        X_train_categ_res = encoder.transform(X_train_categ)
        X_test_categ_res  = encoder.transform(X_test_categ)
    else:
        X_train_categ_res = X_train_categ
        X_test_categ_res  = X_test_categ

    #  ------------     Joy data together  ---------------
    X_train_ready = np.hstack((X_train_categ_res,X_train_real_res))
    X_test_ready = np.hstack((X_test_categ_res,X_test_real_res))
    
    return X_train_ready, X_test_ready




from sklearn import decomposition
    
def pca(X_train, X_test, n_components):
    pca = pca = decomposition.PCA(n_components=n_components, random_state=228)
    pca.fit(X_train)    #[:,35:61]

    #plot_pca_all(X_train.shape[1]+1,pca(X_norm),pca(X_replaced),pca(X_train), name= 'all')
    #print ('PCA_components = 39:')
    #print (round(pca.explained_variance_ratio_[:39].sum(),4)*100) #95.2% 
    #max_feature_idx = np.argmax(pca.components_[1])  #x8, #74
    #print(list(train_data.columns)[max_feature_idx])
    
    X_train_pca = pca.transform(X_train)
    X_test_pca  = pca.transform(X_test)

    return X_train_pca, X_test_pca
