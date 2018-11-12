import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import category_encoders as ce

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
    
    #print(train_data.head(10))
    #print(test_data.head(10))
    #print (train_data['x80'].value_counts().shape)
    #print (test_data['x80'].value_counts().shape)
    #print (train_data['x80'].value_counts())
    #print (test_data['x80'].value_counts())

  #  from pandas.plotting import scatter_matrix
  #  scatter_matrix(train_data, alpha=0.05, figsize=(10, 10));

    X = train_data.drop(['Financial Distress'], axis='columns').values
    y = train_data['Financial Distress'].values

    X_test = test_data.drop(['Financial Distress'], axis='columns').values
    y_test = test_data['Financial Distress'].values

#    print (X[:,:3])
    if y_lab =='negative':
        y_ini = np.where(y >= -0.5, 1, 0)
        y = np.where(y < -0.5, 1, 0)
        y_test = np.where(y_test < -0.5, 1, 0)
    else:
        y_ini  = np.where(y >= -0.5, 1, 0)
        y      = np.where(y >= -0.5, 1, 0)
        y_test = np.where(y_test >= -0.5, 1, 0)
    
    print_count_data(y, y_test)

    print (train_data.head())
    
    return X,y,X_test,y_test,y_ini,train_data


def preprocessing_real_data(X_train,X_test,y_train):
    X_train_categ = np.hstack((X_train[:,:2],X_train[:,81:82]))
    X_test_categ  = np.hstack((X_test[:,:2],X_test[:,81:82]))

    #print (X_train[:, :])
    #print (X_train_categ.shape)
 
    X_train_real = np.hstack((X_train[:,2:81],X_train[:,82:]))
    X_test_real  = np.hstack((X_test[:,2:81],X_test[:,82:]))

    ss = StandardScaler()
    X_train_real2 = ss.fit_transform(X_train_real)
    X_test_real2  = ss.transform(X_test_real)

    ######
    #categ = pd.DataFrame(data = X_train_categ)
    #categ_test = pd.DataFrame(data = X_test_categ)

    encoder = ce.TargetEncoder(cols=[0,1,2], return_df = False)
    encoder.fit(X_train_categ,y_train)

    X_train_categ = encoder.transform(X_train_categ)
    X_test_categ = encoder.transform(X_test_categ)
    #print (categ_test)
    #print (X_test_categ[:,:1])
    #print (train_data[2].value_counts().shape)

    '''
    encoder.fit(X_train_categ)
    print(encoder)
    X_train_categ2 = encoder.transform(X_train_categ)
    print (X_train_categ2)
    #print (X_train_categ2.shape)
    #############################
    '''
    
    X_train_ready = np.hstack((X_train_categ[:,1:2],X_train_real[:,:79],\
                               X_train_categ[:,2:],X_train_real[:,79:] ))
    X_test_ready = np.hstack((X_test_categ[:,1:2],X_test_real[:,:79],\
                               X_test_categ[:,2:],X_test_real[:,79:] ))
    '''
    X_train_ready = np.hstack((X_train_real2[:,:79],\
                               X_train_real2[:,79:] ))
    X_test_ready = np.hstack((X_test_real2[:,:79],\
                              X_test_real2[:,79:] ))
    '''
    return X_train_ready, X_test_ready

from sklearn import decomposition
    
def pca(X_train, X_test, n_components):
    pca = pca = decomposition.PCA(n_components=n_components, random_state=228)
    pca.fit(X_train)    #[:,35:61]
    #plot_pca_all(X_train.shape[1]+1,pca(X_norm),pca(X_replaced),pca(X_train), name= 'all')
    print ('PCA_components = 39:')
    print (round(pca.explained_variance_ratio_[:39].sum(),4)*100) #95.2% 

    #max_feature_idx = np.argmax(pca.components_[1])  #x8, #74
    #print(list(train_data.columns)[max_feature_idx])
    
    X_train_pca = pca.transform(X_train)
    X_test_pca  = pca.transform(X_test)

    return X_train_pca, X_test_pca

import matplotlib.pyplot as plt

def plot_pca (n_features, pca, name):
    x_axis = np.arange(1, n_features)
    x_lables = np.arange(1, n_features,2)
    plt.figure(figsize=(12, 7))
    plt.scatter(x_axis, pca.explained_variance_ratio_, color='g')
    plt.xlabel('Number of component')
    plt.ylabel('Explained variance ratio, %')
    plt.xticks(x_lables)
    plt.grid(True)

    path = 'Graphs/PCA'
    plt.savefig(path +'_'+name+'.png')
  
def plot_pca_all (n_features,pca_X_norm,pca_X_replaced,pca_X_train, name):
    x_axis = np.arange(1, n_features)
    x_lables = np.arange(1, n_features,2)
    plt.figure(figsize=(12, 7))
    plt.scatter(x_axis, pca_X_norm.explained_variance_ratio_, color='blue',\
                label='X_norm')
    plt.scatter(x_axis, pca_X_replaced.explained_variance_ratio_, color='r',\
                label='X_oversampl')
    plt.scatter(x_axis, pca_X_train.explained_variance_ratio_, color='black',\
                label='X_train')
    
    plt.xlabel('Number of component')
    plt.ylabel('Explained variance ratio, %')
    plt.xticks(x_lables)
    plt.legend(loc='upper right')
    plt.grid(True)

    path = 'Graphs/PCA'
    plt.savefig(path +'_'+name+'.png')

def draw_objects(X, y, bdt):
    X_green_list = []
    X_red_list = []
    for i in range(X.shape[0]):
        if (y[i] > 0.5):
            X_green_list.append(X[i, :])
        else:
            X_red_list.append(X[i, :])
    X_green = np.array(X_green_list)
    X_red = np.array(X_red_list)
    plt.figure(figsize=(8, 8))
    plt.plot(X_green[:, 0], X_green[:, 1], 'ro')
    plt.plot(X_red[:, 0], X_red[:, 1], 'bo')
    print (X[:, 0].min(), X[:, 1].min())
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    plt.axis([x_min, x_max, y_min, y_max])
    plot_colors = "rg"
    plot_step = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    path = 'Graphs/Graph'
    plt.savefig(path +'_cb_dev'+'.png')
