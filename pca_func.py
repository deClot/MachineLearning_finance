import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from imblearn.over_sampling import SMOTE
from trees_classifiers import model_RandomForest_fit, model_tree_fit,Tree_graph
from grad_boost import model_GradBoost
from regression import model_logistic_regression
from sklearn.metrics import precision_recall_fscore_support

def pca(X_train, X_test, n_components):
    pca = decomposition.PCA(n_components=n_components, random_state=228,\
                                  copy = False)
    pca.fit(X_train)    #[:,35:61]
    #plot_pca(X_train.shape[1]+1,pca, name= 'after norm_before resampl')

    #print ('PCA_components = 41:')
    #print (round(pca.explained_variance_ratio_[:41].sum(),4)*100) #95.2% 

    #max_feature_idx = np.argmax(pca.components_[1])  #x8, #74
    #print(list(train_data.columns)[max_feature_idx])
    
    X_train_pca = pca.transform(X_train)
    X_test_pca  = pca.transform(X_test)

    return X_train_pca, X_test_pca


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

def pca_recall(X_train,y_train,X_dev,y_dev,X_test,y_test,file_out,comment ):
    '''For comparing pca results fro different models'''
    x_axis = np.linspace(1, X_train.shape[1], num=X_train.shape[1])
    
    plt.figure(figsize=(10, 7))

    tree  = np.zeros(x_axis.shape)
    ranfor = np.zeros(x_axis.shape)
    linreg = np.zeros(x_axis.shape)
    gradboos = np.zeros(x_axis.shape)

    tree2  = np.zeros(x_axis.shape)
    ranfor2 = np.zeros(x_axis.shape)
    linreg2 = np.zeros(x_axis.shape)
    gradboos2 = np.zeros(x_axis.shape)
   
    for n in range(X_train.shape[1]):
        print (n+1)
        pca(X_train, X_test, n+1)
         
        X_replaced, y_replaced = SMOTE(random_state=228, k_neighbors=5).fit_resample(X_train_pca, y_train)
        tr = model_tree_fit(X_replaced,y_replaced,X_dev,y_dev,\
                            X_test_pca,y_test,file_out,\
                            max_depth=5,\
                            out = False)
        rf = model_RandomForest_fit(file_out,X_replaced,y_replaced,X_dev,y_dev,\
                                    X_test_pca,y_test,\
                                    n_estimators=32, max_depth=5, out = False)
        lr = model_logistic_regression(X_replaced,y_replaced,X_dev,y_dev,\
                                       X_test_pca,y_test,\
                                       file_out, flag_degree = False, degree=2, C=1, \
                                       class_weight = None, out=False)
        gb  = model_GradBoost(X_replaced,y_replaced,X_dev,y_dev,\
                              X_test_pca,y_test,file_out,\
                              n_estimators=64,depth=1, out = False)

        all_tree = precision_recall_fscore_support(y_test,tr.predict(X_test_pca))
        prec_tree, rec_tree,f1_all, __ = all_tree
        print (prec_tree, rec_tree)
        all_rf = precision_recall_fscore_support(y_test,rf.predict(X_test_pca))
        prec_rf, rec_rf,f1_rf, __ = all_rf
        all_lr = precision_recall_fscore_support(y_test,lr.predict(X_test_pca))
        prec_lr, rec_lr,f1_lr, __ = all_lr
        all_gb = precision_recall_fscore_support(y_test,gb.predict(X_test_pca))
        prec_gb, rec_gb,f1_gb, __ = all_gb

        tree[n]  = rec_tree[1]
        ranfor[n]  = rec_rf[1]
        linreg[n]  = rec_lr[1]
        gradboos[n]  = rec_gb[1]

        tree2[n]  = rec_tree[0]
        ranfor2[n]  = rec_rf[0]
        linreg2[n]  = rec_lr[0]
        gradboos2[n]  = rec_gb[0]
        
        print ('finished')

    plt.plot(x_axis, tree, color='b', lw=3, alpha=0.7, label='Tree')
    plt.plot(x_axis, ranfor, color='r', lw=3, alpha=0.7, label='RandomForest')
    plt.plot(x_axis, linreg, color='g', lw=3, alpha=0.7, label='Linear Regression')
    plt.plot(x_axis, linreg, color='black', lw=3, alpha=0.7, label='Gradient Boosting')

    plt.plot(x_axis, tree2, color='b', lw=3, alpha=0.7, linestyle='--')
    plt.plot(x_axis, ranfor2, color='r', lw=3, alpha=0.7, linestyle='--')
    plt.plot(x_axis, linreg2, color='g', lw=3, alpha=0.7, linestyle='--')
    plt.plot(x_axis, linreg2, color='black', lw=3, alpha=0.7, linestyle='--')
    plt.xlabel('PCA components_')
    plt.ylabel('Metric')
    plt.legend(loc='upper right')
    plt.grid(True)

    path = 'Graphs/Precice_PCA_'
    plt.savefig(path + comment+ '.png')

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
    plt.savefig(path +'_abdt_dev'+'.png')
