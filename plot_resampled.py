import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from nn_classifier import get_nn_model

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score

def resampled_study(X_train,y_train):
    model = tree.DecisionTreeClassifier(random_state=228, max_depth =5)
    plot_resample(X_train,y_train,model,comment='Tree',k = 20)
    
    model = RandomForestClassifier(n_estimators=32, max_depth=5, random_state=228)   
    plot_resample(X_train,y_train,model,comment='RandomForest',k = 20)
    
    model = LogisticRegression(C=1, class_weight=None,\
                               penalty = 'l2',random_state=228)
    plot_resample(X_train,y_train,model,comment='LogisticRegression',k = 20)
    
    model  =  GradientBoostingClassifier(n_estimators=64, max_depth=1,
                                         random_state=228)
    plot_resample(X_train,y_train,model,comment='GradBoost',k = 20)


def plot_resample(X_train,y_train,model,comment,k):
    x_axis = np.linspace(1, k, num=k)

    plt.figure(figsize=(10, 7))

    random  = np.zeros(x_axis.shape)
    smote   = np.zeros(x_axis.shape)
    adasyn  = np.zeros(x_axis.shape)

    for n in range(k):
        print (n+1)
        X_r,y_r  = RandomOverSampler(random_state=228).fit_resample(X_train,\
                                                                    y_train)
        X_s, y_s = SMOTE(random_state=228, k_neighbors=n+1).fit_resample(X_train,\
                                                                         y_train)
        X_a, y_a = ADASYN(random_state=228, n_neighbors=n+1).fit_resample(X_train,\
                                                                          y_train)

        
        random[n] = cross_val_score(model, X_r, y_r, scoring='recall',\
                                    cv =5).mean()
        smote[n]  = cross_val_score(model, X_s, y_s, scoring='recall',\
                                    cv =5).mean()
        adasyn[n] = cross_val_score(model, X_a, y_a, scoring='recall',\
                                    cv =5).mean()
        print ('finished')

    #print (y_prec)
    plt.plot(x_axis,random, color='b', lw=3, alpha=0.7, label='Random')
    plt.plot(x_axis,smote, color='r', lw=3, alpha=0.7, label='Smote')
    plt.plot(x_axis,adasyn, color='g', lw=3, alpha=0.7, label='Adasyn')
    plt.title(comment)
    plt.xlabel('number of neighbors')
    plt.ylabel('Metric')
    plt.xticks(x_axis)
    
    plt.legend(loc='upper right')
    plt.grid(True)

    path = 'Graphs/Resampled_'
    plt.savefig(path + comment +'k='+str(k)+ '.png')

def plot_resample_test(X_train,y_train,X_test,y_test,comment,k):
    x_axis = np.linspace(1, k, num=k)

    plt.figure(figsize=(10, 7))

    tre  = np.zeros(x_axis.shape)
    randomforest = np.zeros(x_axis.shape)
    linreg  = np.zeros(x_axis.shape)
    neurnet = np.zeros(x_axis.shape)
    gradboost  = np.zeros(x_axis.shape)

    for n in range(k):
        print (n+1)
        X_train2, y_train2 = ADASYN(random_state=228, n_neighbors=n+1).fit_resample(X_train, y_train)

        tr = tree.DecisionTreeClassifier(random_state=228, max_depth =5)
        rf = RandomForestClassifier(n_estimators=32, max_depth=5, random_state=228) 
        lr = LogisticRegression(C=1, class_weight=None,\
                           penalty = 'l2',random_state=228)
        gb = GradientBoostingClassifier(n_estimators=64, max_depth=1,
                                 random_state=228)

        tr = tr.fit(X_train2,y_train2)
        rf = rf.fit(X_train2,y_train2)
        lr = lr.fit(X_train2,y_train2)
        nn = get_nn_model(X_train2.shape[1])
        nn.fit(X_train2,y_train2, epochs=100, batch_size=64,\
                  validation_data=(X_test, y_test), shuffle=True, verbose = 0)
        gb = gb.fit(X_train2,y_train2)
        
        tre[n] = recall_score(y_test, tr.predict(X_test))
        randomforest[n]  = recall_score(y_test, rf.predict(X_test))
        linreg[n] = recall_score(y_test, lr.predict(X_test))

        predictions_test = nn.predict_on_batch(X_test)
        predictions_test = np.where(predictions_test > 0.5, 1, 0)
        neurnet[n] = recall_score(y_test, predictions_test)
        
        gradboost[n] = recall_score(y_test, gb.predict(X_test))
        
        print ('finished')

    #print (y_prec)
    plt.plot(x_axis,tre, color='b', lw=3, alpha=0.7, label='Tree')
    plt.plot(x_axis,randomforest, color='r', lw=3, alpha=0.7, label='Random Forest')
    plt.plot(x_axis,linreg, color='g', lw=3, alpha=0.7, label='Linear Regression')
    plt.plot(x_axis,neurnet, color='black', lw=3, alpha=0.7, label='Neural Network')
    plt.plot(x_axis,gradboost, color='orange', lw=3, alpha=0.7, label='Gradient Boosting')
    plt.title(comment)
    plt.xlabel('number neighbors')
    plt.ylabel('Metric')
    plt.legend(loc='upper right')
    plt.grid(True)

    path = 'Graphs/Resampled_'
    plt.savefig(path + comment +'k='+str(k)+ '.png')

