import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from common import print_prediction

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def model_GradBoost (X_train,y_train,X_dev,y_dev,X_test,y_test,file_out,\
                     n_estimators, depth=1, out = True):
    gb = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=depth,
                                 random_state=228)
    gb.fit(X_train, y_train)

    predictions      = gb.predict(X_train)
    predictions_test = gb.predict(X_test)

    if X_dev == None:
        predictions_dev = None
        dev = False
    else:
        predictions_dev  = gb.predict(X_dev)
    
    print_prediction(predictions,y_train,\
                     predictions_dev, y_dev, \
                     predictions_test, y_test, file_out,dev, out= out)
    
    return gb


def GradBoost_graph(X_train, y_train, X_test, y_test,file_out,name,\
                    X_dev= None,y_dev=None,\
                    n_estimators=64, depth=5):
    x_axis = np.linspace(1, n_estimators, num=n_estimators)
    d_axis = np.linspace(1, depth, num=depth)
    
    plt.figure(figsize=(10, 7))

    y_acc      = np.zeros(x_axis.shape)
    y_rec_pos = np.zeros(x_axis.shape)
    y_rec_neg  =  np.zeros(x_axis.shape)
    y_mean = np.zeros(x_axis.shape)

    res = np.zeros((d_axis.shape[0],2))
    res2 = np.zeros((d_axis.shape[0],2))
    res3 = np.zeros((d_axis.shape[0],2))
    
    i=0
    for d in range(depth):
        i=0
        
        for n in range(n_estimators):
            print (d+1,':',n+1)
            gb = model_GradBoost(X_train,y_train,X_dev, y_dev,X_test,y_test,file_out,\
                       n_estimators = n+1, depth=depth, out = False)

            all = precision_recall_fscore_support(y_test,gb.predict(X_test))
            prec_all, rec_all,f1_all, __ = all

            y_rec_pos[i] = rec_all[0]
            y_rec_neg[i]  = rec_all[1]
            y_mean[i]    = (y_rec_pos[i]+y_rec_neg[i])/2
            #print (accuracy_score(y_test,gb.predict(X_test)))
            y_acc[i]      = accuracy_score(y_test,gb.predict(X_test))
            i=i+1
            print ('finished')
            
        ii = np.argmax(y_rec_neg)
        
        res[d,0] = ii+1
        res[d,1] = np.amax(y_rec_neg)

        #ii =np.argmax(y_rec_pos)
        res2[d,0] = ii+1
        res2[d,1] = y_rec_pos[ii]

        #ii =np.argmax(y_mean)
        res3[d,0] = ii+1
        res3[d,1] = y_mean[ii]
        break


    #y_rec_neg = res[:,1:]
    #y_rec_pos = res2[:,1:]
    #y_mean    = res3[:,1:]
    #plt.plot(d_axis, y_rec_pos, color='b', lw=3, alpha=0.7, label='Recall_positive')
    #plt.plot(d_axis, y_rec_neg, color='g', lw=3, alpha=0.7, label='Recall_negative')
    #plt.plot(d_axis, y_mean, color='black', lw=3, alpha=0.7, label='Recall_mean')
    
    plt.plot(x_axis, y_rec_pos, color='b', lw=3, alpha=0.7, label='Recall_positive')
    plt.plot(x_axis, y_rec_neg, color='r', lw=3, alpha=0.7, label='Recall_negative')
    plt.plot(x_axis, y_mean, color='black', lw=3, alpha=0.7, label='Recall_mean')
    #plt.plot(x_axis, y_acc, color='g', lw=3, alpha=0.7, label='Accuracy')
    plt.title('GradBoosting')
    plt.xlabel('estim')
    plt.ylabel('Metric')
    plt.legend(loc='upper right')
    plt.grid(True)

    path = 'Graphs/GradBoost'
    plt.savefig(path + str(depth)+'depth'+'est'+str(n_estimators)+name+ '.png')
