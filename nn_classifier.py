import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

import common

def get_nn_model(input_size):
    nn = Sequential()
    nn.add(Dense(units=32, activation='relu', kernel_initializer='random_uniform',\
                 input_shape=(input_size,)))
    nn.add(Dropout(0.25))
    nn.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    nn.add(Dropout(0.5))
    
    nn.add(Dense(units=1, activation='sigmoid'))
    
    nn.compile(loss='binary_crossentropy', optimizer='adam')
    return nn

import matplotlib.pyplot as plt

def loss_graph(epochs, batch_size, train_loss, val_loss, name):
    x_axis = np.linspace(1, epochs, num=epochs)
    
    plt.figure(figsize=(10, 7))
    plt.plot(x_axis, train_loss, color='b', lw=3, alpha=0.7, label='Train Loss')
    if val_loss != None:
        plt.plot(x_axis, val_loss, color='r', lw=3, alpha=0.7, label='Val Loss')
    plt.title('Loss graph')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    path = 'Graphs/NN_'
    plt.savefig(path + name + str(epochs) + 'bs'+str(batch_size)+'.png')

def fit_nn_model (epochs, X_train, y_train, X_dev, y_dev,X_text, y_text,\
                  file_out, batch_size = 32, out = True):
    nn = get_nn_model(X_train.shape[1])
    if X_dev == None:
        X_dev = X_test
        y_dev = y_test
        flag = 1

    #validation_data=(X_dev, y_dev),
    logs = nn.fit(X_train,y_train, epochs=epochs, batch_size=batch_size,\
                  shuffle=True, verbose = 0)

    loss_graph(epochs, batch_size, logs.history.get('loss'),\
               logs.history.get('val_loss'), name = name)
    loss_graph(epochs, batch_size, logs.history.get('loss'), name = name)
    
    predictions = nn.predict_on_batch(X_train)
    predictions = np.where(predictions > 0.5, 1, 0)

    predictions_test = nn.predict_on_batch(X_test)
    predictions_test = np.where(predictions_test > 0.5, 1, 0)

    if flag == 1:
        predictions_dev = None
        dev = False
    else:
        predictions_dev = nn.predict_on_batch(X_dev)
        predictions_dev = np.where(predictions_dev > 0.5, 1, 0)
    
    
    #cm = confusion_matrix(y_dev, y_test_pred)
    common.print_prediction(predictions, y_train, \
                                            predictions_dev, y_dev, \
                                            predictions_test, y_test,\
                                            file_out, dev, out= out )

    return nn
