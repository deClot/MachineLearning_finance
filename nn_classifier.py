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
    plt.plot(x_axis, val_loss, color='r', lw=3, alpha=0.7, label='Val Loss')
    plt.title('Loss graph')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    path = 'Graphs/NN_'
    plt.savefig(path + name + str(epochs) + 'bs'+str(batch_size)+'.png')

def fit_nn_model (epochs, X_train, y_train, X_dev, y_dev,file_out, batch_size = 32):
    nn = get_nn_model(X_train.shape[1])
    logs = nn.fit(X_train,y_train, epochs=epochs, batch_size=32,\
                  validation_data=(X_dev, y_dev), shuffle=True, verbose = 0)

    loss_graph(epochs, batch_size, logs.history.get('loss'), logs.history.get('val_loss'),\
           name = 'dev')
    y_train_pred = nn.predict_on_batch(X_train)
    y_train_pred = np.where(y_train_pred > 0.5, 1, 0)

    y_test_pred = nn.predict_on_batch(X_dev)
    y_test_pred = np.where(y_test_pred > 0.5, 1, 0)

    print (X_dev.shape, y_test_pred.shape)
    #cm = confusion_matrix(y_dev, y_test_pred)
    common.print_prediction(y_train_pred,y_train, y_test_pred, y_dev,file_out)

    return nn
