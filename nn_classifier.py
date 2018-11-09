import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout


def get_nn_model(input_size):
    nn = Sequential()
    nn.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform',\
                 input_shape=(input_size,)))
    nn.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    nn.add(Dense(units=1, activation='sigmoid'))
    
    nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return nn

import matplotlib.pyplot as plt

def loss_graph(epochs, train_loss, val_loss):
    x_axis = np.linspace(1, epochs, num=epochs)
    
    plt.figure(figsize=(10, 7))
    plt.plot(x_axis, train_loss, color='b', lw=3, alpha=0.7, label='Train Loss')
    plt.plot(x_axis, val_loss, color='r', lw=3, alpha=0.7, label='Val Loss')
    plt.title('Loss graph')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)


def acc_graph(epochs, train_acc, val_acc):
    x_axis = np.linspace(1, epochs, num=epochs)
    
    plt.figure(figsize=(10, 7))
    plt.plot(x_axis, train_acc, color='g', lw=3, alpha=0.7, label='Train Accuracy')
    plt.plot(x_axis, val_acc, color='orange', lw=3, alpha=0.7, label='Val Accuracy')
    plt.title('Accuracy graph')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
