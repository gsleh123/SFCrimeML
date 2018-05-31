from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
import parse
import pandas as pd

def kerasNN():
    hl_num = 23
    output_dim = 39
    epochs_size = 20
    batches = 64

    X, Y, YDict, test_X = parse.mario()
    print(type(Y))
    model = Sequential()
    model.add(Dense(hl_num, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    temp = list()
    for i in range(len(Y)):
        row = [0] * len(Y)
        row[Y[i]] = 1

        temp.append(row)

    Y_one_hot = pd.Series(temp)

    model.fit(X, Y_one_hot, epochs=epochs_size, batch_size=batches, validation_split=0.1, verbose=1)

