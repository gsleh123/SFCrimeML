from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization

def kerasNN():
    X, Y, YDict, test_X = parse.mario()
