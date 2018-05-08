import xgboost as xgb
import pandas as pd
import parse

def boost():
    X, Y, YDict, test_X = parse.mario()

    trainNP_X = X.values
    trainNP_Y = Y.values
    testNP_X = test_X.values

    dtrain = xgb.DMatrix(trainNP_X, label=trainNP_Y)
