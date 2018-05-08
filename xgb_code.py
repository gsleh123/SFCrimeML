import xgboost as xgb
import pandas as pd
import parse

def boost():
    X, Y, YDict, test_X = parse.mario()
    print(len(YDict))
    trainNP_X = X.values
    trainNP_Y = Y.values
    testNP_X = test_X.values

    dtrain = xgb.DMatrix(trainNP_X, label=trainNP_Y)
    dtest = xgb.DMatrix(testNP_X)

    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 10
    param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'multi:softmax', 'num_class':len(YDict) }
    booster = xgb.train(param, dtrain, num_round, evallist)

    categories = booster.predict(dtest)
    print(categories)

    return categories, YDict
