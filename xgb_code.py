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

    #evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 999
    param = {'max_depth':10, 'eta':0.3, 'silent':1, 'objective':'multi:softmax', 'num_class':len(YDict)}
    #classifier = xgb.train(param, dtrain, num_round)

    cv = xgb.cv(param
                , dtrain
                , num_boost_round = num_round
                , nfold = 4
                , early_stopping_rounds = 10)
    print(cv)

    categories = classifier.predict(dtest)

    return categories, YDict
