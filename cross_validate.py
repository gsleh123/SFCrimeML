from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np
import parse
import xgboost as xgb


def crossValid():
    X, Y, YDict, test_X = parse.mario()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

    trainNP_X = X_train.values
    trainNP_Y = y_train.values
    testNP_X = X_test.values
    testNP_Y = y_test.values

    dtrain = xgb.DMatrix(trainNP_X, label=trainNP_Y)
    dtest = xgb.DMatrix(testNP_X)
    

    num_round = 250
    params = {'max_depth':8, 'eta':0.05, 'silent':1, 'objective':'multi:softmax', 'num_class':39, 'eval_metric':'mlogloss',
              'min_child_weight':3, 'subsample':0.6,'colsample_bytree':0.6, 'nthread':4}    

    model = xgb.train(params, dtrain, num_round)
    categories = model.predict(dtest)
    print type(categories)
    predictions = [round(value) for value in categories]

    accuracy = accuracy_score(y_test, predictions) # accuracy calculation
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
