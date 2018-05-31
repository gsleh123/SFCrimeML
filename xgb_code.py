from sklearn.decomposition import PCA

import xgboost as xgb
import numpy as np
import pandas as pd
import parse

def boost():
    X, Y, YDict, test_X = parse.mario()
    trainNP_X = X.values
    trainNP_Y = Y.values
    testNP_X = test_X.values

    #pca = PCA(n_components=3)
    #pca.fit(X)

    #X_pca = pca.transform(X)
    #test_X_pca = pca.transform(test_X)

    #dtrain = xgb.DMatrix(X_pca, label=trainNP_Y)
    #dtest = xgb.DMatrix(test_X_pca)

    dtrain = xgb.DMatrix(trainNP_X, label=trainNP_Y)
    dtest = xgb.DMatrix(testNP_X)

    #evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 10
    #params = {'max_depth':12, 'min_child_weight':1, 'subsample':1, 'colsample_bytree':0.9, 'eta':0.1, 'silent':0, 'objective':'multi:softmax', 'num_class':len(YDict), 'eval_metric':'mlogloss'}

    params = {'max_depth':8, 'eta':0.05, 'silent':1, 'objective':'multi:softprob', 'num_class':39, 'eval_metric':'mlogloss',
              'min_child_weight':3, 'subsample':0.6,'colsample_bytree':0.6, 'nthread':4}    

    classifier = xgb.train(params, dtrain, num_round)
    
    #rounds(dtrain, params)

    #sample(dtrain, params)

    # error 0.710194 max_dept = 12, min_child = 1
    # error 0.6966112 max_depth = 20, min_child = 1

    #cv(dtrain, params)

    categories = classifier.predict(dtest)
    categories = categories.tolist()

    for i in range(0,len(categories)):
        categories[i].insert(0,i)

    return categories, YDict

def sample(dtrain, params, num_round = 50):
    # Best params: 1.0, 0.9, MAE: 0.7066928

    gridsearch_params = [
        (subsample, colsample)
        for subsample in [i/10. for i in range(7,11)]
        for colsample in [i/10. for i in range(7,11)]
    ]

    min_mlogloss = float("Inf")
    best_params = None

    # We start by the largest values and go down to the smallest
    for subsample, colsample in reversed(gridsearch_params):
        print("CV with subsample={}, colsample={}".format(subsample,colsample))

        # We update our parameters
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample

        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_round,
            seed=42,
            nfold=5,
            early_stopping_rounds=10
        )

        # Update best score
        mean_mlogloss = cv_results['test-mlogloss-mean'].min()
        boost_rounds = cv_results['test-mlogloss-mean'].argmin()
        print("\tmean error {} for {} rounds".format(mean_mlogloss, boost_rounds))
        if mean_mlogloss < min_mlogloss:
            min_mlogloss = mean_mlogloss
            best_params = (subsample,colsample)

    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mlogloss))

def rounds(dtrain, params):
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=999,
        evals=[(dtrain, "Train")],
        early_stopping_rounds=10
    )

    print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))

def cv(dtrain, params):
    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(6,15,2)
        for min_child_weight in range(1,7,2)
    ]

    min_mlogloss = float("Inf")
    best_params = None

    for max_depth, min_child_weight in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))

        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight

        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=999,
            seed=42,
            nfold=5,
            early_stopping_rounds=10
        )

        # Update best MAE
        mean_mlogloss = cv_results['test-mlogloss-mean'].min()
        boost_rounds = cv_results['test-mlogloss-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mlogloss, boost_rounds))

        if mean_mlogloss < min_mlogloss:
            min_mlogloss = mean_mlogloss
            best_params = (max_depth, min_child_weight)

    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mlogloss))
