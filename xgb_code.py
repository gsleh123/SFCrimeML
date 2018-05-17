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
    num_round = 50
    params = {'max_depth':12, 'min_child_weight':1, 'subsample':1, 'colsample_bytree':0.9, 'eta':0.1, 'silent':1, 'objective':'multi:softmax', 'num_class':len(YDict)}
    #classifier = xgb.train(params, dtrain, num_round)
    
    sample(dtrain, params)

    # error 0.710194 max_dept = 12, min_child = 1
    # error 0.6966112 max_depth = 20, min_child = 1

    #cv(dtrain, params)
    #cv = xgb.cv(param
    #            , dtrain
    #            , num_boost_round = num_round
    #            , nfold = 4
    #            , early_stopping_rounds = 10)
    #print(cv)

    categories = classifier.predict(dtest)

    return categories, YDict

def sample(dtrain, params, num_round = 50):
    # Best params: 1.0, 0.9, MAE: 0.7066928

    gridsearch_params = [
        (subsample, colsample)
        for subsample in [i/10. for i in range(7,11)]
        for colsample in [i/10. for i in range(7,11)]
    ]

    min_merror = float("Inf")
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
        mean_merror = cv_results['test-merror-mean'].min()
        boost_rounds = cv_results['test-merror-mean'].argmin()
        print("\tmean error {} for {} rounds".format(mean_merror, boost_rounds))
        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params = (subsample,colsample)

    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_merror))

def cv(dtrain, params, num_round = 50):
    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(20,25)
        for min_child_weight in range(1,3)
    ]

    min_merror = float("Inf")
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
            num_boost_round=num_round,
            seed=42,
            nfold=5,
            early_stopping_rounds=10
        )

        # Update best MAE
        mean_merror = cv_results['test-merror-mean'].min()
        boost_rounds = cv_results['test-merror-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_merror, boost_rounds))

        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params = (max_depth, min_child_weight)

    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_merror))
