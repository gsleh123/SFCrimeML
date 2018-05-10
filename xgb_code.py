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
    params = {'max_depth':10, 'eta':0.3, 'silent':1, 'objective':'multi:softmax', 'num_class':len(YDict)}
    #classifier = xgb.train(param, dtrain, num_round)

    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(6,13,2)
        for min_child_weight in range(1,8,2)
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

    #cv = xgb.cv(param
    #            , dtrain
    #            , num_boost_round = num_round
    #            , nfold = 4
    #            , early_stopping_rounds = 10)
    #print(cv)

    categories = classifier.predict(dtest)

    return categories, YDict
