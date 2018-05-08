import pandas as pd
import datetime as dt

def mario():

    trainData = pd.read_csv('train.csv')
    testData = pd.read_csv('test.csv')

    trainDF = pd.DataFrame(trainData)
    testDF = pd.DataFrame(testData)

    trainDataLabels = list(trainDF)
    testDataLabels = list(testDF)

    trainDF[trainDataLabels[0]] = trainDF[trainDataLabels[0]].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    trainDF[trainDataLabels[0]] = trainDF[trainDataLabels[0]].map(dt.datetime.toordinal)

    testDF[testDataLabels[1]] = testDF[testDataLabels[1]].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    testDF[testDataLabels[1]] = testDF[testDataLabels[1]].map(dt.datetime.toordinal)

    trainDF[trainDataLabels[1]] = trainDF[trainDataLabels[1]].astype('category')
    tempDF = trainDF[trainDataLabels[1]].cat.codes
    YDict = dict(enumerate(trainDF[trainDataLabels[1]].cat.categories))
    trainDF[trainDataLabels[1]] = tempDF
    
    for i in range(2,6):
        trainDF[trainDataLabels[i]] = trainDF[trainDataLabels[i]].astype('category')
        trainDF[trainDataLabels[i]] = trainDF[trainDataLabels[i]].cat.codes

       
    for j in range(2,4):
        testDF[testDataLabels[j]] = testDF[testDataLabels[j]].astype('category')
        testDF[testDataLabels[j]] = testDF[testDataLabels[j]].cat.codes

    X = trainDF.drop([trainDataLabels[1], trainDataLabels[2], trainDataLabels[5], trainDataLabels[6]], axis=1)
    Y = trainDF[trainDataLabels[1]]
    test_X = testDF.drop([testDataLabels[0], testDataLabels[4]], axis=1)

    return X, Y, YDict, test_X
