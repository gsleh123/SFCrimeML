import pandas as pd
import datetime as dt

def mario():

    trainData = pd.read_csv('train.csv')
    testData = pd.read_csv('test.csv')

    trainDF = pd.DataFrame(trainData)
    testDF = pd.DataFrame(testData)

    trainDataLabels = list(trainDF)
    testDataLabels = list(testDF)

    trainDF['Hour'] = trainDF[trainDataLabels[0]].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
    trainDF['Month'] = trainDF[trainDataLabels[0]].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)

    testDF['Hour'] = testDF[testDataLabels[1]].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
    testDF['Month'] = testDF[testDataLabels[1]].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)

    trainDF['StreetNo'] = trainDF[trainDataLabels[6]].apply(lambda x: int(x.rsplit(' ')[0]) if x.rsplit(' ')[0].isdigit() else 0)
    trainDF['Block'] = trainDF[trainDataLabels[6]].apply(lambda x: 1 if x.rsplit(' ')[1]=='Block' else 0)
    testDF['StreetNo'] = testDF[testDataLabels[4]].apply(lambda x: int(x.rsplit(' ')[0]) if x.rsplit(' ')[0].isdigit() else 0)
    testDF['Block'] = testDF[testDataLabels[4]].apply(lambda x: 1 if x.rsplit(' ')[1]=='Block' else 0)


    # Assigning numeric values to different categories of crime
    trainDF[trainDataLabels[1]] = trainDF[trainDataLabels[1]].astype('category')
    tempDF = trainDF[trainDataLabels[1]].cat.codes
    YDict = dict(enumerate(trainDF[trainDataLabels[1]].cat.categories))
    trainDF[trainDataLabels[1]] = tempDF
    
    for i in range(3,5):
        trainDF[trainDataLabels[i]] = trainDF[trainDataLabels[i]].astype('category')
        trainDF[trainDataLabels[i]] = trainDF[trainDataLabels[i]].cat.codes
       
    for j in range(2,4):
        testDF[testDataLabels[j]] = testDF[testDataLabels[j]].astype('category')
        testDF[testDataLabels[j]] = testDF[testDataLabels[j]].cat.codes

    X = trainDF.drop([trainDataLabels[0], trainDataLabels[1], trainDataLabels[2], trainDataLabels[5], trainDataLabels[6], trainDataLabels[7], trainDataLabels[8]], axis=1)
    Y = trainDF[trainDataLabels[1]]
    test_X = testDF.drop([testDataLabels[0], testDataLabels[1], testDataLabels[4], testDataLabels[5], testDataLabels[6]], axis=1)

    return X, Y, YDict, test_X
