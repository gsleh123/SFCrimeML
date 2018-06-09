import numpy as np
import pandas as pd
import datetime as dt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def seasons(month):
    summer=0
    fall=0
    winter=0
    spring=0
    if (month in [6, 7, 8]):
        summer=1
    if (month in [9, 10, 11]):
        fall=1
    if (month in [12, 1, 2]):
        winter=1
    if (month in [3, 4, 5]):
        spring=1
    return summer, fall, winter, spring

def mario():

    trainData = pd.read_csv('train.csv')
    testData = pd.read_csv('test.csv')

    trainDF = pd.DataFrame(trainData)
    testDF = pd.DataFrame(testData)

    trainDataLabels = list(trainDF)
    testDataLabels = list(testDF)

    trainDF['Hour'] = trainDF[trainDataLabels[0]].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
    trainDF['Month'] = trainDF[trainDataLabels[0]].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)
    trainDF['Minute'] = trainDF[trainDataLabels[0]].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').minute)

    testDF['Hour'] = testDF[testDataLabels[1]].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
    testDF['Month'] = testDF[testDataLabels[1]].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)
    testDF['Minute'] = testDF[testDataLabels[1]].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').minute)

    trainDF['StreetNo'] = trainDF[trainDataLabels[6]].apply(lambda x: int(x.rsplit(' ')[0]) if x.rsplit(' ')[0].isdigit() else 0)
    trainDF['Block'] = trainDF[trainDataLabels[6]].apply(lambda x: 1 if x.rsplit(' ')[1]=='Block' else 0)
    trainDF['Intersection'] = trainDF[trainDataLabels[6]].apply(lambda x: 0 if x.find("/") == -1 else 1)
    #trainDF["Summer"], trainDF["Fall"], trainDF["Winter"], trainDF["Spring"]=zip(*trainDF['Month'].apply(seasons))

    testDF['StreetNo'] = testDF[testDataLabels[4]].apply(lambda x: int(x.rsplit(' ')[0]) if x.rsplit(' ')[0].isdigit() else 0)
    testDF['Block'] = testDF[testDataLabels[4]].apply(lambda x: 1 if x.rsplit(' ')[1]=='Block' else 0)
    testDF['Intersection'] = testDF[testDataLabels[4]].apply(lambda x: 0 if x.find("/") == -1 else 1)
    #testDF["Summer"], testDF["Fall"], testDF["Winter"], testDF["Spring"]=zip(*testDF['Month'].apply(seasons))


    # PCA on X and Y coordinates
    #XY_DF = pd.DataFrame({'X' : []})
    #XY_DF['X'] = trainDF[trainDataLabels[7]]
    #XY_DF['Y'] = trainDF[trainDataLabels[8]]

    #pca = PCA(n_components=1, svd_solver='full')
    #pca.fit(XY_DF)
    #XY_pca = pca.transform(XY_DF)

    #trainDF['XY_pca'] = XY_pca
    #print trainDF['XY_pca'].shape

    #XY_DF = pd.DataFrame({'X' : []})
    #XY_DF['X'] = testDF[testDataLabels[5]]
    #XY_DF['Y'] = testDF[testDataLabels[6]]

    #pca = PCA(n_components=1, svd_solver='full')
    #pca.fit(XY_DF)
    #XY_pca = pca.transform(XY_DF)
    #testDF['XY_pca'] = XY_pca
    #print testDF['XY_pca'].shape
    #print "Passed"

    # standard scale XY
    xy_scaler = StandardScaler()
    xy_scaler.fit(trainDF[["X","Y"]])
    trainDF[["X","Y"]]=xy_scaler.transform(trainDF[["X","Y"]])

    xy_scaler = StandardScaler()
    xy_scaler.fit(testDF[["X","Y"]])
    testDF[["X","Y"]]=xy_scaler.transform(testDF[["X","Y"]])

    # other coordinate stuff
    trainDF["rot45_X"] = .707 * trainDF["Y"] + .707 * trainDF["X"] 
    trainDF["rot45_Y"] = .707 * trainDF["Y"] - .707 * trainDF["X"]
    trainDF["rot30_X"] = (1.732/2) * trainDF["X"] + (1./2) * trainDF["Y"]
    trainDF["rot30_Y"] = (1.732/2) * trainDF["Y"] - (1./2) * trainDF["X"]
    trainDF["rot60_X"] = (1./2) * trainDF["X"] + (1.732/2) * trainDF["Y"]
    trainDF["rot60_Y"] = (1./2) * trainDF["Y"] - (1.732/2) * trainDF["X"]
    trainDF["radial_r"] = np.sqrt( np.power(trainDF["Y"],2) + np.power(trainDF["X"],2) )

    testDF["rot45_X"] = .707 * testDF["Y"] + .707 * testDF["X"] 
    testDF["rot45_Y"] = .707 * testDF["Y"] - .707 * testDF["X"]
    testDF["rot30_X"] = (1.732/2) * testDF["X"] + (1./2) * testDF["Y"]
    testDF["rot30_Y"] = (1.732/2) * testDF["Y"] - (1./2) * testDF["X"]
    testDF["rot60_X"] = (1./2) * testDF["X"] + (1.732/2) * testDF["Y"]
    testDF["rot60_Y"] = (1./2) * testDF["Y"] - (1.732/2) * testDF["X"]
    testDF["radial_r"] = np.sqrt( np.power(testDF["Y"],2) + np.power(testDF["X"],2) )

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

    #print X

    return X, Y, YDict, test_X
