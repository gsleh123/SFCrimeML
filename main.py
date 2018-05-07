import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
import datetime as dt

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

for i in range(2,6):
    temp = pd.get_dummies(trainDF[trainDataLabels[i]])
    trainDF[trainDataLabels[i]] = temp

for j in range(2,4):
    temp = pd.get_dummies(testDF[testDataLabels[j]])
    testDF[testDataLabels[j]] = temp



X = trainDF.drop([trainDataLabels[1], trainDataLabels[2], trainDataLabels[5], trainDataLabels[6]], axis=1)
Y = trainDF[trainDataLabels[1]]

test_X = testDF.drop([testDataLabels[0], testDataLabels[4]], axis=1)

logReg = LogisticRegression()
logReg.fit(X, Y)

categories = logReg.predict(test_X)

print(len(categories))


with open('submission.csv', 'w') as csvfile:
    listofcategories = ['Id', 'ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT',
        'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
        'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT', 
        'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES', 
        'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY',
        'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY',
        'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS']

    categoryDict = {key: value for value, key in enumerate(listofcategories)}

    resultWriter = csv.writer(csvfile)
    resultWriter.writerow(listofcategories)

    for i in range(0, len(categories)):
        row = [0] * len(listofcategories)
        row[0] = i
        row[categoryDict[categories[i]]] = 1
        resultWriter.writerow(row)
        
		
