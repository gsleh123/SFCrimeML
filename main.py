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

#trainDF[trainDataLabels[0]] = pd.to_datetime([trainDF[trainDataLabels[0]]])
#trainDF[trainDataLabels[0]] = dt.datetime.strptime(trainDF[trainDataLabels[0]], "%m/%d/%Y")
trainDF[trainDataLabels[0]] = trainDF[trainDataLabels[0]].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
trainDF[trainDataLabels[0]] = trainDF[trainDataLabels[0]].map(dt.datetime.toordinal)
print(trainDF[trainDataLabels[0]].iloc[[2]])

for i in range(2,6):
    print (i)
    temp = pd.get_dummies(trainDF[trainDataLabels[i]])
    trainDF[trainDataLabels[i]] = temp


X = trainDF.drop([trainDataLabels[1], trainDataLabels[6]], axis=1)
Y = trainDF[trainDataLabels[1]]




logReg = LogisticRegression()
logReg.fit(X, Y)

with open('submission.csv', 'w') as csvfile:
    resultWriter = csv.writer(csvfile)
    resultWriter.writerow(['Id', 'ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT',
        'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
        'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT', 
        'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES', 
        'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY',
        'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY',
        'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS'])
		
