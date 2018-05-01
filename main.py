import csv
import pandas as pd

#testDataLabels = ['Dates', 'DayOfWeek', 'PdDistrict', 'Address', 'X	Y
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

trainDF = pd.DataFrame(trainData)
testDF = pd.DataFrame(testData)

trainDataLabels = list(trainDF)
testDataLabels = list(testDF)

print(trainData[trainDataLabels[0]])

with open('submission.csv', 'w') as csvfile:
    resultWriter = csv.writer(csvfile)
    resultWriter.writerow(['Id', 'ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT',
        'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
        'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT', 
        'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES', 
        'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY',
        'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY',
        'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS'])
		
