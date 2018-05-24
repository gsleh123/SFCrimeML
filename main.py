import csv
import logistic_regression as lr
import xgb_code
import neighbors

def submit(categories, YDict):
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
            row[categoryDict[YDict[categories[i]]]] = 1
            resultWriter.writerow(row)

def xgbSubmit(categories, YDict):
    with open('submission.csv', 'w') as csvfile:
        listofcategories = ['Id', 'ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT',
            'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
            'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT', 
            'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES', 
            'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY',
            'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY',
            'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS']

        resultWriter = csv.writer(csvfile)
        resultWriter.writerow(listofcategories)
        resultWriter.writerow(categories)


#categories, YDict = lr.logReg()
categories, YDict = xgb_code.boost()
#categories, YDict = neighbors.nearestNeighbors()

#submit(categories, YDict)
xgbSubmit(categories, YDict)		
