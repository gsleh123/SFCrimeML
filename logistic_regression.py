from sklearn.linear_model import LogisticRegression
import parse

def logReg():
    X, Y, YDict, test_X = parse.mario()

    logReg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    logReg.fit(X, Y)

    categories = logReg.predict_proba(test_X)
    categories = categories.tolist()

    for i in range(0,len(categories)):
        categories[i].insert(0,i)
    
    return categories, YDict
