from sklearn.linear_model import LogisticRegression
import parse

def logReg():
    X, Y, YDict, test_X = parse.mario()

    logReg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    logReg.fit(X, Y)

    categories = logReg.predict(test_X)
    
    return categories, YDict
