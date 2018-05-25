from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import parse
import numpy as np

def randomForest():
    X, Y, YDict, test_X = parse.mario()

    pca = PCA(n_components=3)
    pca.fit(X)

    X_pca = pca.transform(X)
    test_X_pca = pca.transform(test_X)

    forest = RandomForestClassifier(n_estimators=200, max_depth=13)
    #forest.fit(X, Y)
 
    forest.fit(X_pca, Y)

    #categories = forest.predict_proba(test_X)
    categories = forest.predict_proba(test_X_pca)
    categories = categories.tolist()

    for i in range(0,len(categories)):
        categories[i].insert(0,i)
    
    return categories, YDict
