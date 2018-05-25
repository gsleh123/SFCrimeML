from sklearn.neighbors import KNeighborsClassifier
import parse
import numpy as np

def nearestNeighbors():
    X, Y, YDict, test_X = parse.mario()

    neigh = KNeighborsClassifier(n_neighbors=2000)
    neigh.fit(X, Y)
 
    categories = neigh.predict_proba(test_X)
    categories = categories.tolist()

    for i in range(0,len(categories)):
        categories[i].insert(0,i)
    
    return categories, YDict
