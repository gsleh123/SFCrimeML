from sklearn.neighbors import KNeighborsClassifier
import parse

def nearestNeighbors():
    X, Y, YDict, test_X = parse.mario()

    neigh = KNeighborsClassifier(n_neighbors=100)
    neigh.fit(X, Y)
 
    categories = neigh.predict(test_X)
    
    return categories, YDict
