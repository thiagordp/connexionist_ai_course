import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import  preprocessing

def main():
    print("Start")

    X = np.linspace(0, 2 * math.pi, 1000).reshape(-1,1)
    y = np.array([math.sin(x) + np.random.random()/4 for x in X])
    
    r2s = []

    for i in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, shuffle=True)
        model = MLPRegressor(hidden_layer_sizes=(5,), solver="adam", verbose=False, early_stopping=True, max_iter=1000, alpha=0.00001, momentum=0.5) 
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test) 
        r2s.append(r2_score(y_test, y_pred))
    print("Minimum %.2f, Maximum %.2f, Avg: %.2f" % (min(r2s), max(r2s), np.mean(r2s)))






if __name__ == "__main__":
    main()