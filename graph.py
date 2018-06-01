import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse


def hour_graph(X):
    hours_dict = {}

    for i in range (0, 24):
        hours_dict[i] = X.loc[X.Hour == i, 'Hour'].count()

    plt.bar(range(len(hours_dict)), hours_dict.values(), align='center')  # python 2.x
    plt.xticks(range(len(hours_dict)), hours_dict.keys())  # in python 2.x
    plt.title("Crime Distribution by Hour", weight='bold')
    plt.xlabel("Hours")
    plt.ylabel("Number of Crime Instances")
    plt.show()

def month_graph(X):
    month_dict = {}

    for i in range (1, 13):
        month_dict[i] = X.loc[X.Month == i, 'Month'].count()

    plt.bar(range(len(month_dict)), month_dict.values(), align='center')  # python 2.x
    plt.xticks(range(len(month_dict)), month_dict.keys())  # in python 2.x
    plt.title("Crime Distribution by Month", weight='bold')
    plt.xlabel("Month")
    plt.ylabel("Number of Crime Instances")
    plt.show()

def minute_graph(X):
    minute_dict = {}

    for i in range (0, 60):
        minute_dict[i] = X.loc[X.Minute == i, 'Minute'].count()

    plt.bar(range(len(minute_dict)), minute_dict.values(), align='center')  # python 2.x
    plt.xticks(range(len(minute_dict)), minute_dict.keys())  # in python 2.x
    plt.title("Crime Distribution by Minute", weight='bold')
    plt.xlabel("Minute")
    plt.ylabel("Number of Crime Instances")
    plt.show()


X, Y, YDict, test_X = parse.mario()
minute_graph(X)
