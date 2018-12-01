import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd


def plot_grid_search(file_name, title):
    """Function to plot the results of the grid search optimisation process."""

    data = pd.read_csv(file_name) # Read the data to a pandas DataFrame
    data_acc = data.pivot(index="no feats", columns="splits", values="Accuracy")
    data_prec = data.pivot(index="no feats", columns="splits", values="Precision")
    data_rec = data.pivot(index="no feats", columns="splits", values="Recall")
    data_f1 = data.pivot(index="no feats", columns="splits", values="F1")

    X_acc = data_acc.columns.values
    Y_acc = data_acc.index.values
    Z_acc = data_acc.values
    x_acc,y_acc = np.meshgrid(X_acc,Y_acc)

    plt.figure()
    plt.suptitle(title, fontsize=20)
    plt.subplot(221)
    CS = plt.contour(x_acc,y_acc,Z_acc, cmap=cm.RdYlGn)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Accuracy')
    plt.xlabel('Fraction of dataset used for training.')
    plt.ylabel('Maximum number of features')

    X_prec = data_prec.columns.values
    Y_prec = data_prec.index.values
    Z_prec = data_prec.values
    x_prec,y_prec = np.meshgrid(X_prec,Y_prec)

    plt.subplot(222)
    CS = plt.contour(x_prec,y_prec,Z_prec, cmap=cm.RdYlGn)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Precision')
    plt.xlabel('Fraction of dataset used for training.')
    plt.ylabel('Maximum number of features')

    X_rec = data_rec.columns.values
    Y_rec = data_rec.index.values
    Z_rec = data_rec.values
    x_rec,y_rec = np.meshgrid(X_rec,Y_rec)

    plt.subplot(223)
    CS = plt.contour(x_rec,y_rec,Z_rec, cmap=cm.RdYlGn)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Recall')
    plt.xlabel('Fraction of dataset used for training.')
    plt.ylabel('Maximum number of features')

    X_f1 = data_f1.columns.values
    Y_f1 = data_f1.index.values
    Z_f1 = data_f1.values
    x_f1,y_f1 = np.meshgrid(X_f1,Y_f1)

    plt.subplot(224)
    CS = plt.contour(x_f1,y_f1,Z_f1, cmap=cm.RdYlGn)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('F1 Measure')
    plt.xlabel('Fraction of dataset used for training.')
    plt.ylabel('Maximum number of features')

    plt.show()
