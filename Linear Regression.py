#!usr/bin/python
# -*- coding:utf-8 -*-
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    path = '8.Advertising.csv'
    data = pd.read_csv(path)
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    print(x)
    print(y)


    plt.plot(data['TV'], y, 'ro', label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y,'mv', label='Newspaper')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    ln = LinearRegression()
    model = ln.fit(x_train, y_train)
    print(model)
    print(ln.coef_)
    print(ln.intercept_)

    y_hat = ln.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test))**2)
    smse = np.sqrt(mse)
    print(mse, smse)

    t = np.arange(len(x_test))
    plt.plot(t, y_hat, 'r-', linewidth=2, label='test')
    plt.plot(t, y_test, 'g-', linewidth=2, label='real')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()