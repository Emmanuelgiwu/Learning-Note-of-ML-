#！usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

if __name__ =="__main__":
    data = pd.read_csv('8.Advertising.csv')
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    print(x,y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    model = Lasso()


    alpha_can = np.logspace(-3, 2, 10)
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model.fit(x,y)
    print('参数验证：\n', lasso_model.best_params_)

    y_hat = lasso_model.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test))**2)
    amse = np.sqrt(mse)

    t = np.arange(len(x_test))
    plt.plot(t, y_hat,'r-', linewidth =2, label ='预测')
    plt.plot(t, y_test, 'g-', linewidth=2, label='真实')
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.show()
