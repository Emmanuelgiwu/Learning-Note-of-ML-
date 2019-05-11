#!/usr/bin/python
# -*- coding:utf-8 -*-


import numpy as np
from  sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def iris_type(s):
    it = {b'Iris-setosa':0,
          b'Iris-versicolor':1,
          b'Iris-virginica':2}
    return it[s]



if __name__ == "__main__":
    path = u'8.iris.data'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters= {4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    x = x[:, 2:4]
    print(x)
    print(y)

    lr = Pipeline([('sc', StandardScaler()),('clF', LogisticRegression())])
    lr.fit(x, y.ravel())

    N, M = 500, 500
    x1_min, x1_max = x[:,0].min(), x[:,0].max()
    x2_min, x2_max = x[:,1].min(), x[:,1].max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1,t2)
    x_test = np.stack((x1.flat, x2.flat), axis=1)


    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_hat = lr.predict(x_test)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolors='k', s=50, cmap=cm_dark)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)
    plt.grid()
    plt.savefig('2.png')
    plt.show()












    y_hat = lr.predict(x)
    y =y.reshape(-1)
    result = y_hat ==y
    acc =np.mean(result)
    print('准确率: %.2f%%' % (100 *acc))



