from random import seed
import numdifftools as nd
import numpy as np
from numpy import linalg as la
import math
import matplotlib.pyplot as plt


def F(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
FXO = [1,3]

def minimum_search(func, z, d):
    results = np.zeros((1000, 3))
    results[0] = [z[0], z[1], func(z)]
    tau = 0
    dtau = 0.01
    n = 0
    while True:
        tau = tau + dtau
        results[n + 1] = [results[n][0] + tau * d[0], results[n][1] + tau * d[1],
                          func([results[n][0] + tau * d[0], results[n][1] + tau * d[1]])]
        if n != 0 and results[n][2] > results[n-1][2]:
            return results[n-1]
        n = n + 1

def algorithm(func, epsilon, x0):
    d = np.zeros((1000, 2))
    z = np.zeros((1000, 2))
    z[0] = x0
    n = 0
    while True:
        if n != 0 and la.norm(z[n] - z[n - 1]) < epsilon:
            return z[n], n
        d[n] = (-1) * nd.Gradient(func)(z[n])
        x, y, f = minimum_search(func, z[n], d[n])
        z[n + 1] = [x, y]
        n = n + 1



def printChart(points, func, limits):
    xx = np.linspace(limits[0], limits[1], 100)
    yy = np.linspace(limits[2], limits[3], 100)

    # filling the heatmap, value by value
    fun_map = np.empty((xx.size, yy.size))
    for i in range(xx.size):
        for j in range(yy.size):
            fun_map[i, j] = func([yy[j], xx[i]])

    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1, xlabel='$\\gamma$', ylabel='$\\mu$')
    im = s.imshow(
        fun_map,
        extent=(xx[0], xx[-1], yy[0], yy[-1]),
        origin='lower')
    fig.colorbar(im)
    if(points != None):
        s.scatter(points[:, 0], points[:, 1])
        s.plot(points[:, 0], points[:, 1])
    plt.show()



def mag(x):
    return math.sqrt(sum(i ** 2 for i in x))


def testAccurancyInFOfDifferentStartingPoint():
    x = np.array([0, 0])
    v = np.array([1, 1.74])
    stepFactor = 15

    epsilon = 0.0000001
    pX = []
    pYY = []
    pY = []

    for i in range(50):
        z, n = algorithm(F, epsilon,x)

        print(x, z, n)
        pX.append(mag(FXO - x))
        pY.append(mag(FXO - z))
        #pYY.append(n)
        x = x + v * stepFactor

    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1, xlabel='Cartesian distance to from start to x[min]', ylabel='Accuracy')
    #s = fig.add_subplot(1, 1, 1, xlabel='Cartesian distance to from start to x[min]', ylabel='Iterations')
    s.plot(pX,pY)
    plt.show()


def testAccurancyInFOfDifferentStartingPoint():
    x = np.array([[735,  1278.9]])

    epsilon = 0.1
    pX = []
    pYY = []
    pY = []

    for i in range(15):
        z, n = algorithm(F, epsilon,x)

        print(x, z, n)
        pX.append(epsilon)
        #pY.append(mag(FXO - z))
        pYY.append(n)
        epsilon *= 0.1

    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1, xlabel='Epsilon', ylabel='Iteration count')
    #s = fig.add_subplot(1, 1, 1, xlabel='Cartesian distance to from start to x[min]', ylabel='Iterations')
    s.plot(pX,pYY)
    plt.show()

#testAccurancyInFOfDifferentStartingPoint()