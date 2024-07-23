from random import seed
import numdifftools as nd
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import math

def F(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
FXO = [1,3]

def algorithm(func, epsilon, x0):
    n = 0
    x = np.zeros((10000,2))
    tau = 0.1
    x[0] = x0
    while True:
        H = nd.Hessian(func)(x[n])
        G = nd.Gradient(func)(x[n])
        x[n+1] = x[n] - tau * np.dot(la.inv(H),G)
        if la.norm(x[n+1] - x[n]) < epsilon:
            return x[n+1], n
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
        #pY.append(mag(FXO - z))
        pYY.append(n)
        x = x + v * stepFactor

    fig = plt.figure()
    #s = fig.add_subplot(1, 1, 1, xlabel='Cartesian distance from start to x[min]', ylabel='Accuracy')
    s = fig.add_subplot(1, 1, 1, xlabel='Cartesian distance from start to x[min]', ylabel='Iterations')
    s.plot(pX,pYY)
    plt.show()


def testAccurancyInFOfDifferentEpsilon():
    x = np.array([[735,  1278.9]])

    epsilon = 0.1
    pX = []
    pYY = []
    pY = []

    for i in range(15):
        z, n = algorithm(F, epsilon,x)

        print(x, z, n)
        pX.append(epsilon)
        pY.append(mag(FXO - z))
        #pYY.append(n)
        epsilon *= 0.1

    fig = plt.figure()
    #s = fig.add_subplot(1, 1, 1, xlabel='Epsilon', ylabel='Iteration count')
    s = fig.add_subplot(1, 1, 1, xlabel='Epsilon', ylabel='Accuracy')
    s.plot(pX,pY)
    plt.show()

testAccurancyInFOfDifferentStartingPoint()