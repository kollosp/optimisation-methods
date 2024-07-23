import numpy as np
import math

import matplotlib.pyplot as plt

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
    s.scatter(points[:, 0], points[:, 1])
    s.plot(points[:, 0], points[:, 1])
    plt.show()



#two fields in x
def f(x):
    return ((x[0])**2 + x[1]**2)
fX0 = [0,0]


def mag(x):
    return math.sqrt(sum(i ** 2 for i in x))


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum(np.dot(v,b)*b  for b in basis)
        if (w > 1e-10).any():
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)

def orthogonalVector(S):
    d = []
    for s in range(S):
        v = []
        for s1 in range(S):
            if s == s1:
                v.append(1.0)
            else:
                v.append(0.0)
        d.append(v)
    return np.array(d)

def Rosenbrock(func, x0=np.array([0,0]), theta=np.array([0.1,0.1]), beta=-0.5, epsilon=0.05, alpha=2):
    iterations = 0
    _lambda = np.zeros(x0.shape)
    S = x0.shape[0]
    x = []
    d = orthogonalVector(S)
    z = x0.copy()
    z0 = z.copy()

    while True:
        iterations += 1
        #step 1
        for s in range(S):

            z1 = z0.copy()
            z1 = z0 + d[s] * theta[s]

            #if step gives better result
            if(func(z1) < func(z0)):
                z0 = z0 + d[s]*theta[s]
                theta[s] = theta[s]*alpha
                _lambda[s] = _lambda[s] + theta[s]
            else:
                theta[s] = theta[s] * beta

        # step2

        # if new solution is worse than the best known
        if func(z0) >= func(z):

            if mag(theta) < epsilon:
                return [np.array(x), iterations]

            if func(z0) > func(x0):
                #check return condition
                a = np.zeros([S,S])
                for s in range(S):
                    _sum = 0
                    for ss in range(S):
                        _sum = _sum + _lambda[ss] * d[ss]
                    a[s] = _sum

                _lambda = np.zeros(x0.shape)
                d = gram_schmidt(a)

        # if new solution is better than the best known
        else:
            #save best and make next move
            z = z0.copy()

        x.append(z)

#[x, iterations] = Rosenbrock(f,np.array([-4,7]))

#printChart(x, f, [-10,10,-10,10])


# iteration number stayes the same, accurancy is better when we start closer to the minimum
def testAccurancyInFOfDifferentStartingPoint():
    x = np.array([0, 0])
    v = np.array([1, 1])
    stepFactor = 25

    pX = []
    pYY = []
    pY = []

    for i in range(100):
        [result, iterations] = Rosenbrock(f, x)

        if result.shape[0] > 0:
            print(x, iterations, result[-1])
            pX.append(mag(fX0 - x))
            pY.append(mag(fX0 - result[-1]))
            pYY.append(iterations)
            x = x + v * stepFactor

    fig = plt.figure()
    #s = fig.add_subplot(1, 1, 1, xlabel='Cartesian distance to from start to x[min]', ylabel='')
    s = fig.add_subplot(1, 1, 1, xlabel='Cartesian distance to from start to x[min]', ylabel='Iterations')
    s.plot(pX,pYY)
    plt.show()


def testAccurancyInFOfEpsilon():
    x = np.array([100/6, 100/6])
    stepFactor = 0.1
    epsilon = 0.05

    pX = []
    pY = []
    pYY = []

    for i in range(100):
        [result, iterations] = Rosenbrock(f, x, epsilon=epsilon)

        if result.shape[0] > 0:
            print("Epsilon,", epsilon, " , ", mag(fX0 - result[-1]), ", n ,", iterations)
            pX.append(epsilon)
            # pY.append(mag(fX0 - result[0]))
            pY.append(mag(fX0 - result[-1]))
            pYY.append(iterations)
            epsilon = epsilon + stepFactor

    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1, xlabel='Epsilon', ylabel='Iterations')
    #s = fig.add_subplot(1, 1, 1, xlabel='Epsilon', ylabel='Cartesian distance from result to x[min]')
    s.plot(pX, pYY)
    plt.show()

def testAccurancyInFOfBeta():
    x = np.array([100 / 6, 100 / 6])
    stepFactor = 0.01
    beta = -0.7

    pX = []
    pY = []
    pYY = []

    for i in range(98):
        [result, iterations] = Rosenbrock(f, x, beta=beta)

        if result.shape[0] > 0:
            print("Beta,", beta, " , ", mag(fX0 - result[-1]), ", n ,", iterations)
            pX.append(beta)
            # pY.append(mag(fX0 - result[0]))
            pY.append(mag(fX0 - result[-1]))
            pYY.append(iterations)
            beta = beta + stepFactor

    fig = plt.figure()
    #s = fig.add_subplot(1, 1, 1, xlabel='Beta', ylabel='Iterations')
    s = fig.add_subplot(1, 1, 1, xlabel='Beta', ylabel='Cartesian distance from result to x[min]')
    s.plot(pX, pY)
    plt.show()

def testAccurancyInFOfAlpha():
    x = np.array([999,999])
    stepFactor = 0.2
    beta = 1.1

    pX = []
    pY = []
    pYY = []

    for i in range(50):
        [result, iterations] = Rosenbrock(f, x, alpha=beta)

        if result.shape[0] > 0:
            print("Alpha,", beta, " , ", mag(fX0 - result[-1]), ", n ,", iterations)
            pX.append(beta)
            # pY.append(mag(fX0 - result[0]))
            pY.append(mag(fX0 - result[-1]))
            pYY.append(iterations)
            beta = beta + stepFactor

    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1, xlabel='Alpha', ylabel='Iterations')
    #s = fig.add_subplot(1, 1, 1, xlabel='Alpha', ylabel='Cartesian distance from result to x[min]')
    s.plot(pX, pYY)
    plt.show()

def testAccurancyInFOfDifferentTheta():
    x = np.array([999, 999])
    theta= np.array([0.1, 0.1])
    v = np.array([0.1, 0.1])
    stepFactor = 1

    pX = []
    pYY = []
    pY = []

    for i in range(100):
        [result, iterations] = Rosenbrock(f, x, theta=theta.copy())

        if result.shape[0] > 0:
            print(theta, iterations, result[-1])
            pX.append(mag(theta))
            pY.append(mag(fX0 - result[-1]))
            pYY.append(iterations)
            theta = theta + v * stepFactor

    fig = plt.figure()
    #s = fig.add_subplot(1, 1, 1, xlabel='Theta magnitude', ylabel='Iterations')
    s = fig.add_subplot(1, 1, 1, xlabel='Theta magnitude', ylabel='Cartesian distance from result to x[min]')
    s.plot(pX,pY)
    plt.show()


#testAccurancyInFOfDifferentStartingPoint()
#testAccurancyInFOfEpsilon()
#testAccurancyInFOfBeta()
#testAccurancyInFOfAlpha()
testAccurancyInFOfDifferentTheta()
