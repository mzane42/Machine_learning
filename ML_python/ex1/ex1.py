import sys
from numpy import *
import scipy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

def part1():
    A = eye(5)
    print A


def hypothesis(X, theta):
    return X.dot(theta)

def computeCost(X, y, theta):
    m 	 = len(y)
    term = hypothesis(X, theta) - y
    return (term.T.dot(term) / (2 * m))[0, 0]

def gradientDescent(X, y, theta, alpha, iterations):
    grad = copy(theta)
    m 	 = len(y)
    for counter in range(0, iterations):
        inner_sum = X.T.dot(hypothesis(X, grad) - y)
        grad 	 -= alpha / m * inner_sum
    return grad


def plot(X, y):
    pyplot.plot(X, y, 'rx', markersize=5 )
    pyplot.ylabel('Profit in $10,000s')
    pyplot.xlabel('Population of City in 10,000s')


def read_data():
    data = genfromtxt("ex1data1.txt", delimiter=',')
    X, y = data[:, 0], data[:, 1]
    m 	 = len(y)
    y 	 = y.reshape(m, 1)

    plot(X, y)
    pyplot.show(block=True)


def linear_reg():
    data = genfromtxt('ex1data1.txt', delimiter=',')
    X, y = data[:, 0], data[:, 1]
    m 	 = len(y)
    y 	 = y.reshape(m, 1)

    X 			= c_[ones((m, 1)), X]
    theta 		= zeros((2, 1))
    iterations 	= 1500
    alpha 		= 0.01

    cost 	= computeCost(X, y, theta)
    theta 	= gradientDescent(X, y, theta, alpha, iterations)
    print cost
    print theta

    predict1 = array([1, 3.5]).dot(theta)
    predict2 = array([1, 7]).dot(theta)
    print predict1
    print predict2

    plot(X[:, 1], y)
    pyplot.plot(X[:, 1], X.dot(theta), 'b-')
    pyplot.show(block=True)


def plot_data():
    data = genfromtxt("ex1data1.txt", delimiter=',')
    X, y = data[:, 0], data[:, 1]
    m 	 = len(y)
    y 	 = y.reshape(m, 1)
    X 	 = c_[ones((m, 1)), X]

    theta0_vals = linspace(-10, 10, 100)
    theta1_vals = linspace(-4, 4, 100)

    J_vals = zeros((len(theta0_vals), len(theta1_vals)), dtype=float64)
    for i, v0 in enumerate(theta0_vals):
        for j, v1 in enumerate(theta1_vals):
            theta 		 = array((theta0_vals[i], theta1_vals[j])).reshape(2, 1)
            J_vals[i, j] = computeCost(X, y, theta)

    R, P = meshgrid(theta0_vals, theta1_vals)

    fig = pyplot.figure()
    pyplot.contourf(R, P, J_vals.T, logspace(-2, 3, 20))
    pyplot.plot(theta[0], theta[1], 'rx', markersize = 10)
    pyplot.show(block=True)


def main():
    set_printoptions(precision=6, linewidth=200)

    A = eye(5)
    print(A)
    read_data()
    linear_reg()
    plot_data()

    sys.exit()


if __name__ == '__main__':
    main()
