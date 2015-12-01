import sys
import scipy.misc, scipy.optimize, scipy.io, scipy.special
from numpy import *

import pylab
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab

def sigmoid( z ):
	return scipy.special.expit(z)

def computeCost( theta, X, y, lamda ):
	m = shape( X )[0]
	hypo 	   = sigmoid( X.dot( theta ) )
	term1 	   = log( hypo ).dot( -y )
	term2 	   = log( 1.0 - hypo ).dot( 1 - y )
	left_hand  = (term1 - term2) / m
	right_hand = theta.T.dot( theta ) * lamda / (2*m)
	return left_hand + right_hand

def gradientCost( theta, X, y, lamda ):
	m = shape( X )[0]
	grad = X.T.dot( sigmoid( X.dot( theta ) ) - y ) / m
	grad[1:] = grad[1:] + ( (theta[1:] * lamda ) / m )
	return grad

def oneVsAll( X, y, num_classes, lamda ):
	m,n 		= shape( X )
	X 			= c_[ones((m, 1)), X]
	all_theta 	= zeros((n+1, num_classes))

	for k in range(0, num_classes):
		theta 			= zeros(( n+1, 1 )).reshape(-1)
		temp_y 			= ((y == (k+1)) + 0).reshape(-1)
		result 			= scipy.optimize.fmin_cg( computeCost, fprime=gradientCost, x0=theta, \
		args=(X, temp_y, lamda), maxiter=50, disp=False, full_output=True )
		all_theta[:, k] = result[0]
		print "%d Cost: %.5f" % (k+1, result[1])
	return all_theta

def predictOneVsAll( theta, X, y ):
	m,n = shape( X )
	X 	= c_[ones((m, 1)), X]

	correct = 0
	for i in range(0, m ):
		prediction 	= argmax(theta.T.dot( X[i] )) + 1
		actual 		= y[i]
		if actual == prediction:
			correct += 1
	print "Accuracy: %.2f%%" % (correct * 100.0 / m )

def main():
	set_printoptions(precision=6, linewidth=200)
	mat 			 = scipy.io.loadmat( "ex3data1.mat" )
	X, y 			 = mat['X'], mat['y']
	m, n 			 = shape( X )
	input_layer_size = 400
	num_labels 		 = 10
	lamda 			 = 0.1

	theta = oneVsAll( X, y, num_labels, lamda )
	predictOneVsAll( theta, X, y )
	
if __name__ == '__main__':
	main()