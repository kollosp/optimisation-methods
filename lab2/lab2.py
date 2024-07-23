import numpy as np
import math
import plotly.express as px

#two fields in x
def f(x):
	return 10*(x[0]**2 + x[1]**2)
fX0 = [0,0]

def mag(x): 
    return math.sqrt(sum(i**2 for i in x))
 

#HookeJeeves(f, [5,12], 1, 0.999, 0.001, 0.1)

def HookeJeeves(func, x0, theta, beta, epsilon, alpha):
	S = len(x0)
	z = x0
	x = z 
	n = 0

	d = []
	for s in range(S):		
		v = []	
		for s1 in range(S):
			if s == s1:
				v.append(1.0)
			else:
				v.append(0.0) 	
		d.append(v)

	d = np.array(d)

	while True:
		for s in range(S):
			#calculate next position
			#move one step forward for s dimention 
			z_temp = z + theta*d[s]

			if(func(z_temp) < func(z)): 
				#step 2 - save new better position
				z = z_temp
			
			#calculate next position
			#move one step backward for s dimention
			z_temp = z - theta*d[s]

			if(func(z_temp) < func(z)): 
				#step 2 - save new better position
				z = z_temp
				
			#print(z_temp)

		#step 3

		if(func(z) < f(x)):
			#step 4
			x_old = x
			x = z
			z = z + alpha * (x - x_old)

		if(theta < epsilon):
			return [x, n]

		theta = theta*beta 
		n = n + 1

		#print(n, ",", x, "theta", theta)






#iteration number stayes the same, accurancy is better when we start closer to the minimum
def testAccurancyInFOfDifferentStartingPoint(): 
	x = np.array([10,10])
	v = np.array([1,1])
	stepFactor = 0.2
	
	pX = []
	pY = []
	
	for i in range(100):
		result = HookeJeeves(f, x, 1, 0.9, 0.01, 0.1)
		print("Starting point,", x, "," , mag(fX0 - x), " , ", mag(fX0 - result[0]), ";")
		pX.append(mag(fX0 - x))
		pY.append(mag(fX0 - result[0]))
		x = x + v * stepFactor
		
	fig = px.scatter(x=pX, y=pY, labels={"x": "Distance from starting point to minimum", "y": "Distance from result to minimum"} )
	fig.show()
		
def testAccurancyInFOfEpsilon():
	x = np.array([10,10])
	stepFactor = 2
	epsilon = 1
	
	pX = []
	pY = []
	
	for i in range(25):
		result = HookeJeeves(f, x, 1, 0.999, epsilon, 0.1)
		print("Epsilon,", epsilon, " , ", mag(fX0 - result[0]), ", n ,", result[1])
		pX.append(epsilon)
		#pY.append(mag(fX0 - result[0]))
		pY.append(result[1])
		epsilon = epsilon / stepFactor
		
	fig = px.scatter(x=pX, y=pY, labels={"x": "Epsilon", "y": "Iterations count"}, title="Iteration count in function of epsilon")
	fig.show()

def testAccurancyInFOfAlpha():
	x = np.array([200,200])
	stepFactor = 0.01
	alphaMax = 1
	alphaFactor = 1

	pX = []
	pY = []
	
	for i in range(100):
		result = HookeJeeves(f, x, 1, 0.999, 0.05, alphaFactor)
		print("alpha,", alphaMax - alphaFactor, " , ", mag(fX0 - result[0]), ", n ,", result[1])
		pX.append(alphaFactor)
		#pY.append(mag(fX0 - result[0]))
		pY.append(result[1])
		alphaFactor = alphaFactor - stepFactor

	fig = px.scatter(x=pX, y=pY, labels={"x": "Alpha", "y": "Iterations count"}, title="Iteration count in function of alpha")
	#fig = px.scatter(x=pX, y=pY, labels={"x": "Alpha", "y": "Distance from result point to minimum"}, title="Accuracy in function of alpha")
	fig.show()

def testAccurancyInFOfBeta():
	x = np.array([400,400])
	stepFactor = 0.00001
	beta = 0.9999
	
	pX = []
	pY = []
	
	for i in range(100):
		result = HookeJeeves(f, x, 1, beta, 0.01, 0.01)
		print("beta,", beta, " , ", mag(fX0 - result[0]), ", n ,", result[1])
		pX.append(beta)
		#pY.append(mag(fX0 - result[0]))
		pY.append(result[1])
		beta = beta - stepFactor
		
		
	#fig = px.scatter(x=pX, y=pY, labels={"x": "Beta", "y": "Distance from result point to minimum"}, title="Accuracy in function of beta")
	fig = px.scatter(x=pX, y=pY, labels={"x": "Beta", "y": "Iterations count"}, title="Iteration count in function of Beta")
	fig.show()

testAccurancyInFOfAlpha()

