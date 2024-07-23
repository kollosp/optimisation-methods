import numpy as np
import statistics

def f(x): 
	return (x-2.3) * (x-2.3) #min at x=2.3 
def g(x): 
	return (x-2) * (x-6) #min at x=4

def h(x): 
	return x**4 + 2*x**3 + 2 #x=-1.5
	
#minimum at x*=0 f(x*) = -200
def Ackley2(x):
	return -200 * (np.e ** (-0.02*abs(x)))


def golden(func, a = 0, b = 0, epsilon = 0.1, alpha = 0.618):	
	#step 0	
	lambda_n = a + (1-alpha)*(b-a) 
	ni_n = a + alpha * (b-a)
	n = 1

	#step 1
	while(b-a > epsilon):

		if(func(lambda_n) > func(ni_n)): 
			#step 2
			a = lambda_n
			# b stays same
			lambda_n = ni_n
			ni_n = a + alpha*(b-a)

		else: 
			#step 3		
			#a stays same
			b = ni_n
			ni_n = lambda_n
			lambda_n = a + (1 - alpha)*(b - a)

		#step 	4
		n = n + 1


	return [a, b, n]


def quadraticApprox(func, a = 0, b = 0, epsilon = 0.1, alpha = 0.618):
	t1 = a
	t2 = (a + b) / 2
	t3 = b
	n = 0
	while(t2 - t1 > epsilon and t3 - t2 > epsilon): 
		n = n + 1
		
		#step 1
		t4 = (func(t1)*(t3**2 - t2**2) + func(t2)*(t1**2 - t3**2) + func(t3)*(t2**2 - t1**2)) 
		#print(t4)

		t4_prim = 2 * (func(t1) * (t3 - t2) + func(t2) * (t1 - t3) + func(t3) * (t2 - t1) )
		#print(t4_prim)
		
		t4 = t4 / t4_prim
		#print("t4", t4, func(t4) , func(t2))
		if(t4 > t2):
			#step 2
			if(func(t4) > func(t2)):
				#t1 = t1
				#t2 = t2				
				t3 = t4			
			else:
				t1 = t2
				t2 = t4
				#t3 = t3
		else:
 			#step 3
			if(func(t4) > func(t2)):
				t1 = t4		
				#t2 = t2	
				#t3 = t3		
			else:
				#t1 = t1	
				t3 = t2
				t2 = t4
	
	#print([t1,t2,t3])

	#if(t2 - t1 < epsilon):
	#	return [t1,t2]
	#elif(t3 - t2 < epsilon): 
	#	return [t2,t3]
	
	return [t1,t3,n]
	
	


#print("quadratic approximation")
#print("Min for (x-2)^2: ", quadraticApprox(f, -15, 10))
#print("Min for (x-2)*(x-6): ", quadraticApprox(g, -10, 10))
#print("Min for x^3: ", quadraticApprox(h, -5.1, 2.0))

#print("Min for (x-2)^2: ", Ackley2(0))
#test

#This function checks relation between n and search domain.
#results: n is increasing when searched interval is increased.
def testGoldenRelationNandInterval():
	shift = np.pi
	for i in range(100):
		print("Processing golden search in range,", i*2, ",", (golden(Ackley2, -i + shift, i + shift))[2])

#testGoldenRelationNandInterval()
#print("### next test ###")

#This function checks relation between n and tolerance.
#results: epsilon is grater is positive and grater than zero. Lowering epsilon results with increasing n number and improving precision of the algorithm.
def testGoldenRelationNandEpsilon():
	shift = np.pi
	epsilon = 1
	print(" , Epsilon, N")
	for i in range(25):
		print("Processing golden search with epsilon,", epsilon, ",",  golden(Ackley2, -25 + shift, 25 + shift, epsilon)[2])
		epsilon /= 2

#testGoldenRelationNandEpsilon()
#print("### next test ###")


#This function checks relation between n and search domain.
#results: n is increasing when searched interval is increased.
def testQuadraticApproxNandInterval():
	shift = np.pi
	for i in range(100):
		print("Processing QuadraticApprox in range,", i*2 + 2, ",", (quadraticApprox(h, -i-1 + shift, 1+i + shift))[2])
		

#testQuadraticApproxNandInterval()
		
#This function checks relation between n and tolerance.
#results: epsilon is grater is positive and grater than zero. Lowering epsilon results with increasing n number and improving precision of the algorithm.
def testQuadraticApproxNandEpsilon():
	shift = np.pi
	epsilon = 1
	for i in range(25):
		print("Processing QuadraticApprox search with epsilon,", epsilon, ",",  quadraticApprox(h, -25 + shift, 25 + shift, epsilon)[2])
		epsilon /= 2
		
#testQuadraticApproxNandEpsilon()

def comparisionN():
	a = -25
	b = 25
	epsilon = 0.1
	shift = np.pi
	print("range , n golden(f), n quadratic(f), n golden(g), n quadratic(g), n golden(h), n quadratic(h)")
		
	for i in range(100):
	
		gf = golden(f, a-i,b+i, epsilon)[2]
		qf = quadraticApprox(f, a-i,b+i, epsilon)[2]
		gg = golden(g, a-i,b+i, epsilon)[2]
		qg = quadraticApprox(g, a-i,b+i, epsilon)[2]
		gh = golden(h, a-i,b+i, epsilon)[2]
		qh = quadraticApprox(h, a-i,b+i, epsilon)[2]
		
		print(b-a + 2*i, ",", gf, ",", qf, ",", gg, ",", qg, ",", gh, ",", qh)

def comparisionAccurancy():
	a = -25
	b = 25
	epsilon = 0.1
	shift = np.pi
	print("range , n golden(f), n quadratic(f), n golden(g), n quadratic(g), n golden(h), n quadratic(h)")
		
	fxmin = 2.3
	gxmin = 4
	hxmin = -1.5
			
	for i in range(100):
	
		gf = golden(f, a-i,b+i, epsilon)[1]
		qf = quadraticApprox(f, a-i,b+i, epsilon)[1]
		gg = golden(g, a-i,b+i, epsilon)[1]
		qg = quadraticApprox(g, a-i,b+i, epsilon)[1]
		gh = golden(h, a-i,b+i, epsilon)[1]
		qh = quadraticApprox(h, a-i,b+i, epsilon)[1]
		
		print(b-a + 2*i, ",", fxmin - gf, ",", fxmin - qf, ",", gxmin - gg, ",", gxmin - qg, ",", hxmin - gh, ",", hxmin - qh)
	
comparisionAccurancy()
