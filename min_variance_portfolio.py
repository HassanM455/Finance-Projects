import pandas as pd  
import numpy as np 
from scipy.optimize import minimize



covarMat = pd.read_csv('/home/hassan/Downloads/Book1.csv')
omega = np.array(covarMat)
x0 = np.array([0.053 for i in range(20)])

'''_______GRADIENT DESCENT METHOD FOR LAGRANGE FUNCTION

lambda = 0.0001

'''

def gradient_lagrange(x):
	grad = np.array([])
	for i in range(len(x)):
		val = np.dot(omega[i],x) + coe_lagrange*x[i]
		grad = np.append(grad,val)
	else : 
		return grad

def gradient_varFunc(x):
	grad = np.array([])
	for i in range(len(x)):
		val = 2*np.dot(omega[i],x) 
		grad = np.append(grad,val)
	else : 
		return grad

coe_lagrange = np.float(0.0001)
stop_norm = 10
stop_iter = 25000
step_size = 0.0001
it = 1
s = 0
stop_norm2 = 10

while (stop_norm2 > 0.01 and stop_iter > it) or (s < 0.97 and stop_iter > it):
	if it == 1:
		xtry = x0
	g = gradient_lagrange(xtry)
	xtry = xtry - step_size*g
	s = sum(xtry)
	xtry /= s
	newg = gradient_lagrange(xtry)
	stop_norm = np.linalg.norm(newg)
	stop_norm2 = np.linalg.norm(gradient_varFunc(xtry))
	it += 1
print('\n \n \n')
print('final weights : ')
print(xtry)
print('\n \n ')
print('sum of weights : ', sum(xtry))
print('iteration count : ', it)
print('final_gradient norm : ', stop_norm)

test = gradient_varFunc(xtry)
print('variance function gradient norm at final weight point : ' , np.linalg.norm(test))

returns =  pd.read_csv('/home/hassan/Downloads/Book2.csv')
exp_returns = np.array(returns['avg_expected_return'])
print('____________________________________________________________')
variance = np.dot(xtry,np.dot(omega, xtry))
print('the min variance is : ' , variance)
print('expected returns for minimum portfolio : ' , np.dot(xtry , exp_returns))
print('risk of min variance is : ' , np.sqrt(variance))

print('____________________________________________________________')
print('variance for equally weighed portfolio : ', np.dot(x0, np.dot(omega, x0)))

xtry_variance = np.append(xtry, variance)
ind = range(1, 22)
min_var_w = pd.DataFrame({'min_var_weights_Variance' : xtry_variance},
	                      index = ind)


print('returns for equally weighted portfolio : ' , np.dot(exp_returns , x0))
print('risk of equally weighted portfolio is : ' , np.sqrt(np.dot(x0, np.dot(omega, x0))) )
print('\n \n \n ')
min_var_w.to_csv('min_var_weights&variance.csv')
