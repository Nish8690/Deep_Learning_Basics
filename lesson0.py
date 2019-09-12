import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def initialize_parameters(n_x, n_h, n_y):
	np.random.seed(1)
	W1 = np.random.randn(n_h, n_x) * 0.01
	b1 = np.zeros((n_h, 1))
	W2 = np.random.randn(n_y, n_h) * 0.01
	b2 = np.zeros((n_y, 1))

	parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}

	return parameters

parameters = initialize_parameters(3, 2, 1)
'''
print(parameters["W1"])
print(parameters["b1"])
print(parameters["W2"])
print(parameters["b2"])
'''

def initialize_parameters_deep(layer_dims):
	np.random.seed(3)
	parameters = {}
	L = len(layer_dims)

	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
		parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

	return parameters

'''
parameters = initialize_parameters_deep([5, 4, 3])

print(parameters["W1"])
print(parameters["b1"])
print(parameters["W2"])
print(parameters["b2"])
'''

def sigmoid(Z):
	A = 1/(1 + np.exp(-Z))
	cache = Z
	return A, cache

def relu(Z):
	A = np.maximum(Z, 0)
	cache = Z
	return A, cache

def linear_forward(A, W, b):
	Z = np.dot(W, A) + b
	cache = (A, W, b)

	return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
	if activation == "sigmoid":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = sigmoid(Z)

	elif activation == "relu":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)
	
	cache = (linear_cache, activation_cache)

	return A, cache

def L_model_forward(X, parameters):
	caches = []
	A = X
	L = len(parameters) // 2

	for l in range(1, L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], "relu")
		caches.append(cache)

	AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], "sigmoid")
	caches.append(cache)

	return AL, caches

def compute_cost(AL, Y):
	m = Y.shape[1]
	cost = -1/m * np.sum(Y*np.log(AL) + (1 - Y)*np.log(1 - np.log(AL)), axis = 1, keepdims = True)
	cost = np.squeeze(cost)
	return cost

def linear_backward(dZ, cache):
	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = 1/m * np.dot(dZ, A_prev.T)
	db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
	dA_prev = np.dot(W.T, dZ)

	return dA_prev, dW, db

def relu_backward(dA, cache):
	Z = cache
	dZ = np.array(dA, copy=True)
	dZ[Z <= 0] = 0
	return dZ

def sigmoid_backward(dA, cache):
	Z = cache
	s = 1/(1+np.exp(-Z))
	dZ = dA*s*(1-s)
	return dZ

def linear_activation_backward(dA, cache, activation):
	linear_cache, activation_cache = cache
	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)

	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
	
	return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
	grads = {}
	L = len(caches)
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)

	dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

	current_cache = caches[L - 1]

	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")

	for l in reversed(range(L - 1)):
		current_cache = caches[l]
		dA_prev_temp, dW_temp, dB_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
		grads["dA" + str(l + 1)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp
	
	return grads

def update_parameters(parameters, grads, learning_rate):
	L = len(parameters) // 2

	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

	return parameters

df = pd.read_csv("heart.csv")

X = df.iloc[:, :-1].to_numpy().T
Y = df.iloc[:, -1:].to_numpy().T

print(X.shape)
print(Y.shape)

n_x = 13
n_h = 7
n_y = 1

layers_dims = (n_x, n_h, n_y)

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
	grads = {}
	costs = []
	m = X.shape[1]
	(n_x, n_h, n_y) = layers_dims

	parameters = initialize_parameters(n_x, n_h, n_y)

	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	for i in range(0, num_iterations):
		A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
		A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

		cost = compute_cost(A2, Y)

		dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

		dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
		dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

		grads["dW1"] = dW1
		grads["db1"] = db1
		grads["dW2"] = dW2
		grads["db2"] = db2

		parameters = update_parameters(parameters, grads, learning_rate)

		W1 = parameters["W1"]
		b1 = parameters["b1"]
		W2 = parameters["W2"]
		b2 = parameters["b2"]

		if print_cost and i % 100 == 0:
			print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
		if print_cost and i % 100 == 0:
			costs.append(cost)

	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title("Learning rate = " + str(learning_rate))
	plt.show()
	
	return parameters

parameters = two_layer_model(X, Y, layers_dims = (n_x, n_h, n_y), num_iterations = 1000, print_cost = True)
