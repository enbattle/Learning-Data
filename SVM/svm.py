import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np
import random
import neuralNetwork
import time
import statistics
import cvxopt

def polynomialKernal(x1, x2, power=2):
	"""
	Returns the Nth order kernel (initialized to 2)
	"""
	return (1 + np.dot(np.transpose(x1), x2)) ** power

class SVM:
	"""
	Support Vector Machine
		- C: soft margin value
	"""

	def __init__(self, C):
		self.C = C
		self.kernel = polynomialKernal
		self.K = None
		self.a = None
		self.b = None
		self.sv_x = None
		self.sv_y = None

	def createKernal(self, datapoints, ls):
		"""
		Creates the K matrix kernel needed for the quadratic programming solver
		"""
		K = np.zeros((ls, ls))
		for i in range(ls):
			for j in range(ls):
				K[i,j] = self.kernel(datapoints[i], datapoints[j], 8)
		return K

	def QPSolver(self, datapoints, labels, K, ls):
		"""
		Generates Q, p, A, and c matrices to be put into the CVXOPT QP solver
		"""
		Q = self.kernel(datapoints.T * np.array([labels]), 
			datapoints.T * np.array([labels]), 8)
		p = np.ones((ls,)) * -1

		temp1 = np.array([labels])
		temp2 = np.array([labels]) * -1
		temp3 = np.identity(ls)
		A = np.concatenate([temp1, temp2, temp3, temp3])

		temp1 = np.zeros(ls+2)
		temp2 = np.ones(ls) * self.C
		c = np.concatenate([temp1, temp2])

		Qmat = cvxopt.matrix(Q, tc='d')
		pmat = cvxopt.matrix(p, tc='d')
		Amat = -1 * cvxopt.matrix(A, tc='d')
		cmat = -1 * cvxopt.matrix(c, tc='d')

		sol = cvxopt.solvers.qp(Qmat, pmat, Amat, cmat)
		return np.ravel(sol['x'])

	def train(self, datapoints, labels):
		"""
		Train the input (datapoints and labels) by using the kernel and QP solver
		"""

		# Note the dimensions
		len_samples, len_features = datapoints.shape

		# Create the kernel
		self.K = None
		self.K = self.createKernal(datapoints, len_samples)

		# Alpha values
		a = self.QPSolver(datapoints, labels, self.K, len_samples)

		# Finding the support vectors in a, X_sv, and Y_sv
		self.a = None
		self.sv_x = None
		self.sv_y = None
		self.b = None

		sv = a > 0
		self.a = a[sv]
		self.sv_x = datapoints[sv]
		self.sv_y = labels[sv]
		print("{0} support vectors out of {0} points".format(len(self.a), len_samples))

		# Calculating the bias value
		temp = 0
		randVal = np.random.randint(0, len(self.a))
		for i in range(len(self.a)):
			temp += (self.sv_y[i] * self.a[i] * self.kernel(self.sv_x[i], self.sv_x[randVal], 8))
		self.b = self.sv_y[randVal] - temp

	def predict(self, Z):
		"""
		Calculating the final hypothesis, g(x)
		"""

		y_predict = []
		for i, z in enumerate(Z):
			temp = 0

			for j in range(len(self.a)):
				temp += (self.sv_y[j] * self.a[j] * self.kernel(self.sv_x[j], z, 8))

			y_predict.append(temp)

		return np.sign(y_predict + self.b)

	def crossValidate(self, dtrain, ytrain, dtest, ytest, startC, endC):
		"""
		Creates a spread of C values and uses cross validation to pick the best C value
		"""

		# Initialize error list
		errors = []

		# Train the SVM with different C values and store the resulting error value
		for c in range(startC, endC+1):	
			print("Current C:", c)		
			self.C = c
			self.train(dtrain, ytrain)
			y_predict = self.predict(dtest)
			num_incorrect = np.sum([y_predict[j] != ytest[j] for j in range(len(y_predict))])

			errors.append(num_incorrect / len(dtest))

		# Find minimum error value
		minErrorC = min(errors)
		minErrorIndex = errors.index(minErrorC)

		print("Optimal C Found: C =", startC + minErrorIndex)
		print("Error:", minErrorC)

		return startC + minErrorIndex

		"""
		Plot Ecv vs C if needed
		"""
		# C_values = np.arange(startC, endC+1)
		# plt.title("Ecv vs C")
		# plt.xlabel("c")
		# plt.ylabel("Ecv")
		# plt.plot(C_values, errors, color="red")
		# plt.savefig("Ecv vs C")