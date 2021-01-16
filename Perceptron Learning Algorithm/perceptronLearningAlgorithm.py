import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt

'''
Perceptron algorithm that finds the idea weights given a data set and labels

Parameters
----------
coords: data points in 2 dimensions ([x1, x2])
labels: binary target output for each data point (-1 or 1)
epochs: number of iterations that needs to be run
misclassifiedList

Returns
-------
w: resulting weight vector after learning the data points

'''
def perceptron_algorithm(coords, labels, epochs, misclassifiedList):
	w = np.zeros((coords.shape[1]) + 1)

	for i in range(epochs):
		misclassified = 0
		for x, y in zip(coords, labels):
			dotProd = np.dot(x, w[1:]) + w[0]

			target = 1.0 if (dotProd > 0.0) else -1.0

			if(target != y):
				print(y, target, "classified incorrectly")
				misclassified += 1
				w[1:] += y * x
				w[0] += x[0]
			else:
				print(y, target, "classified correctly")

			print("PRINTING W", w)

		misclassifiedList.append(misclassified)

	return w

if __name__ == "__main__":
	# Set up the random generator seed
	np.random.seed(121232141)

	# Total number of random points
	total = 1000

	# Create 20 random points that are linearly separable
	A = 2 * np.random.random_sample((total//2, 2)) + 0
	B = 2 * np.random.random_sample((total//2, 2)) - 2

	# Create the lists that hold the respective binary labels for the 20 points
	A1 = np.ones((total//2, 1))
	B1 = np.negative(np.ones((total//2, 1)))

	x = np.linspace(-3, 3)
	y = -1 * x

	# Plot the scatter points and the target function, y = -x
	plt.plot(x, y, color="blue", label="target function f")
	plt.scatter(A[:,0], A[:,1], marker="o")
	plt.scatter(B[:,0], B[:,1], marker="x")
	plt.title("PLA Testing")
	plt.xlabel("x1")
	plt.ylabel("x2")

	coords = np.concatenate((A, B))
	labels = np.concatenate((A1, B1))

	# Set up and run perceptron algorithm
	misclassifiedList = []
	epochs = 10
	w = perceptron_algorithm(coords, labels, epochs, misclassifiedList)

	# Create the coordinates to plot the separator line
	x = np.linspace(-3, 3)
	slope = -(w[0]/w[2]) / (w[0]/w[1])
	intercept = -(w[0]/w[2])
	y = (slope * x) + intercept

	# # Plot the classified points along with separator line to show that the dataset is linearly separable
	plt.plot(x, y, color="red", label="final hypothesis g")
	plt.legend(loc="upper right")

	# # Plot the number of iterations needed before termination
	# iterations = np.arange(1, epochs+1)
	# plt.plot(iterations, misclassifiedList, color="red")
	# plt.xlabel("Iterations")
	# plt.ylabel("Misclassified")

	plt.show()

