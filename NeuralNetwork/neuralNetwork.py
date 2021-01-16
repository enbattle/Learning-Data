import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np
import random
import statistics
import math
import threading
import time
import csv

class neuralNetwork:
    """
    Neural Network Class
        - layers: layer vector input that defines the number of nodes for each layer
        - learning rate: eta value to determine learning rate (changes with variable learning gradient descent)
        - output_activation: activation function for the output layer of the neural network
    """

    def __init__(self, layers, learning_rate, output_activation):
        self.outputActivation = output_activation
        self.layers = layers
        self.learning_rate = learning_rate

        # Initialize the weights based on the input layer vector
        self.weights = [np.full((self.layers[i]+1, self.layers[i+1]), random.uniform(0, 1))
                        for i in range(len(self.layers)-1)]
        
    def frontPropogation(self, x):
        """
        Front Propogate through the neural net starting with the input datapoint
        """

        # Initialize activated weighted inputs
        X = [x]

        # Loop and multiply inputs based on the current weight vector
        for i in range(len(self.weights)):
            X[-1] = np.insert(X[-1], 0, [1])
            X[-1] = np.transpose(np.array([X[-1]]))

            # Calculate the signal using the indicated weights
            s = np.dot(np.transpose(self.weights[i]), X[-1])

            # Check activations for if we have reached the output layer or we are still in the middle layers
            if(i == len(self.weights) - 1):
                if(self.outputActivation == "identity"):
                    s = self.identity(s)
                elif(self.outputActivation == "tanh"):
                    s = self.tanh(s)
                elif(self.outputActivation == "sign"):
                    s = self.sign(s)
                else:
                    s = self.tanh(s)
            else:
                s = self.tanh(s)

            # Append activated weighted signal
            X.append(s)

        # Return the activated weight inputs
        return X

    def backPropagation(self, x, y):
        """
        Obtain delta values based on the final output value resulting from front propogation
        """

        # Initialize delta array
        deltas = [0] * (len(self.weights) + 1)

        # Call front propogation
        X = self.frontPropogation(x)

        # Set up the delta for the last layer
        if(self.outputActivation == "tanh"):
            deltas[-1] = 2.0 * (X[-1] - y) * (1.0 - X[-1]**2)
        else:
            deltas[-1] = 2.0 * (X[-1] - y)

        # Backward Propogation with the derivative of tanh as the middle layer activations
        for i in range(len(deltas)-2, 0, -1):
            theta_s = (1.0 - np.multiply(X[i], X[i]))[1:self.layers[i]+1]
            weight_delta = (self.weights[i] * deltas[i+1])[1:self.layers[i]+1]
            deltas[i] = np.multiply(theta_s, weight_delta)

        # Return the delta array
        return (X, deltas)

    def gradientDescent(self, datapoints):
        """
        Calculates the Ein value and current Ein gradients based on current weights
        """

        # Initialize Ein and gradient array
        ein = 0
        gradients = [0] * (len(self.weights) + 1)

        # Calculate the gradient
        for z in range(len(datapoints)):
            X, deltas = self.backPropagation(datapoints[z][0], datapoints[z][1]) # Call back propogation

            # Update the Ein
            ein = ein + ((X[-1] - datapoints[z][1])**2) / (4 * len(datapoints))

            # Update the gradient array
            for i in range(1, len(deltas)):
                gx = np.dot(X[i-1], np.transpose(deltas[i])) / (4 * len(datapoints))
                gradients[i] = gradients[i] + gx

        # Return the in sample error and gradients
        return (ein, gradients)

    def train(self, datapoints, epochs=10):
        """
        Train the input data (datapoints and labels) using variable learning gradient descent algorithm
        """

        # Initialize some constants
        alpha = 1.1
        beta = 0.5
        learning_rate = self.learning_rate
        einList = []
        min_ein = math.inf
        coords = datapoints
        
        for e in range(epochs):
            # Randomize the datapoints and labels
            random.shuffle(coords)

            print("Training Regular Iteration:", e)

            # Find the current in sample error and gradients
            currentEin, currentGradients = self.gradientDescent(coords)
            v = [-1 * currentGradients[i] for i in range(len(currentGradients))]

            # Temporary weights that change based on learning rate and the calculated gradients
            newWeights = [self.weights[i] + (learning_rate * v[i + 1]) for i in range(len(self.weights))]
            einList.append(currentEin)

            # If calculated Ein is less than the minimum Ein found, update learning rate by alpha and weights
            # Else, update learning rate by beta
            if(currentEin < min_ein):
                learning_rate *= alpha
                self.weights = newWeights
                min_ein = currentEin
            else:
                learning_rate *= beta

        # Return the list of in sample errors and ending weights
        return (einList, self.weights)

    def trainValidation(self, datapoints, validationpoints, epochs=10):
        """
        Train the input data (datapoints and labels) using variable learning gradient descent algorithm
        along with a validation set to determine early stopping
        """

        # Initialize some constants
        alpha = 1.1
        beta = 0.5
        learning_rate = self.learning_rate
        einList = []
        min_ein = math.inf
        coords = datapoints
        
        for e in range(epochs):
            random.shuffle(coords)

            print("Training Validation Iteration:", e)

            # Find the current in sample error and gradients
            currentEin, currentGradients = self.gradientDescent(coords)
            v = [-1 * currentGradients[i] for i in range(len(currentGradients))]

            # Temporary weights that change based on learning rate and the calculated gradients
            newWeights = [self.weights[i] + (learning_rate * v[i + 1]) for i in range(len(self.weights))]
            einList.append(currentEin)

            # If calculated Ein is less than the minimum Ein found, update learning rate by alpha and weights
            # Else, update learning rate by beta
            if(currentEin < min_ein):
                learning_rate *= alpha
                self.weights = newWeights
                min_ein = currentEin
            else:
                learning_rate *= beta

            # Early Stopping
            if(self.validate(validationpoints) < 1/len(validationpoints)):
                print("Epochs stopping early... current iteration:", e);
                break

        # Return the list of in sample errors and ending weights
        return (einList, self.weights)

    def gradientDescentWeightDecay(self, datapoints, lamb):
        # Initialize Eaug and gradient array
        eaug = 0
        gradients = [0] * (len(self.weights) + 1)

        # Calculate the gradient
        for z in range(len(datapoints)):
            X, deltas = self.backPropagation(datapoints[z][0], datapoints[z][1]) # Call back propogation

            # Update the Eaug
            eaug = eaug + \
                (((X[-1] - datapoints[z][1])**2) / (4 * len(datapoints))) + \
                (lamb * np.sum([np.sum(np.square(self.weights[i])) for i in range(len(self.weights))]) / (4 * len(datapoints)))

            # Update the gradient array
            for i in range(1, len(deltas)):
                gx = (np.dot(X[i-1], np.transpose(deltas[i])) / (4 * len(datapoints))) + (((2 * lamb) * self.weights[i-1]) / (4 * len(datapoints)))
                gradients[i] = gradients[i] + gx

        # Return augmented in sample error and gradients
        return (eaug, gradients)

    def trainWeightDecay(self, datapoints, lamb=0.01, epochs=10):
        """
        Train the input data (datapoints and labels) using variable learning gradient descent algorithm
        and weight decay
        """

        # Initialize some constants
        alpha = 1.1
        beta = 0.5
        learning_rate = self.learning_rate
        eaugList = []
        min_eaug = math.inf
        coords = datapoints
        
        for e in range(epochs):
            random.shuffle(coords)

            print("Training Weight Decay Iteration:", e)

            # Find the current in sample error and gradients
            currentEaug, currentGradients = self.gradientDescentWeightDecay(coords, lamb)
            v = [-1 * currentGradients[i] for i in range(len(currentGradients))]

            # Temporary weights that change based on learning rate and the calculated gradients
            newWeights = [self.weights[i] + (learning_rate * v[i + 1]) for i in range(len(self.weights))]
            eaugList.append(currentEaug)

            # If calculated Eaug is less than the minimum Eaug found, update learning rate by alpha and weights
            # Else, update learning rate by beta
            if(currentEaug < min_eaug):
                learning_rate *= alpha
                self.weights = newWeights
                min_eaug = currentEaug
            else:
                learning_rate *= beta

        # Return the list of in sample errors and ending weights
        return (eaugList, self.weights)

    def validate(self, datapoints):
        """
        Calculates the current validation for the validation data set to determine early stopping
        """

        incorrect = 0
        for x, y in datapoints:
            if(self.predict(x) != y):
                incorrect += 1
        return incorrect/len(datapoints)

    def predict(self, x):
        """
        Prediction function for plotting decision boundaries
        """

        X = self.frontPropogation(x)
        return self.sign(X[-1])

    def tanh(self, x):
        return np.tanh(x)

    def identity(self, x):
        return x

    def sign(self, x):
        return np.sign(x)