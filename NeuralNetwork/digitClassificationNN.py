import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np
import random
import statistics
import math
import neuralNetwork
import time

random.seed(50)

class Predict:
    """
    Predict class used as the classifier for plotting decision boundaries.
    References the Neural Network class to use its predict class method
    """

    def __init__(self, neuralNet):
        self.neuralNet = neuralNet

    def predict(self, X):
        output = []
        for x in X:
            output.append(self.neuralNet.predict(x))

        return np.array(output)

'''
Takes a list of lists, representing a (n x m) grayscale image, and flips it over the vertical axis
    n := rows
    m := columns
'''
def flipImage(image):
    start = 0
    end = len(image) - 1

    # Run loop to switch image[start] and image[end] rows until start++ and end--
    #   pass each other (when n = even) or they equal each other (when n = odd)
    while (start < end):
        tempList = image[start]
        image[start] = image[end]
        image[end] = tempList
        start += 1
        end -= 1

'''
Calculate the average intensity and average symmetry for digits 1 and 5
    linelist := 16x16 grayscale image (represented as a list of lists)
    digitListX := list of calculated average intensity of the digits
    digitListY := list of calculated average symmetry of the digits
'''
def intensityAndSymmetry(linelist, digitListX, digitListY, allCoords):
    digitList = []
    intensity = 0

    tempList = []

    for i in range(1, len(linelist)):
        tempList.append(float(linelist[i]))

        # Add to intensity
        intensity += float(linelist[i])

        # Add row of 16 grayscale values to the overall list
        if (len(tempList) == 16):
            digitList.append(tempList)
            tempList = []

    # Calculate the average intensity as the average
    averageIntensity = intensity / len(linelist)

    # Save the average intensity as an x-coordinate value
    digitListX.append(averageIntensity)

    # Make a copy of the grayscale values for the original image
    digitCopy = digitList.copy()

    # Flip the image over horizontal axis
    flipImage(digitList)

    # Calculate asymmetry as the absolute difference between an image and its flipped version
    #   and symmetry being the negation of asymmetry
    asymmetryValue = 0
    for i in range(len(digitList)):
        for j in range(len(digitList[i])):
            asymmetryValue += abs(digitCopy[i][j] - digitList[i][j])
    averageAsymmetry = asymmetryValue / len(digitList)

    # Save the average symmetry as an y-coordinate value
    digitListY.append(-averageAsymmetry)

    allCoords.append([averageIntensity, -averageAsymmetry])

if __name__ == "__main__":
    '''
    Open data file that contains digit data about 1s and 5s with the first value in each line being
    the digit value, and the 256 values following that be the 16x16 grayscale image values 
    (on a scale from -1 to 1, -1 being dark pixels and 1 being light pixels)
    '''

    trainInput = open("DigitTrain.txt", "r")
    testInput = open("DigitTest.txt", "r")

    # Average intensity (as a list of x-coordinates) of digit 1
    oneX = []
    testOneX = []

    # Average symmetry (as a list of y-coordinates) of digit 1
    oneY = []
    testOneY = []

    # Average intensity (as a list of x-coordinates) of digit 5
    restX = []
    testRestX = []

    # Average symmetry (as a list of y-coordinates) of digit 5
    restY = []
    testRestY = []

    # Holds all x1,x2 coords
    allCoords = []
    allY = []
    testCoords = []
    testY = []

    # Loop through the train data file for every digit data
    for line in trainInput:
        linelist = line.strip().split(" ")

        # Calculate the average intensity and average symmetry value for handwritten digits
        if (int(float(linelist[0])) != 1):
            intensityAndSymmetry(linelist, restX, restY, allCoords)
            allY.append(-1)

        # Calculate the average intensity and average symmetry value for a handwritten digit 1
        if (int(float(linelist[0])) == 1):
            intensityAndSymmetry(linelist, oneX, oneY, allCoords)
            allY.append(1)

    # Loop through the test data file for every digit data
    for line in testInput:
        linelist = line.strip().split(" ")

        # Calculate the average intensity and average symmetry value for handwritten digits
        if (int(float(linelist[0])) != 1):
            intensityAndSymmetry(linelist, testOneX, testOneY, testCoords)
            testY.append(-1)

        # Calculate the average intensity and average symmetry value for a handwritten digit 1
        if (int(float(linelist[0])) == 1):
            intensityAndSymmetry(linelist, testRestX, testRestY, testCoords)
            testY.append(1)

    trainInput.close()
    testInput.close()

    # Combine all the points together to and find the mean and standard deviation for x1 and x2 for normalization
    XCoord = oneX + restX
    YCoord = oneY + restY
    xMean = statistics.mean(XCoord)
    xDev = statistics.stdev(XCoord)
    yMean = statistics.mean(YCoord)
    yDev = statistics.stdev(YCoord)

    testXCoord = testOneX + testRestX
    testYCoord = testOneY + testRestY
    testXMean = statistics.mean(testXCoord)
    testXDev = statistics.stdev(testXCoord)
    testYMean = statistics.mean(testYCoord)
    testYDev = statistics.stdev(testYCoord)

    # Loop through the list of randomized data points, and do a min-max normalization to place points between [-1, 1]
    for i in range(len(allCoords)):
        tempX = (allCoords[i][0] - xMean) / (xDev * 4.3)
        tempY = (allCoords[i][1] - yMean) / (yDev * 4.3)
        allCoords[i] = [tempX, tempY]

    for i in range(len(testCoords)):
        tempX = (testCoords[i][0] - testXMean) / (testXDev * 4.3)
        tempY = (testCoords[i][1] - testYMean) / (testYDev * 4.3)
        testCoords[i] = [tempX, tempY]

    # Pick 300 random training points and 300 testing points
    dtrain = []
    ytrain = []
    dtest = []
    ytest = []

    # Choose random train and test samples
    randomPoints = random.sample(list(enumerate(allCoords)), 300)
    randomTest = random.sample(list(enumerate(testCoords)), 200) 

    for point in randomPoints:
        dtrain.append(allCoords[point[0]])
        ytrain.append(allY[point[0]])

    for point in randomTest:
        dtest.append(testCoords[point[0]])
        ytest.append(testY[point[0]])

    dtrain = np.array(dtrain)
    ytrain = np.array(ytrain)
    dtest = np.array(dtest)
    ytest = np.array(ytest)

    # Constants used for the neural network
    layers = [2, 10, 1]
    learning_rate = 0.001
    epochs = 100
    lamb = 0.01
    output_activation = "identity"

    # Note start time
    start_time = time.time()

    """
    Neural Net (No Weight Decay or Validation)
    """
    myNeuralNetwork = neuralNetwork.neuralNetwork(layers, learning_rate, output_activation)
    Ein, W = myNeuralNetwork.train(list(zip(dtrain, ytrain)), epochs)
    print("Calculation time:", time.time() - start_time, " seconds")
    clf = Predict(myNeuralNetwork)
    plot_decision_regions(X=dtrain, y=ytrain, clf=clf)
    plt.title("Digit 1 and and Not Digit 1 Comparison")
    plt.legend(loc="upper right")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig("Decision Boundary")

    """
    Neural Net with Weight Decay
    """
    # myNeuralNetwork = neuralNetwork.neuralNetwork(layers, learning_rate, output_activation)
    # Ein, W = myNeuralNetwork.trainWeightDecay(list(zip(dtrain, ytrain)), lamb, epochs)
    # print("Calculation time:", time.time() - start_time, " seconds")
    # clf = Predict(myNeuralNetwork)
    # plot_decision_regions(X=dtrain, y=ytrain, clf=clf)
    # plt.title("Digit 1 and and Not Digit 1 Comparison (Weight Decay)")
    # plt.legend(loc="upper right")
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # plt.savefig("Decision Boundary Weight Decay")

    """
    Neural Net with Validation Set
    """
    # dtrainTrain = np.array(dtrain[0:len(dtrain) - len(dtrain)//6])
    # dtrainVal = np.array(dtrain[len(dtrain)//6:len(dtrain)])
    # ytrainTrain = np.array(ytrain[0:len(dtrain) - len(dtrain)//6])
    # ytrainVal = np.array(ytrain[len(dtrain)//6:len(dtrain)])
    # myNeuralNetwork = neuralNetwork.neuralNetwork(layers, learning_rate, output_activation)
    # Ein, W = myNeuralNetwork.trainValidation(list(zip(dtrainTrain, ytrainTrain)), list(zip(dtrainVal, ytrainVal)), epochs)
    # print("Calculation time:", time.time() - start_time, " seconds")
    # clf = Predict(myNeuralNetwork)
    # plot_decision_regions(X=dtrain, y=ytrain, clf=clf)
    # plt.title("Digit 1 and and Not Digit 1 Comparison (Early Stopping)")
    # plt.legend(loc="upper right")
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # plt.savefig("Decision Boundary Early Stopping")

    """
    Plotting the Ein vs Iterations
    """
    # ein = [e[0] for e in Ein]
    # iterations = range(len(Ein))
    # plt.plot(iterations, ein, color="red")
    # plt.xlabel("Iterations")
    # plt.ylabel("Ein")
    # plt.title("Ein(w) vs Iterations")
    # plt.savefig("Ein and Iterations.png")
