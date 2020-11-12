import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np
import random
import statistics

"""
Predict object needed by the plot_decision_regions function of the mlxtend.plotting library
    weight: The learned weight vector generalized for learning points
    order: nth order Legendre Transform    
"""
class Predict:
    def __init__(self, weight, order):
        self.weight = weight
        self.order = order

    def predict(self, X):
        return np.sign(np.dot(polynomialLegendreTransform(X, self.order), self.weight))

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

"""
Calculates the respective legendre dimension for a given polynomial transform dimension

For k in 1, 2, 3,....., n; n = power
Function: L_k(x) = (((2k - 1) / k) * x * L_k-1(x)) - (((k-1) / k) * L_k-2(x))
"""
def legendre(x, power):
    if(power == 0):
        return 1

    if(power == 1):
        return x

    legendreList = [0] * (power+1)
    legendreList[0] = 1
    legendreList[1] = x
    for i in range(2, power+1):
        legendreList[i] = (((2 * i - 1) / i) * x * legendreList[i-1]) - ((i-1) / i) * legendreList[i-2]

    return legendreList[power]

"""
Calculates the nth order Legendre Transform of a given vector, similar to an nth order polynomial transform

Form: (x1, x2) => (1,L1(x1),L1(x2),L2(x1),L1(x1)L1(x2),L2(x2),L3(x1),L2(x1)L1(x2), . . .)

    X: vector of datapoints (2-d)
    n: nth order of transform
"""
def polynomialLegendreTransform(X, n):
    phiX = []
    for i in range(len(X)):
        temp = []
        for j in range(n+1):
            x1 = X[i][0]
            x2 = X[i][1]

            x1Power = j
            x2Power = 0
            while(x1Power >= 0):
                temp.append(legendre(x1, x1Power) * legendre(x2, x2Power))
                x1Power -= 1
                x2Power += 1

        phiX.append(temp)

    return np.array(phiX)

"""
Calculates the learned linear regression weights with weight decay regularization to minimize Eout

Function: w_reg(lambda) = (((transpose(Z) * Z) + (lambda * I))^-1) * transpose(Z) * Y

    coords(Z): Legendre transformed vector of datapoints
    lamb: lambda value
    y: respective binary labels of each datapoint
"""
def linearRegression(coords, lamb, y):
    Z = coords

    transposeZ = np.transpose(Z)
    ZtransposeZ = np.dot(transposeZ, Z)
    lambdaIdentity = np.dot(lamb, np.identity(len(ZtransposeZ)))
    ZtransposeZinv = np.linalg.inv(ZtransposeZ + lambdaIdentity)
    ZtransposeZinvZtranpose = np.dot(ZtransposeZinv, transposeZ)
    weights = np.dot(ZtransposeZinvZtranpose, y)

    return weights

"""
Estimates the error of the linear regression with weight decay regularization using cross validation
    coords(Z): Legendre transformed vector of datapoints
    lamb: lambda value
    y: respective binary labels of each datapoint
"""
def crossValidation(coords, lamb, y):
    Z = coords

    transposeZ = np.transpose(Z)
    ZtransposeZ = np.dot(transposeZ, Z)
    lambdaIdentity = np.dot(lamb, np.identity(len(ZtransposeZ)))
    ZtransposeZinv = np.linalg.inv(ZtransposeZ + lambdaIdentity)
    ZZtransposeZinv = np.dot(Z, ZtransposeZinv)
    H = np.dot(ZZtransposeZinv, transposeZ)

    yhat = np.dot(H, y)

    error = 0
    for n in range(len(y)):
        error += ((yhat[n] - y[n]) / (1 - H[n][n]))**2

    error /= len(y)

    return error

"""
Estimates the error of the linear regression
    coords(Z): Legendre transformed vector of datapoints
    lamb: lambda value
    y: respective binary labels of each datapoint
"""
def crossTest(coords, lamb, y):
    Z = coords

    transposeZ = np.transpose(Z)
    ZtransposeZ = np.dot(transposeZ, Z)
    lambdaIdentity = np.dot(lamb, np.identity(len(ZtransposeZ)))
    ZtransposeZinv = np.linalg.inv(ZtransposeZ + lambdaIdentity)
    ZZtransposeZinv = np.dot(Z, ZtransposeZinv)
    H = np.dot(ZZtransposeZinv, transposeZ)

    yhat = np.dot(H, y)

    error = 0
    for n in range(len(y)):
        error += (yhat[n] - y[n]) ** 2

    error /= len(y)

    return error

if __name__ == "__main__":
    '''
    Open data file that contains digit data about 1s and 5s with the first value in each line being
    the digit value, and the 256 values following that be the 16x16 grayscale image values 
    (on a scale from -1 to 1, -1 being dark pixels and 1 being light pixels)
    '''
    input = open("ZipDigits.txt", "r")

    # Average intensity (as a list of x-coordinates) of digit 1
    oneX = []
    # Average symmetry (as a list of y-coordinates) of digit 1
    oneY = []

    # Average intensity (as a list of x-coordinates) of digit 5
    restX = []
    # Average symmetry (as a list of y-coordinates) of digit 5
    restY = []

    # Holds all x1,x2 coords
    allCoords = []
    allY = []

    # Loop through the file for every digit data
    for line in input:
        linelist = line.strip().split(" ")

        # Calculate the average intensity and average symmetry value for handwritten digits
        if (int(float(linelist[0])) != 1):
            intensityAndSymmetry(linelist, restX, restY, allCoords)
            allY.append(-1)

        # Calculate the average intensity and average symmetry value for a handwritten digit 1
        if (int(float(linelist[0])) == 1):
            intensityAndSymmetry(linelist, oneX, oneY, allCoords)
            allY.append(1)

    input.close()

    # Combine all the points together to and find the mean and standard deviation for x1 and x2 for normalization
    XCoord = oneX + restX
    YCoord = oneY + restY
    xMean = statistics.mean(XCoord)
    xDev = statistics.stdev(XCoord)
    yMean = statistics.mean(YCoord)
    yDev = statistics.stdev(YCoord)

    # Loop through the list of randomized data points, and do a min-max normalization to place points between [-1, 1]
    for i in range(len(allCoords)):
        tempX = (allCoords[i][0] - xMean) / (xDev * 4.3)
        tempY = (allCoords[i][1] - yMean) / (yDev * 4.3)
        allCoords[i] = [tempX, tempY]

    # Pick 300 random training points
    dtrain = []
    ytrain = []
    randomPoints = random.sample(list(enumerate(allCoords)), 300)
    for point in randomPoints:
        dtrain.append(allCoords[point[0]])
        ytrain.append(allY[point[0]])

    dtrain = np.array(dtrain)
    ytrain = np.array(ytrain)

    # Calculate the 8th order Legendre transform of each datapoint
    # Calculate the weight linear regression of the transform vectors (using lambda = 2)
    z = polynomialLegendreTransform(dtrain, 8)
    wreg = linearRegression(z, 0, ytrain)

    # Calculate the cross validation with respect to lambda
    lamb = np.arange(0, 2.01, 0.01)
    crossV = []
    for l in lamb:
        error = crossValidation(z, l, ytrain)
        crossV.append(error)

    crossT = []
    for l in lamb:
        error = crossTest(z, l, ytrain)
        crossT.append(error)

    # COMMENT THIS OUT WHEN YOU ARE PLOTTING THE CROSS VALIDATION
    # The training for the decision boundaries may take about 20 seconds to run
    # Plot the decision boundaries for overfitting and regularization
    clf = Predict(wreg, 8)
    plot_decision_regions(X=dtrain, y=ytrain.astype(int), clf=clf, legend=2)
    plt.xlabel("Average Intensity")
    plt.ylabel("Average Symmetry")
    plt.title("Digit 1 and and Not Digit 1 Comparison, Lambda = 2")
    plt.legend(loc="upper right")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # # COMMENT THIS OUT WHEN YOU ARE PLOTTING THE DECISION BOUNDARIES
    # # Plot the cross validation with respect to lambda
    # plt.plot(lamb, crossV, color="red", label="Ecv")
    # plt.plot(lamb, crossT, color="blue", label="Etest")
    # plt.xlabel("Lambda")
    # plt.ylabel("Error")
    # plt.title("Ecv and Etest Comparison")
    # plt.legend(loc="upper right")

    plt.show()
