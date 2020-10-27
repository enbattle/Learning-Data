import matplotlib.pyplot as plt

'''
Function that takes in a line of data from the training/testing set and
creates a 16x16 grayscale image of the digit
'''
def createDigitImage(digit, linelist):
    tempList = []

    for i in range(1, len(linelist)):
        tempList.append(float(linelist[i]))
        if (len(tempList) == 16):
            digit.append(tempList)
            tempList = []

if __name__ == "__main__":
    '''
    Open data file that contains digit data about 1s and 5s with the first value in each line being
    the digit value, and the 256 values following that be the 16x16 grayscale image values 
    (on a scale from -1 to 1, -1 being dark pixels and 1 being light pixels)
    '''

    input = open("Only1sAnd5sTraining.txt", "r")
    # input = open("Only1sAnd5sTest.txt", "r")

    # Will hold 16x16 grayscale image of digit 1
    one = []

    # Will hold 16x16 grayscale image of digit 5
    five = []

    foundOne = False
    foundFive = False

    for line in input:
        lineList = line.strip().split(" ")
        tempList = []

        if(int(float(lineList[0])) == 5):
            createDigitImage(five, lineList)
            foundFive = True

        if (int(float(lineList[0])) == 1):
            createDigitImage(one, lineList)
            foundOne = True

        if(foundOne and foundFive):
            break

    # Plot the created handwritten digit 1
    plt.imshow(one, cmap="gray")

    # Plot the created handwritten digit 5
    # plt.imshow(five, cmap="gray")

    plt.show()
