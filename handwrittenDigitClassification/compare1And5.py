import matplotlib.pyplot as plt

'''
Takes a list of lists, representing a (n x m) grayscale image, and flips it over the vertical axis
    n := rows
    m := columns
'''
def flipImage(image):
    start = 0
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
def intensityAndSymmetry(linelist, digitListX, digitListY):
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

if __name__ == "__main__":
    '''
    Open data file that contains digit data about 1s and 5s with the first value in each line being
    the digit value, and the 256 values following that be the 16x16 grayscale image values 
    (on a scale from -1 to 1, -1 being dark pixels and 1 being light pixels)
    '''
    input = open("Only1sAnd5sTraining.txt", "r")
    # input = open("Only1sAnd5sTest.txt", "r")

    # Average intensity (as a list of x-coordinates) of digit 1
    oneX = []
    # Average symmetry (as a list of y-coordinates) of digit 1
    oneY = []

    # Average intensity (as a list of x-coordinates) of digit 5
    fiveX = []
    # Average symmetry (as a list of y-coordinates) of digit 5
    fiveY = []

    # Loop through the file for every digit data
    for line in input:
        linelist = line.strip().split(" ")

        # Calculate the average intensity and average symmetry value for a handwritten digit 5
        if (int(float(linelist[0])) == 5):
            intensityAndSymmetry(linelist, fiveX, fiveY)

        # Calculate the average intensity and average symmetry value for a handwritten digit 1
        if (int(float(linelist[0])) == 1):
            intensityAndSymmetry(linelist, oneX, oneY)

    # Plot the values as a scatterplot, with the x-axis being the average intensity, and the
    #   y-axis being the average symmetry value, and see the classification separation
    #   between the 1s and the 5s
    plt.xlabel("Average Intensity")
    plt.ylabel("Average Symmetry")
    plt.title("Digit 1 and 5 Comparison")
    plt.scatter(oneX, oneY, s=20, color="blue", marker="o", label="digit 1")
    plt.scatter(fiveX, fiveY, s=20, color="red", marker="x", label="digit 5")
    plt.legend(loc="upper right")
    plt.show()
