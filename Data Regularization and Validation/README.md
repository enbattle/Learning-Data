# Data Regularization and Validation

Machine-learning classification of handwritten digit classification,
specifically between digit 1 and every other digit using regularization and validation. 

For each of the data (text) files: 
- First value in a line is the digit itself 
- All 256 values following is a 16x16 grayscale image of the handwritten digit

To find the difference between a handwritten digit 1 and the rest of the digits, we need to note two distinct features between the digits.

- Digit 1 will typically more symmetrical than the other digits (higher average symmetry value)
- Other digits will typically occupy more white pixels than Digit 1 (higher average intensity value)

Let: 
- g(x) := final hypothesis for Digit 1 or Not Digit 1
- := grayscale image pixel value in the list 
- := 1, 2, ...., 256

Average intensity mathematical definition: 
- g(x) = sum(x_1, x_2, ...., x_256) / 256 

Average symmetry mathematical definition: 
- g(x) = sum( abs(x_1 - flipImage(x_1)), ... , abs(x_256 - flipImage(x_256))) / 256
    
For examples:
- lambda0PlottingBoundary.png shows an example of a decision boundary plot for when lambda = 0.
- lambda0PlottingBoundary.png shows an example of a decision boundary plot for when lambda = 2. 
- handwrittenDigitComparison.py takes in the digit data (Digit 1 and Not Digit 1) from the text file, ZipDigits.txt, and creates the decision boundary for their classification and the cross validation graph.