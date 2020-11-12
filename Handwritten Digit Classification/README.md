# Handwritten Digit Classification

Machine-learning classification of handwritten digit classification,
specifically between digit 1 and digit 5. 

For each of the data (text) files: 
- First value in a line is the digit itself 
- All 256 values following is a 16x16 grayscale image of the handwritten digit

To find the difference between a handwritten digit 1 and 5, we need to note two distinct features
between a 1 and a 5.

- Digit 1 will typically more symmetrical than Digit 5 (higher average symmetry value)
- Digit 5 will typically occupy more white pixels than Digit 1 (higher average intensity value)

Let: 
- g(x) := final hypothesis for Digit 1 or 5 
- := grayscale image pixel value in the list 
- := 1, 2, ...., 256

Average intensity mathematical definition: 
- g(x) = sum(x_1, x_2, ...., x_256) / 256 

Average symmetry mathematical definition: 
- g(x) = sum( abs(x_1 - flipImage(x_1)), ... , abs(x_256 - flipImage(x_256))) / 256
    
For examples:
- Digit1Example.png shows an example of a 16x16 grayscale image of a handwritten digit 1.
- Digit5Exampl.png shows an example of a 16x16 grayscale image of a handwritten digit 5. 
- handwrittenDigits.py takes in the digit data from the text files in order to create the images.
- compare1And5.py takes in the digit data from the text files and creates a scatterplot showing the distinct 
classification between a digit 1 and digit 5 (along with some outliers).