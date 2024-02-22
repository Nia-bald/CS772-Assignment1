# CS772-Assignment 1: Implementation of Backpropagation and Training a Palindrome Network
1. Implement BP (using any existing tool/platform not allowed)
2. Think of the correct architecture for Palindrome
3. Train a feedforward n/w for solving the 10-bit palindrome problem (input- bit strings of 1 and 0); there will be 1024 input strings labeled 1 (if the string is Palindrome) and 0 (non-P)
4. Train and Test using 4-fold cross-validation
5. Measure Precision
6. Find out what the hidden layer neurons are doing (VIMP)

# Files specifications
1. Demo: contains all the code related to the DEMO presented in the presentation
2. mygrad: contains all the class and functions related to DAG backpropagation implementation
3. preparedata: code used to create Data for one neuron architecture
4. ReLuOneNueronArchi: __The Main Code containing the model architecture and The 4 fold cross validation evaluation results for one neuron architecture__
5. parameters: Contains saved weights and biases for all the models that we have tried
6. useModel: file containing code demonstrating how to use the saved parameter also contains ideal case demonstration
7. Extra: contains all the other architecture we tried and files and images used during the presentation

# Tools used
1. numpy: for exponentiation operation, as math gives overflow error
2. pickle: to save parameters and data
3. Pandas: to display data, not really needed but used it anyway
4. seaborn, sklearn and matplotlib: to plot the confusion matrix
5. streamlit: used for creating Demo


# Demo:
https://huggingface.co/spaces/Piyushmryaa/CS772_ASSIGNMENT1