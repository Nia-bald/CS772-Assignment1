# CS772-Assignment 1: Implementation of Backpropagation and Training a Palindrome Network
1. Implement BP (using any existing tool/platform not allowed)
2. Think of the correct architecture for Palindrome
3. Train a feedforward n/w for solving the 10-bit palindrome problem (input- bit strings of 1 and 0); there will be 1024 input strings labeled 1 (if the string is Palindrome) and 0 (non-P)
4. Train and Test using 4-fold cross-validation
5. Measure Precision
6. Find out what the hidden layer neurons are doing (VIMP)
## To do:
1. finalize the architecture
3. see the difference when momentum term is introduced
4. update the PPT
5. give a proper analysis on results which are wrong


### Insights
1. Xoring input at opposing ends of input might work but since xoring operation is not linearly separable it cannot be done by one neuron