import streamlit as st
from mygrad import Layer, Value
import pickle
# Define the predict function
def predict(x):
    x1 = hiddenLayer1(x)    
    final = outputLayer([x1] + x)
    return final.data

# Load model
def loadModel():
    neuron1weightsbias, outputneuronweightsbias = [], []
    with open(f'parameters/neuron1weightsbias_fn_reLu.pckl', 'rb') as file:
        neuron1weightsbias = pickle.load(file)
    with open('parameters/outputneuronweightsbias_fn_reLu.pckl', 'rb') as file:
        outputneuronweightsbias = pickle.load(file)
    hiddenLayer1_ = Layer(10, 1, 'reLu')
    outputLayer_ = Layer(11, 1, 'sigmoid')

    hiddenLayer1_.neurons[0].w = [Value(i) for i in neuron1weightsbias[:-1]]
    hiddenLayer1_.neurons[0].b = Value(neuron1weightsbias[-1])

    outputLayer_.neurons[0].w = [Value(i) for i in outputneuronweightsbias[:-1]]
    outputLayer_.neurons[0].b = Value(outputneuronweightsbias[-1])
    return hiddenLayer1_, outputLayer_

hiddenLayer1, outputLayer = loadModel()

st.title("Neural Network Prediction")

st.header("Input")
inputs = st.text_input("Input 10 digits Binary no")
input = []
flag = 0
if len(inputs)!=10:
    st.write("Error: Input not equal to 10 bits")
    flag =1
for i in inputs:
    if i!='0' and i!='1':
        st.write("Please input Binary number only")
        flag = 1
    else:
        input.append(int(i))

# Prediction
if st.button("Predict"):
    if flag:
        st.stop()
    result = predict(input)
    st.success(f"The prediction is: {result}")
