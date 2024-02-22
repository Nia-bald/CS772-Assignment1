import math
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Any

def sigmoid(x):
    # print(x)
    return 1/(1+np.exp(-x))
def modsigmoid(x):
    return 2/(1+math.exp(abs(x)))


class Value:

    def __init__(self, data, _children = (), _op='', label = ''):
        self.data = data
        self.grad = 0.0 # represents derivative of the parent node with respec to current node
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.label = label

    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0*out.grad  # out and self are addresses here, so if it gets executed in outer node then out == currentnode and self and other == children, so even if we are assigning a different address to out in current node, since out was used in this node, out will be current node when executing the function
            other.grad += 1.0*out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self, other), '*')
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other,(int, float))

        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other*(self.data**(other-1))*out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other): # other*self
        return self*other
    
    def __truediv__(self, other):
        return self*other**-1
    
    def __neg__(self):
        return self*-1

    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        return self + other

    

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - t**2)*out.grad
        out._backward = _backward
        return out
    def sin(self):
        x = self.data
        out = Value(math.sin(x), (self, ), 'sin')
        def _backward():
            self.grad += math.cos(x)*out.grad
        out._backward = _backward
        return out
    def cos(self):
        x = self.data
        out = Value(math.cos(x), (self, ), 'cos')
        def _backward():
            self.grad += -math.sin(x)*out.grad
        out._backward = _backward
        return out
    def tan(self):
        x = self.data
        out = Value(math.tan(x), (self, ), 'tan')
        def _backward():
            self.grad += (1/math.cos(x)**2)*out.grad
        out._backward = _backward
        return out
    def cot(self):
        x = self.data
        out = Value(math.cot(x), (self, ), 'cot')
        def _backward():
            self.grad += -(1/math.sin(x)**2)*out.grad
        out._backward = _backward
        return out
    def sinh(self):
        x = self.data
        out = Value(math.sinh(x), (self, ), 'sinh')
        def _backward():
            self.grad += math.cosh(x)*out.grad
        out._backward = _backward
        return out
    def cosh(self):
        x = self.data
        out = Value(math.cosh(x), (self, ), 'sinh')
        def _backward():
            self.grad += math.sinh(x)*out.grad
        out._backward = _backward
        return out
    

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data*out.grad
        out._backward = _backward
        return out
    def reLu(self):
        x = self.data
        out = Value(max(0, x), (self, ), 'reLu')
        def _backward():
            if x > 0:
                self.grad += out.grad
            else:
                self.grad += 0
        out._backward = _backward
        return out
    
    def sigmoid(self):
        x = self.data
        s = sigmoid(x)
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s*(1 - s)*out.grad
        out._backward = _backward
        return out
    def log(self):
        x = self.data
        # print(x)
        out = Value(math.log(x), (self,), 'log')

        def _backward():
            self.grad += (1/x)*out.grad
        out._backward = _backward
        return out
    
    def modsigmoid(self):
        x = self.data
        s = modsigmoid(x)
        out = Value(s, (self,), 'modsigmoid')

        def _backward():
            if x >= 0:
                self.grad += -((2*x)/(x*(1+x)**2))*out.grad
            else:
                self.grad += -((2*x)/(-x*(1-x)**2))*out.grad

        out._backward = _backward
        return out
        

    def sinc(self):
        if x == 0:
            print('error 0 not valdid input')
            return  
        x = self.data
        out = Value(math.sinx(x)/x, (self, ), 'sinc')
        def _backward():
            self.grad += ((2*x*math.sin(x) - (x**2)*math.cos(x))/(x**4))*out.grad
        out._backward = _backward
        return out
        
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
              visited.add(v)
              for child in v._prev:
                build_topo(child)
              topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
class Neuron:
    def __init__(self, nin, activation='sigmoid'):
        self.w = [Value(random.uniform(-2, 2)) for _ in range(nin)]
        self.b = Value(random.uniform(-2, 2))
        self.activation = activation
    
    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x): # Neuron()(x)

        act = sum((xi*wi for xi, wi in zip(x, self.w)), self.b)
        if self.activation == 'sigmoid':
            out = act.sigmoid()
        if self.activation == 'reLu':
            out = act.reLu()
        if self.activation == 'modsigmoid':
            out = act.modsigmoid()
        if self.activation == '':
            return act
        return out



class Layer:
    def __init__(self, nin, nout, activation='sigmoid'):
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs