import numpy as np
from sklearn.model_selection import train_test_split
from typing import Sequence, Optional
from logistic_regression import sigmoid, predict, binary_cross_entropy


class Neuron:
    """neuron that uses the sigmoid function for learning

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # creates a neuron
    # each neuron contains w and b
    # each neuron takes in gradient and updates w and b with it
    # each neuron outputs a new prediction 
    def __init__(self, features: int) -> None:
        self._w = np.zeros(features)
        self._b = 0
    
    @property
    def w(self):
        return self._w
    
    @property
    def b(self):
        return self._b
    
    @w.setter
    def w(self, w:np.ndarray):
        if w.ndim != 1:
            raise ValueError("w can only be 1d")
        self._w = w
    
    @b.setter
    def b(self, b: float):
        if not isinstance(b, float):
            raise ValueError("b can only be a float")
        self._b = b 
        
    def proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X, self.w, self.b)
    
    def predict(self, X:np.ndarray) -> int:
        """uses the neurons weights to predict y value for x

        Args:
            X (np.ndarray): data to predict as 2d array

        Returns:
            int: 1 or 0
        """
        return predict(X, self.w, self.b)
    
    def cost(self, X:np.ndarray, y:np.ndarray):
        return binary_cross_entropy(X, y, self.w, self.b)
    
class Layer:
    # each layer containes multiple neurons
    # a layer should take in a list of data, send them through each neuron 
    # and return their output
    def __init__(self, neuron_count: int) -> None:
        self._neuron_count = neuron_count
        self._neurons = []

    @property
    def neuron_count(self):
        return self._neuron_count
    
    @neuron_count.setter
    def neuron_count(self, count):
        self._neuron_count = count
    
    @property
    def neurons(self) -> Sequence[Neuron]:
        return self._neurons
    
    @neurons.setter
    def neurons(self, neurons: Sequence[Neuron]):
        ittr = isinstance(neurons, Sequence)
        neur = all(isinstance(n, Neuron) for n in neurons)
        if ittr is not True or neur is not True:
            raise ValueError("neurons can must be of type Sequence[Neuron]")
        
        self._neurons = neurons
        
    def compile(self, features: int) -> None:
        """adds all the neurons with weights set to 0 to the layer

        Args:
            features (int): features for w (X.shape[1])
        """
        self.neurons = [Neuron(features) for _ in range(self.neuron_count)]
    
    def proba(self, X:np.ndarray) -> np.ndarray: 
        """calculates the probability of all neurons

        Args:
            X (np.ndarray): data as 2d array

        Returns:
            np.ndarray: 1d array of neurons proba
        """
        return np.array([n.proba(X) for n in self.neurons]).T # transpose to get correct shape
    
    def cost(self, X:np.ndarray, y:np.ndarray):
        return np.mean([n.cost(X, y) for n in self.neurons])

class SigmoidNN:
    # creates a nerual network comprized of many layers.
    # containes multiple layers
    # function to train predict, and evaluate
    def __init__(self) -> None:
        self.layers = []
        
    def sequential(self, layers: list[Layer]):
        self.layers = layers
        
    def compile(self, features: int):
        features_current = features
        for layer in self.layers:
            layer.compile(features_current)
            features_current = layer.neuron_count

    def forward(self, X: np.ndarray, y: np.ndarray):   
        a = X # set initial a to X
        for i in range(len(self.layers)):
            layer = self.layers[i] 
            a = layer.proba(a)
                
    def predict(self, X:np.ndarray) -> np.ndarray:
        a = X # set initial a to X
        
        for i in range(len(self.layers)):
            layer = self.layers[i] 
            a = layer.proba(a)
        return a
    