import numpy as np
import functools
import itertools as itr
import json

class ActivationFunction:
    @staticmethod
    def get(input: float) -> float:
        """Get the result of applying the activation function."""
        pass

    @staticmethod
    def getDeriv(input: float) -> float:
        """Get the local derivative (slope) of the activation function at the value of input."""
        pass

class Sigmoid(ActivationFunction):
    @staticmethod
    def get(input: float) -> float:
        return 1.0/(1.0+np.exp(-input))

    @staticmethod
    def getDeriv(input: float) -> float:
        t = Sigmoid.get(input)
        return t * (1.0-t)

class ReLu(ActivationFunction):
    @staticmethod
    def get(input: float) -> float:
        return max(0, input)

    @staticmethod
    def getDeriv(input: float) -> float:
        return 1.0 if input > 0 else 0
    
class LReLu(ActivationFunction):
    @staticmethod
    def get(input: float) -> float:
        return max(input, 0.1*input)

    @staticmethod
    def getDeriv(input: float) -> float:
        return 1.0 if input > 0 else 0.1
    
class Tanh(ActivationFunction):
    @staticmethod
    def get(input: float) -> float:
        eX = np.exp(input)
        eMX = np.exp(-input)
        return (eX - eMX) / (eX + eMX)

    @staticmethod
    def getDeriv(input: float) -> float:
        t = Tanh.get(input)
        return 1.0 - t**2
    
class Linear(ActivationFunction):
    @staticmethod
    def get(input: float) -> float:
        return input

    @staticmethod
    def getDeriv(input: float) -> float:
        return 1.0
        
class Layer:
    def __init__(self, *args, afunc = Sigmoid):
        # Assignment of the activation function
        self.afunc = afunc
        
        if args:
            # Set random weights and biases
            n_inputs = args[0]
            n_neurons = args[1]

            self.weights: np.ndarray = np.random.normal(0.0, 1.0, (n_neurons, n_inputs))
            # shape := n_neurons x [w(1), w(2), ..., w(n_inputs)] (numpy matrix)

            self.biases = np.zeros(n_neurons)
            # shape := [b(1), b(2), ..., b(n_neurons)]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # FeedForward
        self.z: np.ndarray = self.weights @ inputs + self.biases
        self.output: np.ndarray = self.afunc.get(self.z)
        return self.output
        # Activation function is applied to every component of the dot product
        # Dot product of weights and inputs: For every neuron the dot product of its weights and the inputs

    def mutated(self, amount: float):
        newLayer = Layer(afunc=self.afunc)
        newLayer.weights = self.weights + np.random.uniform(-amount, amount, self.weights.shape)
        newLayer.biases = self.biases + np.random.uniform(-amount, amount, self.biases.shape)
        return newLayer
    

class NeuralNetwork:
    def __init__(self, *args, afunc=Sigmoid):
        """Randomly initialize the network given its layer sizes."""
        self.shape: list[int] = args
        self.size: int = len(args)

        # Initiate layers randomly
        self.layers: list[Layer] = [ Layer(inp, out, afunc=afunc) 
                        for inp, out in zip(args[:-1], args[1:])]
        # shape := [layer(s1,s2), layer(s2,s3), ..., layer(sn-1,sn)]

    def run(self, inputs: list[float]) -> np.array:
        """Get the model's outputs to the given inputs."""
        self.netinputs: np.ndarray = np.array(inputs)
        
        # Forward through each layer
        return functools.reduce(lambda inp, layer: layer.forward(inp), self.layers, self.netinputs)
    
    def mutated(self, amount: float = 0.05):
        """Get a mutated copy of the network, applicable in Neuroevolution"""
        newNetwork = NeuralNetwork()
        newNetwork.shape = self.shape
        newNetwork.size = self.size
        newNetwork.layers = [layer.mutated(amount=amount) for layer in self.layers]
        return newNetwork

    def gradient(self, input: list[float], label: list[float]) -> (list[np.mat], list[np.array]):
        """Compute the delta for the GRADIENT DESCENT based on a single training case."""
        # Get the output
        output: np.ndarray = self.run(input)
        label = np.array(label)
        self.error: float = (output - label) ** 2

        activations: list[np.ndarray] = [l.output for l in self.layers[-2::-1]] + [self.netinputs]

        # Calculate the gradient for the last layer's activation
        lastLayer: Layer = self.layers[-1]
        delta_layer_z: np.ndarray = 2*(output - label) * lastLayer.afunc.getDeriv(lastLayer.z)

        # Calculate the gradient compnent for the weights and biases for this example
        delta_w = []
        delta_b = []

        for curL, prevL in itr.pairwise(reversed(self.layers)):
            l_delta_w = np.outer(delta_layer_z, prevL.output)
            # This is the gradient of the layer's weights
            delta_w.append(l_delta_w)

            l_delta_b = delta_layer_z
            # This is the gradient of the layer's biases
            delta_b.append(l_delta_b)

            delta_layer_z = prevL.afunc.getDeriv(prevL.z) * (delta_layer_z @ curL.weights)
            # This is the gradient of the previous layer's pre-activation (z)
        
        first_delta_w = np.outer(delta_layer_z, self.netinputs)
        # This is the gradient of the first layer's weights
        delta_w.append(first_delta_w)

        first_delta_b = delta_layer_z
        # This is the gradient of the first layer's biases
        delta_b.append(first_delta_b)

        # Return the two gradient components for the network's weights and biases
        return delta_w[::-1], delta_b[::-1]
    
    def batch_update(self, batch: list[(list[float], list[float])], learning_rate: float = 0.15, log: bool = False):
        """Update the network's parameters by applying gradient descent using backpropagation to a training batch.
        The training set is a list of tuples (of arrays) '(inp, label)', and 'lr' is the learning rate"""

        gradient_w = [np.zeros(l.weights.shape) for l in self.layers]
        gradient_b = [np.zeros(l.biases.shape) for l in self.layers]
        # Together, these form the "gradient descent" of the Neural Network

        if log: error = np.zeros(self.layers[-1].biases.shape)

        for input, label in batch:
            # Get the gradient components of this test case
            delta_w, delta_b = self.gradient(input, label)
            if log: error += self.error.flatten()

            # Add them to the final gradient descent
            gradient_w = [acc + d for acc, d in zip(gradient_w, delta_w)]
            gradient_b = [acc + d for acc, d in zip(gradient_b, delta_b)]
        
        # Divide by the batch size to get the average for the gradient.
        gradient_w = [delta / len(batch) for delta in gradient_w]
        gradient_b = [delta / len(batch) for delta in gradient_b]

        if log:
            error /= len(batch)
            print(f"Error: {np.average(error)}")

        # Change the network's parameters by the NEGATIVE GRADIENT (GRADIENT DESCENT) multiplied by the learning rate
        for layer, delta_w, delta_b in zip(self.layers, gradient_w, gradient_b):
            layer.weights -= delta_w * learning_rate
            layer.biases -= delta_b.flatten() * learning_rate
    
    def toJSON(self) -> str:
        return json.dumps(self)
    
    @staticmethod
    def fromJSON(json: str):
        return json.loads(json, object_hook=lambda d: NeuralNetwork(**d))

class Normalizer:
    def __init__(self, arr: np.ndarray) -> None:
        self.mean = np.mean(arr)
        self.std = np.std(arr)

    def encode(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / self.std
    
    def decode(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.std + self.mean
    