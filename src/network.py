import numpy as np
import random
import matplotlib.pyplot as plt
import json

class QuadraticCost:
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y)**2
    
    @staticmethod
    def delta(z, a, y, activation):
        return (a - y) * activation.prime(z)
    
class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y, activation):
        return (a - y)
    
class Sigmoid:
    @staticmethod
    def fn(z):
        return 1.0/(1.0 + np.exp(-z))
    
    @classmethod
    def prime(cls, z):
        return cls.fn(z) * (1 - cls.fn(z))
    
class ReLU:
    @staticmethod
    def fn(z):
        return np.maximum(0.0, z)
    
    @staticmethod
    def prime(z):
        return (z > 0).astype(float)
    

class Network:
    def __init__(self, sizes, cost=CrossEntropyCost, activation=Sigmoid):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.activation = activation
        
        self.default_weight_initializer()
        
        self.evaluation_costs = []
        self.evaluation_accuracies = []
        self.training_costs = []
        self.training_accuracies = []
        
    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        
    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.activation.fn(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0, evaluation_data=None):
        n = len(training_data)
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
                
            train_cost = self.total_cost(training_data, lmbda)
            train_acc = self.accuracy(training_data, convert=True) / n
            self.training_costs.append(train_cost)
            self.training_accuracies.append(train_acc)
            
            if evaluation_data:
                n_data = len(evaluation_data)
                eval_cost = self.total_cost(evaluation_data, lmbda, convert=True)
                eval_acc = self.accuracy(evaluation_data) / n_data
                self.evaluation_costs.append(eval_cost)
                self.evaluation_accuracies.append(eval_acc)
                print(f"Epoch {j} training: Eval Accuracy: {eval_acc*100:.2f}% | Eval Loss: {eval_cost:.4f}")
            else:
                print(f"Train Accuracy: {train_acc*100:.2f}% | Train Loss: {train_cost:.4f}")
    
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [(1 - eta*(lmbda/n))*w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x]
        zs = []
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation.fn(z)
            activations.append(activation)
            
        delta = self.cost.delta(zs[-1], activation, y, self.activation)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation.prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    
    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
    def plot_performance(self):
        """Plots training vs evaluation loss and accuracy side-by-side."""
        epochs = range(len(self.training_costs))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(epochs, self.training_costs, label='Training Loss', color='blue')
        if self.evaluation_costs:
            axes[0].plot(epochs, self.evaluation_costs, label='Evaluation Loss', color='red')
        axes[0].set_title('Loss Over Epochs')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Cost')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(epochs, self.training_accuracies, label='Training Accuracy', color='blue')
        if self.evaluation_accuracies:
            axes[1].plot(epochs, self.evaluation_accuracies, label='Evaluation Accuracy', color='red')
        axes[1].set_title('Accuracy Over Epochs')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def save(self, filename):
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": self.cost.__name__,
            "activation": self.activation.__name__
        }
        with open(filename, "w") as f:
            json.dump(data, f)
            
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_network(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    
    cost = globals()[data["cost"]]
    activation = globals()[data["activation"]]
    
    net = Network(data["sizes"], cost=cost, activation=activation)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net