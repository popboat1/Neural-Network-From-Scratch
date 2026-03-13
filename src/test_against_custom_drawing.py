from network import load_network
from customDrawing import MNISTDrawer

net = load_network(r'D:\Python\AI\Neural Network From Scratch\src\baseline_mnist_model.json')


drawer = MNISTDrawer(net)