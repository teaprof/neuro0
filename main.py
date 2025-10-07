import torch.nn as nn
import numpy as np

size = 10

def gen(type):
    if type == 0:
        nn = np.arange(0, size, 1, dtype=np.float16)/size
    else:
        nn = np.arange(size, 0, -1, dtype=np.float16)/size
    return (np.random.rand(size) < nn).astype(np.int8)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
       
        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 3
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # Second 2D convolutional layer, taking in the 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        # First fully connected layer
        self.fc1 = nn.Linear(9216, 128)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(128, 10)


print(gen(0))
print(gen(1))