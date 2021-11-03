"""Neaural network configuration. """

# number of input, hidden and output nodes
import os
from pathlib import Path

INPUT_NODES = 784
HIDDEN_NODES = 200
OUTPUT_NODES = 10

# learning rate
LEARNING_RATE = 0.01

# EPOCHS is the number of times the training data set is used for training
EPOCHS = 10

BASE_DIR = Path(__file__).resolve().parent.parent

MNIST_DATASET_DIR = os.path.join(BASE_DIR, 'neural_network/mnist_dataset')
