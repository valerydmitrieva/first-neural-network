"""Run neural network. """

import os

import matplotlib
import matplotlib.pyplot
import numpy
import scipy

from conf import settings
from neural_network.core import NeuralNetwork
from neural_network.utils import prepare_images


def mnist_train(network):
    """Train the neural network on mnist dataset. """

    file_path = os.path.join(settings.MNIST_DATASET_DIR, 'mnist_train.csv')

    # load the mnist training data CSV file into a list
    with open(file_path, 'r') as training_data_file:
        training_data_list = training_data_file.readlines()

    # train the neural network

    for e in range(settings.EPOCHS):
        # go through all records in the training data set
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')

            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(settings.OUTPUT_NODES) + 0.01

            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            network.train(inputs, targets)

            ## create rotated variations
            # rotated anticlockwise by x degrees
            inputs_plusx_img = scipy.ndimage.interpolation.rotate(
                inputs.reshape(28, 28), 10, cval=0.01, order=1, reshape=False)
            network.train(inputs_plusx_img.reshape(784), targets)

            # rotated clockwise by x degrees
            inputs_minusx_img = scipy.ndimage.interpolation.rotate(
                inputs.reshape(28, 28), -10, cval=0.01, order=1, reshape=False)
            network.train(inputs_minusx_img.reshape(784), targets)


def mnist_test(network):
    """Test the neural network on other mnist dataset. """

    file_path = os.path.join(settings.MNIST_DATASET_DIR, 'mnist_test.csv')

    # load the mnist test data CSV file into a list
    with open(file_path, 'r') as test_data_file:
        test_data_list = test_data_file.readlines()

    # scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = network.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)

    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    print(f"performance = {scorecard_array.sum() / scorecard_array.size}")


def my_images_test(network):
    """Test the neural network with our own images. """

    dir_path = os.path.join(settings.BASE_DIR, 'neural_network/my_own_images')

    my_data_list = prepare_images(dir_path)

    for item in my_data_list:
        # plot image
        matplotlib.pyplot.imshow(item[1:].reshape(28, 28), cmap='Greys', interpolation='None')

        # correct answer is first value
        correct_label = item[0]
        print("correct label ", correct_label)

        # data is remaining values
        inputs = item[1:]

        # query the network
        outputs = network.query(inputs)

        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        print("network says ", label)

        # append correct or incorrect to list
        if (label == correct_label):
            print("match!")
        else:
            print("no match!")


def main():
    """Create, train and test neural network. """

    network = NeuralNetwork(
        settings.INPUT_NODES, settings.HIDDEN_NODES, settings.OUTPUT_NODES, settings.LEARNING_RATE
    )

    mnist_train(network)
    mnist_test(network)
    my_images_test(network)


if __name__ == '__main__':
    main()
