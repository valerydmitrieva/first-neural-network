import os

import imageio
import numpy


def prepare_images(path_to_images):
    """Transform images to dataset. """

    # our own image test data set
    our_own_dataset = []

    # load the png image data as test data set
    for image_file_name in os.listdir(path_to_images):
        # use the filename to set the correct label
        label = int(image_file_name[0])

        # load image data from png files into an array
        print(f"loading {image_file_name}")
        file_path = os.path.join(path_to_images, image_file_name)
        img_array = imageio.imread(file_path, as_gray=True)

        # reshape from 28x28 to list of 784 values, invert values
        # ordinary: 0 - black, 255 - white; in mnist dataset reverse
        img_data = 255.0 - img_array.reshape(784)

        # then scale data to range from 0.01 to 1.0
        img_data = (img_data / 255.0 * 0.99) + 0.01

        # append label and image data  to test data set
        record = numpy.append(label, img_data)
        our_own_dataset.append(record)

    return our_own_dataset
