import tensorflow
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

single_image = mnist.train.images[1].reshape(28,28)

plt.imshow(single_image, cmap= 'gist_gray')
plt.show()