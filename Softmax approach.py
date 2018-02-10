import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

single_image = mnist.train.images[1].reshape(28,28)

#plt.imshow(single_image, cmap= 'gist_gray')
#plt.show()

# PLACEHOLDERS
x = tf.placeholder(tf.float32,shape=[None,784])

# VARIABLES
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# CREATE GRAPH OPERATIONS
y = tf.matmul(x,W) + b

# LOSS FUNCTION
y_true = tf.placeholder(tf.float32, [None,10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))

# OPTIMIZER
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
train = optimizer.minimize(cross_entropy)

# CREATE SESSION
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):

        batch_x, batch_y = mnist.train.next_batch(100)

        sess.run(train,feed_dict={x:batch_x, y_true:batch_y})

    # EVALUATE THE MODEL

    correct_prediction = tf.equal(tf.argmax(y,axis=1), tf.argmax(y_true,1))

    # [True,False,True...] --> [1,0,1...]

    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # THIS IS WHAT IS HAPPENING:
    # PREDICTED [3,4] TRUE [3,9]
    # [TRUE, FALSE]
    # [1.0,0.0]
    # acc = 0.5

    print(sess.run(acc,feed_dict={x:mnist.test.images, y_true:mnist.test.labels}))