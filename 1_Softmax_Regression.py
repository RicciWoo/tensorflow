# load MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# print dimension of train, test and validation data
print(mnist.train.images.shape, mnist.train.labels.shape) # (55000, 784) (55000, 10)
print(mnist.test.images.shape, mnist.test.labels.shape) # (10000, 784) (10000, 10)
print(mnist.validation.images.shape, mnist.validation.labels.shape) # (5000, 784) (5000, 10)

# load module, create an interactive session, create data matrix
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])

# define weight matrix and bias vector
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define respond vector
y = tf.nn.softmax(tf.matmul(x, W) + b)

# create vector of estimate of y
y_ = tf.placeholder(tf.float32, [None, 10])

# define loss function as cross entropy
cross_entropy = tf.reduce_mean(- tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# define optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# use global initializer to initialize
tf.global_variables_initializer().run()

# feed input, and run
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	train_step.run({x: batch_xs, y_: batch_ys})

# validation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
