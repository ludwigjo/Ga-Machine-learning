from __future__ import print_function
import tensorflow as tf

import numpy as np
from PIL import Image

tf.logging.set_verbosity(tf.logging.ERROR)

learning_rate = 0.001
n_steps = 500
batch_size = 128
display_step = 100

n_input = 784		#28*28
hidden_l1 = 256
hidden_l2 = 256
n_labels = 10		#0-9
dropout = 0.5

from tensorflow.examples.tutorials.mnist import input_data			#importing mnist dataset
#from tf.data import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_labels])
keep_prob = tf.placeholder(tf.float32)

#weights = {
#	'w1': tf.Variable(tf.random_normal([n_input, hidden_l1])),
#	'w2': tf.Variable(tf.random_normal([hidden_l1, hidden_l2])),
#	'out': tf.Variable(tf.random_normal([hidden_l2, n_labels]))
#}

weights = {
	'w1': tf.get_variable('W1', shape=[n_input, hidden_l1],
	       initializer=tf.contrib.layers.variance_scaling_initializer()),
	'w2':  tf.get_variable('w2', shape=[hidden_l1, hidden_l2], initializer=tf.contrib.layers.variance_scaling_initializer()),
	'out': tf.get_variable('out', shape=[hidden_l2, n_labels], initializer=tf.contrib.layers.variance_scaling_initializer())
}
'''
weights = {
	'w1': tf.get_variable('W1', shape=[n_input, hidden_l1],
		       initializer=tf.contrib.layers.xavier_initializer()),
	'w2':  tf.get_variable('w2', shape=[hidden_l1, hidden_l2], initializer=tf.contrib.layers.xavier_initializer()),
	'out': tf.get_variable('out', shape=[hidden_l2, n_labels], initializer=tf.contrib.layers.xavier_initializer())
}
'''
biases = {
	'b1': tf.Variable(tf.random_normal([hidden_l1])),
	'b2': tf.Variable(tf.random_normal([hidden_l2])),
	'out': tf.Variable(tf.random_normal([n_labels]))
}

def neural_net(x):
	# Hidden fully connected layer with 256 neurons
	layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
	# Hidden fully connected layer with 256 neurons
	layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
	# Output fully connected layer with a neuron for each class
	layer_drop = tf.nn.dropout(layer_2, keep_prob)
	output_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return output_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.relu(logits)

loss_func_opt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_func_opt)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for step in range(1, n_steps+1):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		# Run optimization op (backprop)
		sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})
		if step % display_step == 0 or step == 1:
			# Calculate batch loss and accuracy
			#loss, acc = sess.run([loss_func_opt, accuracy], feed_dict={X: batch_x, Y: batch_y})
			#print("Step " + str(step) + ", Minibatch Loss= " + \
			#	"{:.4f}".format(loss) + ", Training Accuracy= " + \
			#	"{:.3f}".format(acc))
			print("Average Accuracy:", \
						sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1.0}))


	print("Optimization Done!")

	print("Test Accuracy:", \
			sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1.0}))

	img = np.invert(Image.open("test_img.png").convert('L')).ravel()
	prediction = sess.run(tf.argmax(logits, 1), feed_dict={X: [img]})
	print ("Prediction for test image:", np.squeeze(prediction))