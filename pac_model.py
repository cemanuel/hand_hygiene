import tensorflow as tf
import numpy as np

IMG_HEIGHT = 240
IMG_WIDTH = 320
class HygeineDetectionModel(object):
	def __init__(self, sess, batch_size, stage_of_development, learning_rate, learning_rate_decay_factor):
		self.batch_size = batch_size
		self.batch_inputs = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 1], name="BATCH_IMAGE_INPUTS")
		self.batch_targets = tf.placeholder(tf.int32, shape=[None, 1], name="BATCH_LABELS")
		self.stage_of_development = stage_of_development

		self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
		self.learning_rate_decay_factor = learning_rate_decay_factor
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self.learning_rate_decay_factor, use_locking=False)
		self.global_step = tf.Variable(0, trainable=False)

		#[Batch Size X 240 X 320 X 1] -> [Batch Size X 240 X 320 X 4]
		conv1 = tf.layers.conv2d(inputs=self.batch_inputs,
									filters=4,
									kernel_size=[5, 5],
									padding="same",
									activation=tf.nn.relu)
		#[Batch Size X 240 X 320 X 4] -> [Batch Size X (240 - 2)/2 + 1 X (320 - 2)/2 + 1 X 4] -> [Batch Size X 120 X 160 X 4]
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
		#[Batch Size X 120 X 160 X 4] -> [Batch Size X 120 X 160 X 8]
		conv2 = tf.layers.conv2d(inputs=pool1,
									filters=8,
									kernel_size=[5, 5],
									padding="same",
									activation=tf.nn.relu)
		#[Batch Size X 120 X 160 X 8] -> [Batch Size X (120 - 2)/2 + 1 X (160 - 2)/2 + 1 X 8] -> [Batch Size X 60 X 80 X 8]
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
		#[Batch Size X 60 X 80 X 8] -> [Batch Size X 60 X 80 X 16]
		conv3 = tf.layers.conv2d(inputs=pool2,
									filters=16,
									kernel_size=[5, 5],
									padding="same",
									activation=tf.nn.relu)
		#[Batch Size X 60 X 80 X 16] -> [Batch Size X (60-2)/2 + 1 X (80-2)/2 + 1 X 16] -> [Batch Size X 30 X 40 X 16]
		pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
		#[Batch Size X 30 X 40 X 16] -> [Batch Size X 30 X 40 X 32]
		conv4 = tf.layers.conv2d(inputs=pool3,
									filters=32,
									kernel_size=[5, 5],
									padding="same",
									activation=tf.nn.relu)
		#[Batch Size X 30 X 40 X 32] -> [Batch Size X (30-2)/2 + 1 X (40-2)/2 + 1 X 32] -> [Batch Size X 15 X 20 X 32]
		pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
		pool4_flat = tf.reshape(pool4, [-1, 15 * 20 * 32])
		dense = tf.layers.dense(inputs=pool4_flat, units=9600, activation=tf.nn.relu)
		dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=self.stage_of_development=="training")

		logits = tf.layers.dense(inputs=dropout, units=2)
		self.predictions = {
			"classes": tf.argmax(input=logits, axis=1),
			"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
		}

		if self.stage_of_development == "training":
			onehot_labels = tf.one_hot(indices=tf.cast(tf.reshape(self.batch_targets, [-1]), tf.int32), depth=2)
			self.loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
			params_of_trainable_variables = tf.trainable_variables()
			opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
			gradient = tf.gradients(self.loss, params_of_trainable_variables)
			#clipped_gradient, norm = tf.clip_by_global_norm(gradient, max_gradient_norm)
			#self.gradient_norm = norm
			self.update_gradient = opt.apply_gradients(zip(gradient, params_of_trainable_variables), global_step=self.global_step)

		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)
    		
  # Add evaluation metrics (for EVAL mode)


	def step_with_dictionary(self, sess, batched_processed_images, batched_labels):
		input_feed = {}
		output_feed = None
		input_feed[self.batch_targets.name] = np.array(batched_labels).reshape(self.batch_size, 1).astype(np.float32)
		input_feed[self.batch_inputs.name] = np.array(batched_processed_images).reshape(self.batch_size, IMG_HEIGHT, IMG_WIDTH, 1).astype(np.float32)
		if self.stage_of_development == "training":
			output_feed = [self.update_gradient, self.loss]
		else:
			output_feed = [self.predictions['classes']]
		outputs = sess.run(output_feed, input_feed)
		if self.stage_of_development == "training":
			return outputs[1]
		else:
			return outputs[0]







