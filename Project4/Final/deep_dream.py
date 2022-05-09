'''deep_dream.py
Core functions used in the DeepDream algorithm. Implemented in TensorFlow
Sawyer Strong & Erik Cohen
CS 343: Neural Networks
Project 4: Transfer Learning
'''
import tensorflow as tf
import numpy as np


class DeepDream():
	def __init__(self, net, selected_layer_inds, all_layer_names, filter_inds=[]):
		'''
		Parameters:
		-----------
		net: TensorFlow Keras Model object. Network configured to return netAct values when
			presented an input image.
		selected_layer_inds: Python list of ints. Of all the layers in `net`, which are we taking netAct
			values from?
		all_layer_names: Python list of strings. Names of ALL non-input layers in `net` (not just selected).
		filter_inds: Python list of ints.
			Determines whether we take netAct values from particular neurons or across the entire layer.
			For usage, see note in forward() docstring.
		'''
		# Tf model
		self.net = net

		# Indices of layer of the net we are interested in focusing on netAct valuees
		self.selected_layer_inds = selected_layer_inds

		# Names of all non-input layers of net
		self.all_layer_names = all_layer_names

		# Indices of filters within each layer which we want to only take their netAct values
		self.filter_inds = filter_inds

	def forward(self, img_tf, verbose=False):
		'''Computes forward pass of `net` with the input `img_tf` to get the mean netAct values at the
		network layers that have indicies `selected_layer_inds`
		OR netAct values of specific filters that have indices `filter_inds`.

		Parameters:
		-----------
		img_tf: tf.Variable. shape=(img_y, img_x, n_chans). Input image
		verbose: Boolean. If true, print debug information.

		Returns:
		-----------
		netActs: Python list of floats, one per network layer. len(netActs) == len(selected_layer_inds)
			netAct values are averaged over each layer OR averaged over specific neurons within a layer
			with indicies `filter_inds`

		TODO:
		1) Add a leading singleton batch dimension to `img_tf`
		2) Pass the img_tf thru the `net` (treat `net` like it is a function), save the output, which are
		the netAct values at each layer.
		3) Average the netAct values for each selected layer of the net.
			Note: This will depend on whether filter_inds is empty or not
			(see NOTE below).
			Selecting non-contiguous indicies of a tf.Variable needs a special function.
			Check out https://www.tensorflow.org/api_docs/python/tf/gather.
			Note: Regardless of filter_inds, you are only adding one scalar/float value per layer.
			Also check out: https://www.tensorflow.org/api_docs/python/tf/math
		4) Print out the layer name, if verbose, when you are processing its mean netAct value.

		NOTE:
			If self.filter_inds empty, then this function should append the mean netAct value across ALL neurons
				within layer i to the output list.
			If self.filter_inds non-empty, then this function should append the mean netAct value across ALL spatial neurons
				within layer i that have indices given by filter_inds to the output netAct list.
				For example, if filter_inds = [55, 56], at each layer, we only take the mean netAct values
				across these two cells at every position in the image.
		'''
		img_tf = tf.expand_dims(img_tf, axis=0)

		netActs = self.net(img_tf)
		avg_netAct_per_layer = []
		
		for i, layer_netAct in enumerate(netActs):
			if i in self.selected_layer_inds:
				if verbose:
					print("Averaging netAct value of layer:", self.all_layer_names[i])
				if len(self.filter_inds) == 0:
					temp = layer_netAct
				else:
					# layer_netAct_flattened = tf.reshape(layer_netAct, [-1])
					temp = tf.gather(layer_netAct, self.filter_inds, axis = 3)

				avg_netAct_per_layer.append(tf.math.reduce_mean(temp))

		return avg_netAct_per_layer

	def image_gradient(self, img_tf, eps=1e-8, normalized=True, verbose=False):
		'''Computes the (normalized) gradients for each selected network layer with respect to
		the input image `img_tf`.

		Parameters:
		-----------
		img_tf: tf.Variable. shape=(img_y, img_x, n_chans). Input image
		eps: float. Small number used in normalization to prevent divide-by-zero errors
		normalized: boolean. Do we normalize each gradient within each layer by its standard deviation?
		verbose: Boolean. If true, print debug information.

		Returns:
		-----------
		grads: Python list of image gradients [shape=(img_y, img_x, n_chans)], one per selected
			network layer. len(list) = len(selected_layer_inds)

		TODO:
		1) Inside a persistent GradientTape block, have the tape "watch" the input image (there is a watch function)
			and do a forward pass thru the network with the input image.
			https://www.tensorflow.org/api_docs/python/tf/GradientTape
		2) Use the gradient tape to compute the gradient of each layer's netAct with respect to
			the input image.
		3) If we're normalizing, divide gradient by standard deviation (see notebook for equation).
			https://www.tensorflow.org/api_docs/python/tf/math
		'''
		with tf.GradientTape(persistent=True) as g:
			g.watch(img_tf)
			netActs = self.forward(img_tf)

		grads = []
		for layer_netAct in netActs:
			grad = g.gradient(layer_netAct, img_tf)
			if normalized:
				grad = grad/(tf.math.sqrt(tf.math.reduce_mean(tf.math.square(grad))) + eps)

			grads.append(grad)

		return grads

	def gradient_ascent(self, img_tf, n_iter=10, step_sz=0.01, clip_low=0, clip_high=1, verbose=False):
		'''Performs gradient ascent on the input img `img_tf`.
		The function computes the layer gradients respect to `img_tf`, then adds a fraction (`step_sz`) of
		each layer's gradient back to the input image. For efficiency, use `assign_add`.

		This process is performed a total of `n_iter` times.
		Finally, values in the modified image are clipped below `clip_low` and above `clip_high`.
		https://www.tensorflow.org/api_docs/python/tf/clip_by_value

		Parameters:
		-----------
		img_tf: tf.Variable. shape=(img_y, img_x, n_chans). Input image
		n_iter: int. Number of times to compute layer gradients and add them back to the image.
		step_sz: float. Proportion of each layer gradient to add to the image on each update.
		clip_low: int. Before returning, values below this value are replaced with this thresholded value
		clip_high: int. Before returning, values above this value are replaced with this thresholded value
		verbose: Boolean. If true, print debug information.

		Returns:
		-----------
		img_tf: tf.Variable. shape=(img_y, img_x, n_chans). Input image modified via gradient ascent.
		'''
		for i in range(n_iter):
			if verbose:
				print("Current iteration of gradient ascent:", i)
			grads = self.image_gradient(img_tf)
			print(grads)
			for j in range(len(self.selected_layer_inds)):
				img_tf = img_tf + step_sz * grads[j]

		img_tf = tf.clip_by_value(img_tf, clip_low, clip_high)
		return img_tf

	def gradient_ascent_multiscale(self, img_tf, n_scales=4, scale_factor=1.3, n_iter=10, step_sz=0.01,
								   clip_low=-1, clip_high=2, verbose=False):
		'''Multi-scale version of gradient ascent algorithm.
		Runs gradient ascent on the input image `img_tf`, `n_iter` times, with gradient step size `step_sz`.
		Then we multiplicatively enlarge the image by a factor of `scale_factor` and run gradient
		ascent again (if `n_scales` > 1). The resizing process happens a total of `n_scales` times.
		https://www.tensorflow.org/api_docs/python/tf/image/resize

		NOTE:
		- The output of the built-in resizing process is NOT a tf.Variable (its an ordinary tensor).
		But we need a tf.Variable to compute the image gradient during gradient ascent.
		So, wrap the resized image in a tf.Variable.

		Parameters:
		-----------
		img_tf: tf.Variable. shape=(img_y, img_x, n_chans). Input image
		n_scales: int. Number of times to resize the input image after running gradient ascent algorithm.
		scale_factor: float. Factor multiplied to current size of image to resize it.
		n_iter: int. Number of times to compute layer gradients and add them back to the image.
		step_sz: float. Proportion of each layer gradient to add to the image on each update.
		clip_low: int. Before returning, values below this value are replaced with this thresholded value
		clip_high: int. Before returning, values above this value are replaced with this thresholded value
		verbose: Boolean. If true, print debug information.

		Returns:
		-----------
		img_tf: tf.Variable. shape=(img_y, img_x, n_chans). Input image modified via gradient ascent.
		'''

		if n_scales == 1:
			return self.gradient_ascent(img_tf, n_iter, step_sz, verbose=verbose, clip_low=clip_low, clip_high=clip_high)

		for i in range(n_scales):
			if verbose:
				print("Scale number:", i)

			img_tf = self.gradient_ascent(img_tf, n_iter, step_sz, verbose=verbose, clip_low=clip_low, clip_high=clip_high)

			dims = img_tf.shape.as_list()
			scaled_img = tf.image.resize(img_tf, (int(dims[0]*scale_factor), int(dims[1]*scale_factor)))
			img_tf = tf.Variable(scaled_img)

		return img_tf

	def tf2array(self, tf_img):
		'''Converts a tf.Variable -> ndarray for plotting via matplotlib.

		Parameters:
		-----------
		img_tf: tf.Variable. shape=(img_y, img_x, n_chans). Deep Dream image

		Returns:
		-----------
		img: ndarray. shape=(img_y, img_x, n_chans). Input image modified for visualization.

		TODO:
		1) Clip values outside of [0, 1]
		2) Remove all singleton dimensions
		3) Convert to ndarray
		4) Normalize to [0, 1] based on the dynamic range of the image (the CS 251 way).
		'''
		tf_img = tf.clip_by_value(tf_img, 0, 1)
		tf_img = tf.squeeze(tf_img)
		np_img = tf_img.numpy()
		np_img = (np_img - np.min(np_img))/(np.max(np_img)-np.min(np_img))

		return np_img

class Nightmare(DeepDream):
	def __init__(self, net):
		super().__init__(net, [len(net.layers)-1], [layer.name for layer in net.layers[1:]], [0])


	def image_gradient(self, image, label, verbose=False):
	    
	    with tf.GradientTape() as tape:
	        tape.watch(image)
	        prediction = self.net(image)
	        loss = tf.keras.losses.MSE(label, prediction)
	    
	    gradient = tape.gradient(loss, image)
	    
	    signed_grad = tf.sign(gradient)
	    
	    return signed_grad

	def wrong(self, img, real_label, n_iter = 5, step_sz=0.01):
		for i in range(n_iter):
		    changes = self.image_gradient(img, real_label)
		    img = img + changes * step_sz

		return img

	def fake(self, img, fake_label, n_iter = 5, step_sz=0.01):
		for i in range(n_iter):
		    changes = self.image_gradient(img, fake_label)
		    img = img - changes * step_sz

		return img

	def minimal_wrong(self, img, real_label, step_sz=0.01):
		i = 0
		while np.asarray(real_label).argmax() == self.net(img).numpy().argmax():
		    changes = self.image_gradient(img, real_label).numpy()
		    img = img + changes * step_sz
		    i += 1
		    if i == 100:
		    	print("failed to fool image, exiting")
		    	return i, img

		return i, img

# 	def minimal_fake(self, img, fake_label, step_sz=0.01):
# 		i = 0
# 		while np.asarray(fake_label).argmax() != self.net(img).numpy().argmax():
# 		    changes = self.image_gradient(img, fake_label).numpy()
# 		    img = img - changes * step_sz
# 		    i += 1
# 		    if i == 100:
# 		    	print("failed to fake image, exiting")
# 		    	return img
# 
# 		return img


