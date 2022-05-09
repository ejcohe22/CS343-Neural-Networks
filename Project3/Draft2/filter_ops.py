'''filter_ops.py
Implements the convolution and max pooling operations.
Applied to images and other data represented as an ndarray.
Sawyer Strong & Erik Cohen
CS343: Neural Networks
Project 3: Convolutional neural networks
'''
import numpy as np
import math


def conv2_gray(img, kers, verbose=True):
	'''Does a 2D convolution operation on GRAYSCALE `img` using kernels `kers`.
	Uses 'same' boundary conditions.

	Parameters:
	-----------
	img: ndarray. Grayscale input image to be filtered. shape=(height img_y (px), width img_x (px))
	kers: ndarray. Convolution kernels. shape=(Num kers, ker_sz (px), ker_sz (px))
		NOTE: Kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
	verbose: bool. I suggest making helpful print statements showing the shape of various things
		as you go. Only execute these print statements if verbose is True.

	Returns:
	-----------
	filteredImg: ndarray. `img` filtered with all the `kers`. shape=(Num kers, img_y, img_x)

	Hints:
	-----------
	- Remember to flip your kernel since this is convolution!
	- Be careful of off-by-one errors, especially in setting up your loops. In particular, you
	want to align your convolution so that it starts aligned with the top-left corner of your
	padded image and iterate until the right/bottom sides of the kernel fall in the last pixel on
	the right/bottom sides of the padded image.
	- Use the 'same' padding formula for compute the necessary amount of padding to have the output
	image have the same spatial dimensions as the input.
	- I suggest using indexing/assignment to 'frame' your input image into the padded one.
	'''
	img_y, img_x = img.shape
	n_kers, ker_x, ker_y = kers.shape

	# Flip the Kernel
	kers_flipped = kers[:, ::-1, ::-1]

	# Add Zero Padding
	p = math.ceil((ker_x -1)/2)

	cols = np.zeros((img_y, p))
	padded_img = np.append(cols, img, 1)
	padded_img = np.append(padded_img, cols, 1)

	rows = np.zeros((p, padded_img.shape[1]))
	padded_img = np.append(rows, padded_img, 0)
	padded_img = np.append(padded_img, rows, 0)


	# Create Output Image
	filteredImg = np.zeros([n_kers, img_y, img_x])

	# Apply Conv
	for k in range(n_kers):
		for j in range(img_y):
			for i in range(img_x):
				filteredImg[k, j, i] = np.sum(np.multiply(padded_img[j:j+ker_y, i:i+ker_x], kers_flipped[k,:,:]))
	
	return filteredImg


def conv2(img, kers, verbose=True):
	'''Does a 2D convolution operation on COLOR or grayscale `img` using kernels `kers`.
	Uses 'same' boundary conditions.

	Parameters:
	-----------
	img: ndarray. Input image to be filtered. shape=(N_CHANS, height img_y (px), width img_x (px))
		where n_chans is 1 for grayscale images and 3 for RGB color images
	kers: ndarray. Convolution kernels. shape=(Num filters, ker_sz (px), ker_sz (px))
		NOTE: Each kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
	verbose: bool. I suggest making helpful print statements showing the shape of various things
		as you go. Only execute these print statements if verbose is True.

	Returns:
	-----------
	filteredImg: ndarray. `img` filtered with all `kers`. shape=(Num filters, N_CHANS, img_y, img_x)

	What's new:
	-----------
	- N_CHANS, see above.

	Hints:
	-----------
	- You should not need more for loops than you have in `conv2_gray`.
	- When summing inside your nested loops, keep in mind the keepdims=True parameter of np.sum and
	be aware of which axes you are summing over. If you use keepdims=True, you may want to remove
	singleton dimensions.
	'''
	n_chans, img_y, img_x = img.shape
	n_kers, ker_x, ker_y = kers.shape

	# Flip the Kernel
	kers_flipped = kers[:, ::-1, ::-1]

	# Add Zero Padding
	p = math.ceil((ker_x -1)/2)

	cols = np.zeros((n_chans, img_y, p))
	padded_img = np.append(cols, img, axis=2)
	padded_img = np.append(padded_img, cols, axis=2)

	rows = np.zeros((n_chans, p, padded_img.shape[2]))
	padded_img = np.append(rows, padded_img, axis=1)
	padded_img = np.append(padded_img, rows, axis=1)


	# Create Output Image
	filteredImg = np.zeros([n_kers, n_chans, img_y, img_x])

	# Apply Conv
	for k in range(n_kers):
		for j in range(img_y):
			for i in range(img_x):
				filteredImg[k, 0:n_chans, j, i] = np.sum(np.multiply(padded_img[0:n_chans, j:j+ker_y, i:i+ker_x], kers_flipped[k,:,:]), axis=(1,2))
	
	return filteredImg


def conv2nn(imgs, kers, bias, verbose=True):
	'''General 2D convolution operation suitable for a convolutional layer of a neural network.
	Uses 'same' boundary conditions.

	Parameters:
	-----------
	imgs: ndarray. Input IMAGES to be filtered. shape=(BATCH_SZ, n_chans, img_y, img_x)
		where batch_sz is the number of images in the mini-batch
		n_chans is 1 for grayscale images and 3 for RGB color images
	kers: ndarray. Convolution kernels. shape=(n_kers, N_CHANS, ker_sz, ker_sz)
		NOTE: Each kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
	bias: ndarray. Bias term used in the neural network layer. Shape=(n_kers,)
		i.e. there is a single bias per filter. Convolution by the c-th filter gets the c-th
		bias term added to it.
	verbose: bool. I suggest making helpful print statements showing the shape of various things
		as you go. Only execute these print statements if verbose is True.

	What's new (vs conv2):
	-----------
	- Multiple images (mini-batch support)
	- Kernels now have a color channel dimension too
	- Collapse (sum) over color channels when computing the returned output images
	- A bias term

	Returns:
	-----------
	output: ndarray. imgs filtered with all kers. shape=(BATCH_SZ, n_kers, img_y, img_x)

	Hints:
	-----------
	- You may need additional loop(s).
	- Summing inside your loop can be made simpler compared to conv2.
	- Adding the bias should be easy.
	'''
	batch_sz, n_chans, img_y, img_x = imgs.shape
	n_kers, n_ker_chans, ker_x, ker_y = kers.shape

	# Flip the Kernel
	kers_flipped = kers[:, :, ::-1, ::-1]

	# Add Zero Padding to all imgs in batch
	p = math.ceil((ker_x -1)/2)

	cols = np.zeros((batch_sz, n_chans, img_y, p))
	padded_img = np.append(cols, imgs, axis=3)
	padded_img = np.append(padded_img, cols, axis=3)

	rows = np.zeros((batch_sz, n_chans, p, padded_img.shape[3]))
	padded_img = np.append(rows, padded_img, axis=2)
	padded_img = np.append(padded_img, rows, axis=2)


	# Create Output Images
	filteredImg = np.zeros([batch_sz, n_kers, img_y, img_x])

	# Apply Conv
	for l in range(batch_sz): 
		for k in range(n_kers):
			for m in range(n_ker_chans):
				for j in range(img_y):
					for i in range(img_x):
						filteredImg[l, k, j, i] = np.sum(np.multiply(padded_img[l, :, j:j+ker_y, i:i+ker_x], kers_flipped[k,m,:,:]))
			filteredImg[l,k,:,:] = filteredImg[l,k,:,:] + bias[k].astype(np.int)

	return filteredImg


def get_pooling_out_shape(img_dim, pool_size, strides):
	'''Computes the size of the output of a max pooling operation along one spatial dimension.

	Parameters:
	-----------
	img_dim: int. Either img_y or img_x
	pool_size: int. Size of pooling window in one dimension: either x or y (assumed the same).
	strides: int. Size of stride when the max pooling window moves from one position to another.

	Returns:
	-----------
	int. The size in pixels of the output of the image after max pooling is applied, in the dimension
		img_dim.
	'''
	return (math.floor((img_dim-pool_size)/strides))+1


def max_pool(inputs, pool_size=2, strides=1, verbose=True):
	''' Does max pooling on inputs. Works on single grayscale images, so somewhat comparable to
	conv2_gray.

	Parameters:
	-----------
	inputs: Input to be filtered. shape=(height img_y, width img_x)
	pool_size: int. Pooling window extent in both x and y.
	strides: int. How many "pixels" in x and y to skip over between successive max pooling operations
	verbose: bool. I suggest making helpful print statements showing the shape of various things
		as you go. Only execute these print statements if verbose is True.

	Returns:
	-----------
	outputs: Input filtered with max pooling op. shape=(out_y, out_x)
		NOTE: out_y, out_x determined by the output shape formula. The input spatial dimensions are
		not preserved (unless pool_size=1...but what's the point of that? 🙂

	NOTE: There is no padding in the max-pooling operation.

	Hints:
	-----------
	- You should be able to heavily leverage the structure of your conv2_gray code here
	- Instead of defining a kernel, indexing strategically may be helpful
	- You may need to keep track of and update indices for both the input and output images
	- Overall, this should be a simpler implementation than conv2_gray
	'''
	img_y, img_x = inputs.shape

	# Compute the output shape
	out_x = get_pooling_out_shape(img_x, pool_size, strides)
	out_y = get_pooling_out_shape(img_y, pool_size, strides)

	if verbose: print(f"Max pooling: Img of shape ({img_y},{img_x}), with a pooling window of {pool_size} and stride of {strides} will result in an output of size ({out_y},{out_x})")

	outputs = np.ndarray((out_y,out_x))

	for i in range(out_y):
		img_i = i*strides
		for j in range(out_x):
			img_j = j*strides
			if verbose: print(f"Output at ({i},{j}) gets info from a max pool with top left at ({img_j},{img_i})")
			outputs[i,j] = np.max(inputs[img_i:img_i+pool_size,img_j:img_j+pool_size])
	return outputs


def max_poolnn(inputs, pool_size=2, strides=1, verbose=True):
	''' Max pooling implementation for a MaxPooling2D layer of a neural network

	Parameters:
	-----------
	inputs: Input to be filtered. shape=(mini_batch_sz, n_chans, height img_y, width img_x)
		where n_chans is 1 for grayscale images and 3 for RGB color images
	pool_size: int. Pooling window extent in both x and y.
	strides: int. How many "pixels" in x and y to skip over between successive max pooling operations
	verbose: bool. I suggest making helpful print statements showing the shape of various things
		as you go. Only execute these print statements if verbose is True.

	Returns:
	-----------
	outputs: Input filtered with max pooling op. shape=(mini_batch_sz, n_chans, out_y, out_x)
		NOTE: out_y, out_x determined by the output shape formula. The input spatial dimensions are
		not preserved (unless pool_size=1...but what's the point of that?)

	What's new (vs max_pool):
	-----------
	- Multiple images (mini-batch support)
	- Images now have a color channel dimension too

	Hints:
	-----------
	- If you added additional nested loops, be careful when you reset your input image indices
	'''
	mini_batch_sz, n_chans, img_y, img_x = inputs.shape

	# Compute the output shape
	out_x = get_pooling_out_shape(img_x, pool_size, strides)
	out_y = get_pooling_out_shape(img_y, pool_size, strides)

	if verbose:
		if verbose: print(f"Max pooling: Img of shape ({img_y},{img_x}), with a pooling window of {pool_size} and stride of {strides} will result in an output of size ({out_y},{out_x})")
		print(f"All {n_chans} channels will be preserved")

	outputs = np.ndarray((mini_batch_sz,n_chans,out_y,out_x))

	for n in range(mini_batch_sz):
		for c in range(n_chans):
			for i in range(out_y):
				img_i = i*strides
				for j in range(out_x):
					img_j = j*strides
					if verbose: print(f"Output at ({i},{j}) gets info from a max pool with top left at ({img_j},{img_i})")
					outputs[n,c,i,j] = np.max(inputs[n,c,img_i:img_i+pool_size,img_j:img_j+pool_size])
	return outputs
