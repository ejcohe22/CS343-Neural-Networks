'''layer.py
Represents a layer of a neural network
Sawyer Strong & Erik Cohen
CS343: Neural Networks
Project 3: Convolutional Neural Networks
'''
import numpy as np
import filter_ops
import optimizer
np.set_printoptions(precision=7, suppress=True)
np.random.seed(0)


class Layer():
    def __init__(self, number, name, activation='linear', reg=0, verbose=True):
        '''Set variables that any network layer should have defined.
        '''
        self.number = number
        self.name = name
        self.verbose = verbose

        # Which activation function are we using for this layer?
        self.activation = activation

        # Regularization strength
        self.reg = reg

        # Network weights (aka convolution kernels) and bias coming into the current layer
        self.wts = None
        self.b = None

        # Keep track of raw input features passed into layer
        self.input = None

        # Forward pass computation placeholders
        self.net_in = None
        self.net_act = None
        self.b = None

        # Backward pass computation placeholders
        self.d_wts = None
        self.d_b = None

        # Per-layer weight and bias optimizers/updaters
        self.wt_optimizer = None
        self.b_optimizer = None

    def get_wts(self):
        '''Get a copy of this layer's weights.
        '''
        return self.wts.copy()

    def get_d_wts(self):
        '''Get a copy of this layer's weight gradient.
        '''
        return self.d_wts.copy()

    def set_wts(self, wts):
        '''Overwrite this layer's weights with `wts`. Use for weight updates during backprop.
        '''
        self.wts = wts

    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,C-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
            e.g. if y = [0, 2, 1] and num_classes (C) = 4 we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''
        N = y.shape[0]
        y_one_hot = np.zeros((N, num_classes))
        y_one_hot[np.arange(N), y] = 1

        return y_one_hot

    def linear(self):
        '''Linear activation function: f(x) = x.
        Note, this should work with ANY shape x.

        Sets the instance variable self.net_act

        Returns:
        -----------
        No return
        '''
        self.net_act = self.net_in

    def relu(self):
        '''Rectified linear activation function. f(x) is defined:
        x for x > 0
        0 for x <=0
        Note, this should work with ANY shape x.

        Sets the instance variable self.net_act

        Hint: This can be efficiently implemented with a single line of code.

        Returns:
        -----------
        No return
        '''
        #y_net_act = np.where(y_net_in < 0, 0, y_net_in)
        self.net_act = np.where(self.net_in < 0, 0, self.net_in)

    def softmax(self):
        '''Softmax activation function. See notebook for a refresher on the
        mathematical equation.

        "net in" shape: (B, C). B = mini-batch size. C = # classes in dataset.
        "net act". shape: (B, C).

        Sets the instance variable self.net_act

        Hint: Remember the numeric stability trick.

        Returns:
        -----------
        No return
        '''
        net_in_reduced = self.net_in - np.max(self.net_in, keepdims=True, axis=1)
        exp_sum = np.sum(np.exp(net_in_reduced), keepdims=True, axis=1)
        self.net_act = np.exp(net_in_reduced) / exp_sum

    def loss(self, y):
        '''Computes the loss for this layer. Only should be called on the output
        layer. We assume here that the output layer will have a softmax activation
        function, so we need to compute the loss according to the cross entropy.
        Feel free to generalize this for an extension.

        This method is pre-filled for you (shouldn't require modification).

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(B,) for mini-batch size B.

        Returns:
        -----------
        loss: float. Mean (cross-entropy) loss over the mini-batch.
        '''
        if self.activation == 'softmax':
            # compute cross-entropy loss
            return self.cross_entropy(y)

    def cross_entropy(self, y):
        '''Computes UNREGULARIZED cross-entropy loss.
        The network handles the regularization.
        See notebook for a refresher on the mathematical equation.

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(B,) for mini-batch size B.

        Returns:
        -----------
        loss: float. Mean loss over the mini-batch.
        '''

        B = y.shape[0]

        log_act = np.log(self.net_act)
        correct_loss = log_act[np.arange(B), y.astype(np.integer)]
        loss = ((-1/B) * np.sum(correct_loss))
        correct_acts = self.net_act(np.arange(self.net_act.shape[0],y))
        return loss

    def forward(self, inputs):
        '''Computes the forward pass through this particular layer.

        TODO:
        1. Save the input passed in the appropriate instance variable.
        2. Compute the net_in
        3. Compute the net_act
        4. Return a COPY (not a reference) of net_act.

        Parameters:
        -----------
        inputs: ndarray. Inputs coming into the current layer. shape=anything!

        Returns:
        -----------
        A COPY (not a reference) of net_act.
        '''
        self.input = inputs
        self.compute_net_in()
        self.compute_net_act()
        return np.copy(self.net_act)

    def backward(self, d_upstream, y):
        '''Do the backward pass through this layer.

        Parameters:
        -----------
        d_upstream: ndarray. `d_net_act` gradient for the current network layer, derived based on
            the gradient flows one level up.
            shape = shape of `self.net_act` for current layer.
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(B,) for mini-batch size B.

        Returns:
        -----------
        dprev_net_act: gradient of current layer's netIn function with respect to inputs coming from
            the previous layer (one level down).
            shape = shape of net_act one layer down
        d_wts: gradient with respect to current layer's wts. shape=shape of self.wts
        d_b: gradient with respect to current layer's bias. shape=shape of self.b

        TODO:
        1. Compute the gradient d_net_in via function call
        2. Compute the gradients dprev_net_act, d_wts, and d_b via function call
        3. Save the wt and bias gradients to the appropriate instance variables.
        '''
        # If the upstream gradient is undefined, this means that we are just starting the
        # backprop process and need to start with the gradient of the loss function (cross-entropy
        # with respect to the last layer's netAct function (softmax)
        if d_upstream is None:
            d_upstream = self.compute_dlast_net_act()
        self.d_net_in = self.backward_netAct_to_netIn(d_upstream,y)
        self.dprev_net_act, self.d_wts, self.d_b = self.backward_netIn_to_prevLayer_netAct(d_upstream)
        

    def compute_dlast_net_act(self):
        '''Computes the gradient of the loss function with respect to the last layer's netAct.
        If neurons in last layer are called z_k, this returns `dz_net_act`
        Used during backprop.
        This is the same as the 1st step of backward in the MLP project.

        This should not require modification.
        '''
        net_act_copy = self.net_act.copy()

        if self.activation == 'softmax':
            dlast_net_act = -1/(len(net_act_copy) * net_act_copy)
        else:
            raise RuntimeError('Output layer isnt softmax, so how to compute dlast_net_act is unspecified.')

        return dlast_net_act

    def compile(self, optimizer_name, **kwargs):
        '''Create optimizer objects for this layer which specifies the algorithm
        to use to do gradient descent. We have one for the wts, one for the bias.

        This method is pre-filled for you (shouldn't require modification).

        Parameters:
        -----------
        optimizer_name: string. optimizer name to use.
        **kwargs: keyword args that configures the optimizer (e.g.
        learning rate: `lr=0.001`)
        '''
        self.wt_optimizer = optimizer.Optimizer.create_optimizer(optimizer_name, **kwargs)
        self.b_optimizer = optimizer.Optimizer.create_optimizer(optimizer_name, **kwargs)

    def update_weights(self):
        '''Have the optimizer update the weights during training based on the wt/b gradient IN THIS PARTICULAR LAYER
        (backprop).

        We need the optimizer to update 1) the wts, 2) the bias.

        This method is pre-filled for you (shouldn't require modification).

        Returns:
        -----------
        None
        '''
        # We have no work to do if this is a layer without weights
        if self.wts is None:
            return

        if self.wt_optimizer is None:
            raise RuntimeError('Weight optimizer objects not defined. Call net.compile() before training.')

        if self.b_optimizer is None:
            raise RuntimeError('Bias optimizer objects not defined. Call net.compile() before training.')

        # Provide the optimizer with the current weights and its gradients
        self.wt_optimizer.prepare(self.wts, self.d_wts)
        # Do the weight update
        new_wts = self.wt_optimizer.update_weights()

        # Provide the optimizer with the current bias and its gradients
        self.b_optimizer.prepare(self.b, self.d_b)
        # Do the bias update
        new_b = self.b_optimizer.update_weights()

        self.wts = new_wts
        self.b = new_b

    def compute_net_in(self):
        '''Computes self.net_in. Always unique to layer type, so subclasses
        will override this (do not fill this in here).
        '''
        pass

    def compute_net_act(self):
        '''Call the appropriate activation function configured for this layer,
        indicated by the stored string `self.activation`.

        Throw an error if the activation function string is not one that you
        implemented.
        '''
        if self.activation == "linear":
            self.linear()
        elif self.activation == "softmax":
            self.softmax()
        elif self.activation == "relu":
            self.relu()
        else:
            raise Exception("Error: Invalid activation (String self.activation in layer.py unrecognized).")

    def backward_netAct_to_netIn(self, d_upstream, y):
        '''Calculates the gradient `d_net_in` for the current layer.
        Gradient calculation moves us THRU net_act TO net_in for the current layer during backprop.
        This should involve the derivative of the activation function for the current layer.

        Parameters:
        -----------
        d_upstream: ndarray. `net_act` Gradient for the current layer (d_net_act).
            This is computed during backprop from the layer above the current one.
            shape = shape of `self.net_act` in current layer.
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(B,) for mini-batch size B.

        Returns:
        -----------
        d_net_in: gradient that takes us from current layer's activation function to netIn.
            shape = (shape of self.net_in)

        TODO:
        1. Implement gradient for linear
        2. Implement gradient for relu
        2. Implement gradient for softmax

        '''
        if self.activation == 'relu':
            # TODO: compute correct gradient here
            d_net_in = d_upstream * np.where((self.net_act == 0),0,1)
        elif self.activation == 'linear':
            # TODO: compute correct gradient here
            d_net_in = d_upstream
        elif self.activation == 'softmax':
            # TODO: compute correct gradient here
            y_one_hot = self.one_hot(y)
            d_net_in = d_upstream * self.net_act * (y_one_hot - self.net_act)
        else:
            raise ValueError('Error! Unknown activation function ', self.activation)
        return d_net_in


class Dense(Layer):
    '''Dense (fully-connected) layer. Each units recieves (weighted) input from all units in the previous layer.
    These are the layers used in a multilayer perceptron.

    NOTE: Units are spatially arranged in 1D only - no 2D like Conv2D.
    '''
    def __init__(self, number, name, units, n_units_prev_layer,
                 wt_scale=1e-3, activation='linear', reg=0, verbose=True):
        super().__init__(number, name, activation=activation, reg=reg, verbose=verbose)
        '''
        Parameters:
        -----------
        number: int. Current layer number in the net. 0, ..., L-1,
            where L is the total number of layers.
        name: string. Human-readable string for identification/debugging.
            e.g. 'Conv2'
        units: int. Number of hidden units in the layer.
        n_units_prev_layer: int. Total number of units in the previous layer. If the previous layer is 2D,
            then this is the product of all units, collapsed into 1D. For example: if previous layer is MaxPool2D
            then there n_units_prev_layer = n_kers*n_chans*img_y*img_x
        wt_scale: float. Scales the magnitude of the random starting wts for each filter/kernel
        activation: string. Which activation function are we using?
        reg: Weight regularization strength
        verbose: Print debug info for this layer?

        TODO:
        1. Initialize this layer's weights (filters) to random values sampled
        from a standard normal distribution (mean 0, stdev 1). shape=(n_units_prev_layer, units)
        Remember to scale the result by `wt_scale`.
        2. Initialize this layer's bias terms to a 1d ndarray (same way as wts).
        Each unit in this layer has its own bias term.
        '''
        self.wts = np.random.normal(0,1,(n_units_prev_layer,units))
        self.wts = self.wts * wt_scale
        self.b = np.random.normal(0,1,(units,))
        self.b = self.b * wt_scale

    def compute_net_in(self):
        '''Computes `self.net_in` via Dense dot product of inputs (like in ADALINE/a MLP).

        NOTE: The input dimensions can be anything, but we DO know that the 1st dimension is
        always for mini-batch. Before computing netIn we want to collapse across all dimensions of
        the input tensor, EXCEPT for mini-batch. For example, we want
        self.net_in to have shape=(B, A), where A is the product of the OTHER input dimensions.
        and B is the mini-batch size.

        Hint: You did this in Project 0
        '''
        B = self.input.shape[0]        
        A = np.prod(self.input.shape[1:])
        self.net_in = np.dot(np.reshape(self.input,(B,A)),self.wts)+self.b

    def backward_netIn_to_prevLayer_netAct(self, d_upstream):
        '''Computes the `dprev_net_act`, `d_wts`, `d_b` gradients for a Dense layer.
        `dprev_net_act` is the gradient that gets us thru the current layer and TO the layer below.
        i.e. it will be the upstream gradient for the layer below the current one.

        Parameters:
        -----------
        d_upstream: Same shape as self.net_in (output of Dense backward_netAct_to_netIn()).
            shape=(mini_batch_sz, n_units)

        Returns:
        -----------
        dprev_net_act: gradient that gets us thru the current layer and TO the layer below.
            shape = (shape of self.input)
        d_wts: gradient of current layer's wts. shape=shape of self.wts = (n_prev_layer_units, units)
        d_b: gradient of current layer's bias. shape=(units,)

        NOTE:
        -----------
        Look back at your MLP project for inspiration.
            The rules/forms of equations are the same.
        Pay attention to shapes:
            You will need to do reshaping -before- computing one of the "d" variables,
            and -after- computing another of the "d" variables.
            Printing out shapes here when running the test code is super helpful.
            Shape errors will frequently show up at this backprop stage, one layer down.
        Regularize your wts
        '''
        dprev_net_act, d_wts, d_b = None

        '''
        ********************FROM MLP, FOR REFERENCE********************

        # Loss -> z_net_act
        dz_net_act = -1/(len(z_net_act) * z_net_act)

        # z_net_act -> z_net_in
        y_one_hot = self.one_hot(y, self.num_output_units)
        dz_net_in = dz_net_act * z_net_act * (y_one_hot - z_net_act)

        # z_net_in -> z_wts
        dz_wts = y_net_act.T @ dz_net_in + (reg * self.z_wts)

        # z_net_in -> z_b
        dz_b = np.sum(dz_net_in, axis=0) 

        # z_wts -> y_net_act
        dy_net_act = dz_net_in @ self.z_wts.T

        # y_net_act -> y_net_in
        dy_net_in = dy_net_act * np.where(y_net_act<=0, 0, 1)

        # y_net_in -> y_wts
        dy_wts = features.T @ dy_net_in + (reg * self.y_wts)

        # y_net_in -> y_b
        dy_b = np.sum(dy_net_in, axis=0)

        '''

        
        '''
        # net_in -> wts
        d_wts = y_net_act.T @ dz_net_in + (reg * self.z_wts)

        # net_in -> b
        d_b = np.sum(dz_net_in, axis=0) 

        # wts -> prev_net_act
        dprev_net_act = dz_net_in @ self.z_wts.T
        '''

        return (dprev_net_act, d_wts, d_b)
        


class Conv2D(Layer):
    '''Convolutational layer that does a 2D spatial convolution on input `images`.
    Each neuron in the layer has receptive field ('kernels' or 'filters') weights
    that are learned.
    '''
    def __init__(self, number, name, n_kers, ker_sz,
                 n_chans=3, wt_scale=0.01, activation='linear', reg=0, verbose=True):
        super().__init__(number, name, activation=activation, reg=reg, verbose=verbose)
        '''
        Parameters:
        -----------
        number: int. Current layer number in the net. 0, ..., L-1,
            where L is the total number of layers.
        name: string. Human-readable string for identification/debugging.
            e.g. 'Conv2'
        n_kers: int. Number of units/filters in the layer.
        n_chans: int. Number of color channels in the inputs
        wt_scale: float. Scales the magnitude of the random starting wts for each filter/kernel
        activation: string. Which activation function are we using?
        reg: Weight regularization strength
        verbose: Print debug info for this layer?

        TODO:
        1. Initialize this layer's weights (filters) to random values sampled
        from a standard normal distribution (mean 0, stdev 1). The filters should
        have shape=(n_kers, n_chans, ker_sz, ker_sz).
        2. Scale the magnitude of the wts by `wt_scale`.
        3. Initialize this layer's bias in the same way. shape=(n_kers,)
        '''
        std = 1
        self.wts = np.random.normal(0, std, (n_kers,n_chans,ker_sz,ker_sz))
        self.wts = self.wts * wt_scale
        self.b = np.random.normal(0, std, (n_kers,))
        self.b = self.b * wt_scale

    def compute_net_in(self):
        '''Computes `self.net_in` via convolution.
        Convolve the input tensor with the layer's learned convolution kernels.
        With a convolution layer, the learned filters are the same thing as the learned
        weights.

        Parameters:
        -----------
        All parameters needed for convolution are instance variables.
        All these values will be filled with valid values before this function is called during the
        forward pass through the network.

        Returns:
        -----------
        No return

        Hint:
        This should be an easy one-liner, you've done all the hard work last week :)
        '''
        self.net_in = filter_ops.conv2nn(self.input,self.wts,self.b,self.verbose)

    def backward_netIn_to_prevLayer_netAct(self, d_upstream):
        '''Computes backward `dprev_net_act`, `d_wts`, d_b` gradients that gets us
        THRU the conv layer to the wts/bias and the layer below (if it exists).

        This method is pre-filled for you (shouldn't require modification)
        (Phew! :)

        Parameters:
        -----------
        d_upstream: ndarray. Same shape as self.net_act (output of conv2 forward netAct).
            shape=(batch_sz, n_kers, img_y, img_x)

        Returns:
        -----------
        dprev_net_act. Input gradient. Same shape as self.input. shape=(batch_sz, n_chans, img_y, img_x)
        d_wts. Wt gradient. Same shape as self.wts. shape=(n_kers, n_chans, ker_sz, ker_sz)
        d_b. Bias gradient. Same shape as self.b. shape=(n_kers,)
        '''
        batch_sz, n_chans, img_y, img_x = self.input.shape
        n_kers, n_ker_chans, ker_x, ker_y = self.wts.shape

        if self.verbose:
            print(f'batch_sz={batch_sz}, n_chan={n_chans}, img_x={img_y}, img_y={img_x}')
            print(f'n_kers={n_kers}, n_ker_chans={n_ker_chans}, ker_x={ker_x}, ker_y={ker_y}')

        if ker_x != ker_y:
            print('Kernels must be square!')
            return

        if n_chans != n_ker_chans:
            print('Number of kernel channels doesnt match input num channels!')
            return

        ker_sz = ker_x

        # flip the kernels for convolution
        kers = self.wts[:, :, ::-1, ::-1]

        # Compute the padding for 'same' output
        p_x = int(np.ceil((ker_sz - 1) / 2))
        p_y = int(np.ceil((ker_sz - 1) / 2))

        # Pad the input images
        inputPadded = np.zeros([batch_sz, n_chans, img_y + 2*p_y, img_x + 2*p_x])
        img_y_p, img_x_p = inputPadded.shape[2:]

        # Embed the input image data into the padded images
        inputPadded[:, :, p_y:img_y+p_y, p_x:img_x+p_x] = self.input

        # batch_sz, n_chans, img_y, img_x
        dprev_net_act = np.zeros_like(inputPadded)
        # wts: n_kers, n_ker_chans, ker_x, ker_y
        d_wts = np.zeros_like(self.wts)

        if self.verbose:
            print(f'pad_x={p_y}, pad_y={p_x}')
            print(f'Padded shape is {inputPadded.shape}')

        for n in range(batch_sz):
            for k in range(n_kers):
                for y in range(img_y):
                    for x in range(img_x):
                        # When the filter is aligned with input position (x, y), accumulate
                        # the filter's weights – all globally weighted by the upstream gradient
                        # at the current (x, y) position.
                        dprev_net_act[n, :, y:y+ker_sz, x:x+ker_sz] += d_upstream[n, k, y, x] * kers[k]

                        # Filter wts updated based on upstream change times inputs present in the filter
                        # (this amount of input made the output change this much)
                        d_wts[k] += d_upstream[n, k, y, x] * inputPadded[n, :, y:y+ker_sz, x:x+ker_sz]

        d_b = np.sum(d_upstream, axis=(0, 2, 3))

        # regularize the weight gradient
        d_wts += self.reg*self.wts

        return dprev_net_act, d_wts, d_b


class MaxPooling2D(Layer):
    '''Max pooling layer. 2D because we pool over the spatial dimensions of the
    prior layer.

    NOTE: This layer contains NO WEIGHTS (self.wts = None). There is an activation
    function, however.
    '''
    def __init__(self, number, name,
                 pool_size=2, strides=1, activation='linear', reg=0, verbose=True):
        super().__init__(number, name, activation=activation, reg=reg, verbose=verbose)
        '''
        This method is pre-filled for you (shouldn't require modification).

        Parameters:
        -----------
        number: int. Current layer number in the net. 0, ..., L-1,
            where L is the total number of layers.
        name: string. Human-readable string for identification/debugging.
            e.g. 'MaxPoolLayer'
        pool_size: int. x/y pooling filter extent (a square).
        strides: int. How many "spaces" to slide over in either x and y
            between max pooling operations. Affects output spatial extent.
        activation: string. Which activation function are we using?
        reg: Weight regularization strength
        verbose: Print debug info for this layer?
        '''
        self.pool_size = pool_size
        self.strides = strides

    def compute_net_in(self):
        '''Computes `self.net_in` via max pooling.

        Parameters:
        -----------
        All parameters needed for convolution are instance variables.
        All these values will be filled with valid values before this function is called during the
        forward pass through the network.

        Returns:
        -----------
        No return

        Hint:
        This should be an easy one-liner, you've done all the hard work last week :)
        '''
        self.net_in = filter_ops.max_poolnn(self.input,self.pool_size,self.strides,self.verbose)

    def backward_netIn_to_prevLayer_netAct(self, d_upstream):
        '''Computes the dprev_net_act gradient, getting us thru the MaxPool2D layer to the layer
        below. No `d_wts` nor `d_b` because there are no weights in a MaxPool2D layer!

        Parameters:
        -----------
        d_upstream: ndarray. Same shape as self.net_act (output of max_pool forward).
            shape=(mini_batch_sz, n_chans, out_y, out_x)

        Returns:
        -----------
        dprev_net_act: dprev_net_act gradient, the upstream gradient for the layer below the current one.
            shape = (shape of self.input)

        Hints:
        -----------
        Here is the general strategy for this method:
            - Maintain indices for the upstream (x,y) (d_upstream) position and another set for the self.input indices.
                Remember input_xy >= d_upstream_xy due to the pooling.
                Remember to update the input indicies using the strides, d_upstream indices increment by 1.
            - Find within each sliding pooling window position the INDEX of the most active input
                (when the network was running in forward mode).
            - Within the current window, we update the value of dprev_net_act, but only at the
                max index from the forward pass (i.e. the position of the input value that got thru during the forward pass).
                We -accumulate- the current d_upstream value here (not the input value).
            - If you use argmax, it results a LINEAR index, but it might be useful to have the corresponding SUBSCRIPT
                index. For example, if the 2x2 window is [[0, 1], [2, 0]], argmax will return 3, but we want (x, y)=(0, 1).
                Check out the provided helper function: ind2sub(self, ind, sz) to convert linear indices to subscripts.
        '''
        mini_batch_sz, n_chans, img_y, img_x = self.input.shape
        mini_batch_sz_d, n_chans_d, out_y, out_x = d_upstream.shape

        if mini_batch_sz != mini_batch_sz_d:
            print(f'mini-batches do not match! {mini_batch_sz} != {mini_batch_sz_d}')
            exit()

        if n_chans != n_chans_d:
            print(f'n_chans do not match! {n_chans} != {n_chans_d}')
            exit()
        pass

    def ind2sub(self, linear_ind, sz):
        '''Converts a linear index to a subscript index based on the window size sz
        '''
        return np.unravel_index(linear_ind, sz)
