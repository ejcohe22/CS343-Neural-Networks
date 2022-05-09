'''network.py
Represents  a neural network (collection of layers)
YOUR NAMES HERE
CS343: Neural Networks
Project 3: Convolutional Neural Networks
'''
import time
import numpy as np

import layer
import optimizer
import filter_ops
import accelerated_layer

np.random.seed(0)


class Network():
    '''Represents a neural network with some number of layers of various types.
    To create a specific network, create a subclass (e.g. ConvNet4) then
    add layers to it. For this project, the focus will be on building the
    ConvNet4 network.
    '''
    def __init__(self, reg=0, verbose=True):
        '''This method is pre-filled for you (shouldn't require modification).
        '''
        # Python list of Layer object references that make up out network
        self.layers = []
        # Regularization strength
        self.reg = reg
        # Whether we want network-related debug/info print outs
        self.verbose = verbose

        # Python list of ints. These are the indices of layers in `self.layers`
        # that have network weights. This should be all types of layers
        # EXCEPT MaxPool2D
        self.wt_layer_inds = []

        # As in former projects, Python list of loss, training/validation
        # accuracy during training recorded at some frequency (e.g. every epoch)
        self.loss_history = []
        self.train_acc_history = []
        self.validation_acc_history = []

    def compile(self, optimizer_name, **kwargs):
        '''Tells each network layer how weights should be updated during backprop
        during training (e.g. stochastic gradient descent, adam, etc.)

        This method is pre-filled for you (shouldn't require modification).

        NOTE: This NEEDS to be called AFTER creating your ConvNet4 object,
        but BEFORE you call `fit()` to train the net (otherwise, how does your
        net know how to update the weights?).

        Parameters:
        -----------
        optimizer_name: string. Name of optimizer class to use to update wts.
            See optimizer::create_optimizer for specific ones supported.
        **kwargs: Any number of optional parameters that get passed to the
            optimizer of your choice. e.g. learning rate.
        '''
        # Only set an optimizer for each layer with weights
        for l in [self.layers[i] for i in self.wt_layer_inds]:
            l.compile(optimizer_name, **kwargs)

    def fit(self, x_train, y_train, x_validate, y_validate, mini_batch_sz=100, n_epochs=10, acc_freq=4, print_every=1):
        '''Trains the neural network on data

        Parameters:
        -----------
        x_train: ndarray. shape=(num training samples, n_chans, img_y, img_x).
            Training data.
        y_train: ndarray. shape=(num training samples,).
            Training data classes, int coded.
        x_validate: ndarray. shape=(num validation samples, n_chans, img_y, img_x).
            Every so often during training (see acc_freq param), we compute
            the accuracy of the network in classifying the validation set
            (out-of-training-set generalization). This is the data we use.
        y_validate: ndarray. shape=(num validation samples,).
            Validation data classes, int coded.
        mini_batch_sz: int. Mini-batch training size.
        n_epochs: int. Number of training epochs.
        acc_freq: int. How many training iterations (weight updates) we wait
            before checking accuracy on the training and validation sets?

        TODO: Complete this method's implementation.
        1. In the main training loop, randomly sample to get a mini-batch.
        2. Do forward pass through network using the mini-batch.
        3. Do backward pass through network using the mini-batch.
        4. Compute the loss on the mini-batch, add it to our loss history list
        5. Call each layer's update wt method.
        6. Use the Python time module to print out the runtime (in minutes) for iteration 0 only.
            Also printout the projected time for completing ALL training iterations.
            (For simplicity, you don't need to consider the time taken for computing
            train and validation accuracy).

        '''

        iter_per_epoch = max(int(len(x_train) / mini_batch_sz), 1)
        n_iter = n_epochs * iter_per_epoch

        print('Starting to train...')
        print(f'{n_iter} iterations. {iter_per_epoch} iter/epoch.')


        for i in range(n_iter):
            idx = np.random.randint(x_train.shape[0], size=mini_batch_sz)
            batch_features = x_train[idx, :]
            batch_labels = y_train[idx]

            loss = self.forward(batch_features, batch_labels)
            self.backward(batch_labels)

            self.loss_history.append(loss)
            for layer in self.layers:
                layer.update_weights()

            if i % print_every == 0:
                print("iteration: {:d} | loss: {:f}".format(i, loss)) 

            if (i+1) % acc_freq == 0:
                train_acc = self.accuracy(x_train, y_train, mini_batch_sz=mini_batch_sz)
                val_acc = self.accuracy(x_validate, y_validate, mini_batch_sz=mini_batch_sz)

                self.train_acc_history.append(train_acc)
                self.validation_acc_history.append(val_acc)
                print(f'  Train acc: {train_acc}, Val acc: {val_acc}')

        if self.verbose == True:
            print("Finished training!")

    def predict(self, inputs):
        '''Classifies novel inputs presented to the network using the current
        weights.

        Parameters:
        -----------
        inputs: ndarray. shape=shape=(num test samples, n_chans, img_y, img_x)
            This is test data.

        Returns:
        -----------
        pred_classes: ndarray. shape=shape=(num test samples)
            Predicted classes (int coded) derived from the network.

        TODO:
        1. Do forward pass through network with `inputs`.
        2. Return the predicted class for each input sample.

        Hints:
        -----------
        - The most active net_in values in the output layer gives us the predictions.
        (We don't need to check net_act).
        '''
        temp_in = inputs
        for l in self.layers:
            temp_in = l.forward(temp_in)

        return np.argmax(temp_in, axis=1)

    def accuracy(self, inputs, y, samp_sz=500, mini_batch_sz=15):
        '''Computes accuracy using current net on the inputs `inputs` with classes `y`.

        This method is pre-filled for you (shouldn't require modification).

        Parameters:
        -----------
        inputs: ndarray. shape=shape=(num samples, n_chans, img_y, img_x)
            We are testing the classification accuracy on these data.
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(N,) for mini-batch size N.
        samp_sz: int. If the number of samples is bigger than this number,
            we take a random sample from `inputs` of this size. We do this to
            keep performance of this method reasonable.
        mini_batch_sz: Because it might be tricky to hold all the training
            instances in memory at once, process and evaluate the accuracy of
            samples from `input` in mini-batches. We merge the accuracy scores
            across batches so the result is no different than processing all at
            once.
        '''
        n_samps = len(inputs)

        # Do we subsample the input?
        if n_samps > samp_sz:
            subsamp_inds = np.random.choice(n_samps, samp_sz)
            n_samps = samp_sz
            inputs = inputs[subsamp_inds]
            y = y[subsamp_inds]

        # How many mini-batches do we split the data into to test accuracy?
        n_batches = int(np.ceil(n_samps / mini_batch_sz))
        # Placeholder for our predicted class ints
        y_pred = np.zeros(len(inputs), dtype=np.int32)

        # Compute the accuracy through the `predict` method in batches.
        # Strategy is to use a 1D cursor `b` to extract a range of inputs to
        # process (a mini-batch)
        for b in range(n_batches):
            low = b*mini_batch_sz
            high = b*mini_batch_sz+mini_batch_sz
            # Tolerate out-of-bounds as we reach the end of the num samples
            if high > n_samps:
                high = n_samps

            # Get the network predicted classes and put them in the right place
            y_pred[low:high] = self.predict(inputs[low:high])

        # Accuracy is the average proportion that the prediction matchs the true
        # int class
        acc = np.mean(y_pred == y)

        return acc

    def forward(self, inputs, y):
        '''Do forward pass through whole network

        Parameters:
        -----------
        inputs: ndarray. Inputs coming into the input layer of the net. shape=(B, n_chans, img_y, img_x)
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(B,) for mini-batch size B.

        Returns:
        -----------
        loss: float. REGULARIZED loss.

        TODO:
        1. Call the forward method of each layer in the network.
            Make the output of the previous layer the input to the next.
        2. Compute and get the loss from the LAST network layer.
        2. Compute and get the weight regularization via `self.wt_reg_reduce()` (implement this next)
        4. Return the sum of the loss and the regularization term.
        '''
        temp_in = inputs
        for l in self.layers:
            temp_in = l.forward(temp_in)
        loss = self.layers[-1].cross_entropy(y)
        wt_reg = self.wt_reg_reduce()
        return loss + wt_reg


    def wt_reg_reduce(self):
        '''Computes the loss weight regularization for all network layers that have weights

        Returns:
        -----------
        wt_reg: float. Regularization for weights from all layers across the network.

        NOTE: You only can compute regularization for layers with wts!
        Layer indicies with weights are maintained in `self.wt_layer_inds`.
        The network regularization `wt_reg` is simply the sum of all the regularization terms
        for each individual layer.
        '''
        wt_reg = 0
        for i in self.wt_layer_inds:
            wt_reg += 0.5 * self.reg * np.sum(np.square(self.layers[i].wts))
        return wt_reg

    def backward(self, y):
        '''Initiates the backward pass through all the layers of the network.

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(B,) for mini-batch size B.

        Returns:
        -----------
        None

        TODO:
        1. Initialize d_upstream, d_wts, d_b to None.
        2. Loop through the network layers in REVERSE ORDER, calling the `Layer` backward method.
            Remember that the output of layer.backward() becomes the d_upstream to the next layer down.
            We don't care about d_wts, d_b in this method (computed/stored in Layer).
        '''
        d_upstream, d_wts, d_b = None, None, None
        for l in reversed(self.layers) :
            d_upstream, d_wts, d_b = l.backward(d_upstream, y)

    def save_model(self, filename):
        # TODO: implement layer.save_model
        assert filename[-4:] == ".txt", "file must be of .txt extension"

        save_string = "Neural Network Model\n"

        
        for layer in self.layers:
            save_string += str(layer.save_model())+'\t' # TODO: work in progress
        save_string += '\n'
        '''
        #Save Conv Layer
        save_string += (self.layers[0].wts.tostring())+'@'+(self.layers[0].b.tostring())+'\n'

        #Save Dense Layer 1
        save_string += (self.layers[2].wts.tostring())+'@'+(self.layers[2].b.tostring())+'\n'

        #Save Dense Layer 2
        save_string += (self.layers[3].wts.tostring())+'@'+(self.layers[3].b.tostring())+'\n'
        '''
        save_string += str(self.wt_layer_inds)+'\n'
        save_string += str(self.reg)+'\n'   
        save_string += str(self.verbose)+'\n'
        save_string += str(self.loss_history)+'\n'        
        save_string += str(self.train_acc_history)+'\n'
        save_string += str(self.validation_acc_history)+'\n'        

        with open(filename, 'w') as f:
            f.write(save_string)

    def load_model(self, filename):
        # TODO: read layer.save_model texts and make a new layer and put them in self.layers
        # work in progress
        assert filename[-4:] == ".txt", "file must be of .txt extension"

        with open(filename, 'r') as f:
            print("loading ", f.readline())

            self.layers = []

            layer_data = f.readline().split('\t')
            for i, layer_info in enumerate(layer_data[:-1]):
                layer_info = layer_info.split('@')
                if layer_info[0] == "Dense":
                    L = layer.Dense(number=i,name="Dense"+str(i),units=1,n_units_prev_layer=1,wt_scale=1,activation=layer_info[1],reg=int(layer_info[2]),verbose=eval(layer_info[3]))
                    L.wts = np.asarray(eval(layer_info[4]))
                    L.b = np.asarray(eval(layer_info[5]))
                elif layer_info[0] == "Conv2D":
                    L = accelerated_layer.Conv2DAccel(number=i,name="Conv"+str(i),n_kers=1,ker_sz=1,n_chans=1,wt_scale=1,activation=layer_info[1],reg=int(layer_info[2]),verbose=eval(layer_info[3]))
                    L.wts = np.asarray(eval(layer_info[4]))
                    L.b = np.asarray(eval(layer_info[5]))
                elif layer_info[0] == "MaxPooling2D":
                    L = accelerated_layer.MaxPooling2DAccel(number=i,name="Pool"+str(i),pool_size=int(layer_info[1]),strides=int(layer_info[2]),activation=layer_info[3],reg=int(layer_info[4]),verbose=eval(layer_info[5]))
                self.layers.append(L)
            '''

            layer0 = f.readline().split('@')
            self.layers[0].wts = np.fromstring(layer0[0])
            self.layers[0].b = np.fromstring(layer0[1])

            layer2 = f.readline().split('@')
            self.layers[2].wts = np.fromstring(layer2[0])
            self.layers[2].b = np.fromstring(layer2[1])

            layer3 = f.readline().split('@')
            self.layers[3].wts = np.fromstring(layer3[0])
            self.layers[3].b = np.fromstring(layer3[1])
            '''

            self.wt_layer_inds = eval(f.readline())
            self.reg = int(f.readline())
            self.verbose = eval(f.readline())
            self.loss_history = eval(f.readline())
            self.train_acc_history = eval(f.readline())
            self.validation_acc_history = eval(f.readline())

    def dream_img(self, img, rate_of_change=0.01, y=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], intensity_iterations=5):
        y = np.asarray([y])
        for i in range(intensity_iterations):
            loss = self.forward(img, y)

            d_upstream, d_wts, d_b = None, None, None
            for l in reversed(self.layers) :
                d_upstream, d_wts, d_b = l.backward(d_upstream, y)

            d_I = d_upstream
            img -= rate_of_change*d_I

        return img



class ConvNet4(Network):
    '''
    Makes a ConvNet4 network with the following layers: Conv2D -> MaxPooling2D -> Dense -> Dense

    1. Convolution (net-in), Relu (net-act).
    2. Max pool 2D (net-in), linear (net-act).
    3. Dense (net-in), Relu (net-act).
    4. Dense (net-in), soft-max (net-act).
    '''
    def __init__(self, input_shape=(3, 32, 32), n_kers=(32,), ker_sz=(7,), dense_interior_units=(100,),
                 pooling_sizes=(2,), pooling_strides=(2,), n_classes=10, wt_scale=1e-3, reg=0, verbose=True):
        '''
        Parameters:
        -----------
        input_shape: tuple. Shape of a SINGLE input sample (no mini-batch). By default: (n_chans, img_y, img_x)
        n_kers: tuple. Number of kernels/units in the 1st convolution layer. Format is (32,), which is a tuple
            rather than just an int. The reasoning is that if you wanted to create another Conv2D layer, say with 16
            units, n_kers would then be (32, 16). Thus, this format easily allows us to make the net deeper.
        ker_sz: tuple. x/y size of each convolution filter. Format is (7,), which means make 7x7 filters in the FIRST
            Conv2D layer. If we had another Conv2D layer with filters size 5x5, it would be ker_sz=(7,5)
        dense_interior_units: tuple. Number of hidden units in each dense layer. Same format as above.
            NOTE: Does NOT include the output layer, which has # units = # classes.
        pooling_sizes: tuple. Pooling extent in the i-th MaxPooling2D layer.  Same format as above.
        pooling_strides: tuple. Pooling stride in the i-th MaxPooling2D layer.  Same format as above.
        n_classes: int. Number of classes in the input. This will become the number of units in the Output Dense layer.
        wt_scale: float. Global weight scaling to use for all layers with weights
        reg: float. Regularization strength
        verbose: bool. Do we want to term network-related debug print outs on?
            NOTE: This is different than per-layer verbose settings, which are turned manually on below.

        TODO:
        1. Assemble the layers of the network and add them (in order) to `self.layers`.
        2. Remember to define self.wt_layer_inds as the list indicies in self.layers that have weights.
        '''
        super().__init__(reg, verbose)

        n_chans, h, w = input_shape

        # 1) Input convolutional layer

        C = layer.Conv2D(number=0,name="Conv",n_kers=n_kers[0],ker_sz=ker_sz[0],n_chans=n_chans,wt_scale=wt_scale,activation="relu",reg=reg,verbose=False)

        self.layers.append(C)

        # 2) 2x2 max pooling layer

        P = layer.MaxPooling2D(number=1,name="Pool",pool_size=pooling_sizes[0],strides=pooling_strides[0],activation="linear",reg=reg,verbose=False)

        self.layers.append(P)

        # 3) Dense layer

        pool_net_act_size_x = filter_ops.get_pooling_out_shape(w, pooling_sizes[0], pooling_strides[0])
        #print("pool_net_act_size_x: ",pool_net_act_size_x)
        pool_net_act_size_y = filter_ops.get_pooling_out_shape(h, pooling_sizes[0], pooling_strides[0])
        #print("pool_net_act_size_y: ",pool_net_act_size_y)
        #print("n_kers: ",n_kers[0])
        pool_net_act_size = pool_net_act_size_x * pool_net_act_size_x * n_kers[0]
        #print("pool_net_act_size: ",pool_net_act_size)

        D = layer.Dense(number=2,name="Dense",units=dense_interior_units[0],n_units_prev_layer=pool_net_act_size,wt_scale=wt_scale,activation="relu",reg=reg,verbose=False)

        self.layers.append(D)

        # 4) Dense softmax output layer

        O = layer.Dense(number=3,name="Output",units=n_classes,n_units_prev_layer=dense_interior_units[0],wt_scale=wt_scale,activation="softmax",reg=reg,verbose=False)

        self.layers.append(O)

        self.wt_layer_inds = [0,2,3]

class convNet4Accel(Network):
    '''
    Makes a ConvNet4 network with the following layers: Conv2D -> MaxPooling2D -> Dense -> Dense

    1. Convolution (net-in), Relu (net-act).
    2. Max pool 2D (net-in), linear (net-act).
    3. Dense (net-in), Relu (net-act).
    4. Dense (net-in), soft-max (net-act).
    '''
    def __init__(self, input_shape=(3, 32, 32), n_kers=(32,), ker_sz=(7,), dense_interior_units=(100,),
                 pooling_sizes=(2,), pooling_strides=(2,), n_classes=10, wt_scale=1e-3, reg=0, verbose=True):
        '''
        Parameters:
        -----------
        input_shape: tuple. Shape of a SINGLE input sample (no mini-batch). By default: (n_chans, img_y, img_x)
        n_kers: tuple. Number of kernels/units in the 1st convolution layer. Format is (32,), which is a tuple
            rather than just an int. The reasoning is that if you wanted to create another Conv2D layer, say with 16
            units, n_kers would then be (32, 16). Thus, this format easily allows us to make the net deeper.
        ker_sz: tuple. x/y size of each convolution filter. Format is (7,), which means make 7x7 filters in the FIRST
            Conv2D layer. If we had another Conv2D layer with filters size 5x5, it would be ker_sz=(7,5)
        dense_interior_units: tuple. Number of hidden units in each dense layer. Same format as above.
            NOTE: Does NOT include the output layer, which has # units = # classes.
        pooling_sizes: tuple. Pooling extent in the i-th MaxPooling2D layer.  Same format as above.
        pooling_strides: tuple. Pooling stride in the i-th MaxPooling2D layer.  Same format as above.
        n_classes: int. Number of classes in the input. This will become the number of units in the Output Dense layer.
        wt_scale: float. Global weight scaling to use for all layers with weights
        reg: float. Regularization strength
        verbose: bool. Do we want to term network-related debug print outs on?
            NOTE: This is different than per-layer verbose settings, which are turned manually on below.

        TODO:
        1. Assemble the layers of the network and add them (in order) to `self.layers`.
        2. Remember to define self.wt_layer_inds as the list indicies in self.layers that have weights.
        '''
        super().__init__(reg, verbose)

        n_chans, h, w = input_shape

        # 1) Input convolutional layer

        C = accelerated_layer.Conv2DAccel(number=0,name="Conv",n_kers=n_kers[0],ker_sz=ker_sz[0],n_chans=n_chans,wt_scale=wt_scale,activation="relu",reg=reg,verbose=False)

        self.layers.append(C)

        # 2) 2x2 max pooling layer

        P = accelerated_layer.MaxPooling2DAccel(number=1,name="Pool",pool_size=pooling_sizes[0],strides=pooling_strides[0],activation="linear",reg=reg,verbose=False)

        self.layers.append(P)

        # 3) Dense layer

        pool_net_act_size_x = filter_ops.get_pooling_out_shape(w, pooling_sizes[0], pooling_strides[0])
        #print("pool_net_act_size_x: ",pool_net_act_size_x)
        pool_net_act_size_y = filter_ops.get_pooling_out_shape(h, pooling_sizes[0], pooling_strides[0])
        #print("pool_net_act_size_y: ",pool_net_act_size_y)
        #print("n_kers: ",n_kers[0])
        pool_net_act_size = pool_net_act_size_x * pool_net_act_size_x * n_kers[0]
        #print("pool_net_act_size: ",pool_net_act_size)

        D = layer.Dense(number=2,name="Dense",units=dense_interior_units[0],n_units_prev_layer=pool_net_act_size,wt_scale=wt_scale,activation="relu",reg=reg,verbose=False)

        self.layers.append(D)

        # 4) Dense softmax output layer

        O = layer.Dense(number=3,name="Output",units=n_classes,n_units_prev_layer=dense_interior_units[0],wt_scale=wt_scale,activation="softmax",reg=reg,verbose=False)

        self.layers.append(O)

        self.wt_layer_inds = [0,2,3]

class convNet4AccelSmallKernels(Network):
    '''
    Makes a ConvNet4 network with the following layers: Conv2D -> MaxPooling2D -> Dense -> Dense

    1. Convolution (net-in), Relu (net-act).
    2. Max pool 2D (net-in), linear (net-act).
    3. Dense (net-in), Relu (net-act).
    4. Dense (net-in), soft-max (net-act).
    '''
    def __init__(self, input_shape=(3, 32, 32), n_kers=(64,), ker_sz=(3,), dense_interior_units=(100,),
                 pooling_sizes=(2,), pooling_strides=(2,), n_classes=10, wt_scale=1e-3, reg=0, verbose=True):
        '''
        Parameters:
        -----------
        input_shape: tuple. Shape of a SINGLE input sample (no mini-batch). By default: (n_chans, img_y, img_x)
        n_kers: tuple. Number of kernels/units in the 1st convolution layer. Format is (32,), which is a tuple
            rather than just an int. The reasoning is that if you wanted to create another Conv2D layer, say with 16
            units, n_kers would then be (32, 16). Thus, this format easily allows us to make the net deeper.
        ker_sz: tuple. x/y size of each convolution filter. Format is (7,), which means make 7x7 filters in the FIRST
            Conv2D layer. If we had another Conv2D layer with filters size 5x5, it would be ker_sz=(7,5)
        dense_interior_units: tuple. Number of hidden units in each dense layer. Same format as above.
            NOTE: Does NOT include the output layer, which has # units = # classes.
        pooling_sizes: tuple. Pooling extent in the i-th MaxPooling2D layer.  Same format as above.
        pooling_strides: tuple. Pooling stride in the i-th MaxPooling2D layer.  Same format as above.
        n_classes: int. Number of classes in the input. This will become the number of units in the Output Dense layer.
        wt_scale: float. Global weight scaling to use for all layers with weights
        reg: float. Regularization strength
        verbose: bool. Do we want to term network-related debug print outs on?
            NOTE: This is different than per-layer verbose settings, which are turned manually on below.

        TODO:
        1. Assemble the layers of the network and add them (in order) to `self.layers`.
        2. Remember to define self.wt_layer_inds as the list indicies in self.layers that have weights.
        '''
        super().__init__(reg, verbose)

        n_chans, h, w = input_shape

        # 1) Input convolutional layer

        C = accelerated_layer.Conv2DAccel(number=0,name="Conv",n_kers=n_kers[0],ker_sz=ker_sz[0],n_chans=n_chans,wt_scale=wt_scale,activation="relu",reg=reg,verbose=False)

        self.layers.append(C)

        # 2) 2x2 max pooling layer

        P = accelerated_layer.MaxPooling2DAccel(number=1,name="Pool",pool_size=pooling_sizes[0],strides=pooling_strides[0],activation="linear",reg=reg,verbose=False)

        self.layers.append(P)

        # 3) Dense layer

        pool_net_act_size_x = filter_ops.get_pooling_out_shape(w, pooling_sizes[0], pooling_strides[0])
        #print("pool_net_act_size_x: ",pool_net_act_size_x)
        pool_net_act_size_y = filter_ops.get_pooling_out_shape(h, pooling_sizes[0], pooling_strides[0])
        #print("pool_net_act_size_y: ",pool_net_act_size_y)
        #print("n_kers: ",n_kers[0])
        pool_net_act_size = pool_net_act_size_x * pool_net_act_size_x * n_kers[0]
        #print("pool_net_act_size: ",pool_net_act_size)

        D = layer.Dense(number=2,name="Dense",units=dense_interior_units[0],n_units_prev_layer=pool_net_act_size,wt_scale=wt_scale,activation="relu",reg=reg,verbose=False)

        self.layers.append(D)

        # 4) Dense softmax output layer

        O = layer.Dense(number=3,name="Output",units=n_classes,n_units_prev_layer=dense_interior_units[0],wt_scale=wt_scale,activation="softmax",reg=reg,verbose=False)

        self.layers.append(O)

        self.wt_layer_inds = [0,2,3]

class convNet4AccelBigKernels(Network):
    '''
    Makes a ConvNet4 network with the following layers: Conv2D -> MaxPooling2D -> Dense -> Dense

    1. Convolution (net-in), Relu (net-act).
    2. Max pool 2D (net-in), linear (net-act).
    3. Dense (net-in), Relu (net-act).
    4. Dense (net-in), soft-max (net-act).
    '''
    def __init__(self, input_shape=(3, 32, 32), n_kers=(32,), ker_sz=(9,), dense_interior_units=(100,),
                 pooling_sizes=(2,), pooling_strides=(2,), n_classes=10, wt_scale=1e-3, reg=0, verbose=True):
        '''
        Parameters:
        -----------
        input_shape: tuple. Shape of a SINGLE input sample (no mini-batch). By default: (n_chans, img_y, img_x)
        n_kers: tuple. Number of kernels/units in the 1st convolution layer. Format is (32,), which is a tuple
            rather than just an int. The reasoning is that if you wanted to create another Conv2D layer, say with 16
            units, n_kers would then be (32, 16). Thus, this format easily allows us to make the net deeper.
        ker_sz: tuple. x/y size of each convolution filter. Format is (7,), which means make 7x7 filters in the FIRST
            Conv2D layer. If we had another Conv2D layer with filters size 5x5, it would be ker_sz=(7,5)
        dense_interior_units: tuple. Number of hidden units in each dense layer. Same format as above.
            NOTE: Does NOT include the output layer, which has # units = # classes.
        pooling_sizes: tuple. Pooling extent in the i-th MaxPooling2D layer.  Same format as above.
        pooling_strides: tuple. Pooling stride in the i-th MaxPooling2D layer.  Same format as above.
        n_classes: int. Number of classes in the input. This will become the number of units in the Output Dense layer.
        wt_scale: float. Global weight scaling to use for all layers with weights
        reg: float. Regularization strength
        verbose: bool. Do we want to term network-related debug print outs on?
            NOTE: This is different than per-layer verbose settings, which are turned manually on below.

        TODO:
        1. Assemble the layers of the network and add them (in order) to `self.layers`.
        2. Remember to define self.wt_layer_inds as the list indicies in self.layers that have weights.
        '''
        super().__init__(reg, verbose)

        n_chans, h, w = input_shape

        # 1) Input convolutional layer

        C = accelerated_layer.Conv2DAccel(number=0,name="Conv",n_kers=n_kers[0],ker_sz=ker_sz[0],n_chans=n_chans,wt_scale=wt_scale,activation="relu",reg=reg,verbose=False)

        self.layers.append(C)

        # 2) 2x2 max pooling layer

        P = accelerated_layer.MaxPooling2DAccel(number=1,name="Pool",pool_size=pooling_sizes[0],strides=pooling_strides[0],activation="linear",reg=reg,verbose=False)

        self.layers.append(P)

        # 3) Dense layer

        pool_net_act_size_x = filter_ops.get_pooling_out_shape(w, pooling_sizes[0], pooling_strides[0])
        #print("pool_net_act_size_x: ",pool_net_act_size_x)
        pool_net_act_size_y = filter_ops.get_pooling_out_shape(h, pooling_sizes[0], pooling_strides[0])
        #print("pool_net_act_size_y: ",pool_net_act_size_y)
        #print("n_kers: ",n_kers[0])
        pool_net_act_size = pool_net_act_size_x * pool_net_act_size_x * n_kers[0]
        #print("pool_net_act_size: ",pool_net_act_size)

        D = layer.Dense(number=2,name="Dense",units=dense_interior_units[0],n_units_prev_layer=pool_net_act_size,wt_scale=wt_scale,activation="relu",reg=reg,verbose=False)

        self.layers.append(D)

        # 4) Dense softmax output layer

        O = layer.Dense(number=3,name="Output",units=n_classes,n_units_prev_layer=dense_interior_units[0],wt_scale=wt_scale,activation="softmax",reg=reg,verbose=False)

        self.layers.append(O)

        self.wt_layer_inds = [0,2,3]

