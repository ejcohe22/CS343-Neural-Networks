'''mlp.py
Constructs, trains, tests 3 layer multilayer layer perceptron networks
Sawyer Strong & Erik Cohen
CS343: Neural Networks
Project 2: Multilayer Perceptrons
'''
import numpy as np

np.random.seed(0)


class MLP():
    '''
    MLP is a class for multilayer networks with different loss functions

    The structure of our MLP will be:

    Input layer (X units)
    -> Hidden layer (Y units) with Rectified Linear activation (ReLu)
    -> Output layer (Z units) with softmax activation

    Due to the softmax, activation of output neuron i represents the probability that
    the current input sample belongs to class i.

    NOTE: We will keep our bias weights separate from our feature weights to simplify computations.
    '''
    def __init__(self, num_input_units, num_hidden_units, num_output_units):
        '''Constructor to build the model structure and intialize the weights. There are 3 layers:
        input layer, hidden layer, and output layer. Since the input layer represents each input
        sample, we don't learn weights for it.

        Parameters:
        -----------
        num_input_units: int. Num input features
        num_hidden_units: int. Num hidden units
        num_output_units: int. Num output units. Equal to # data classes.
        '''
        self.num_input_units = num_input_units
        self.num_hidden_units = num_hidden_units
        self.num_output_units = num_output_units

        self.initialize_wts(num_input_units, num_hidden_units, num_output_units) 


    def get_y_wts(self):
        '''Returns a copy of the hidden layer wts'''
        return self.y_wts.copy()


    def initialize_wts(self, M, H, C, std=0.1): 
        ''' Randomly initialize the hidden and output layer weights and bias term

        Parameters:
        -----------
        M: int. Num input features
        H: int. Num hidden units
        C: int. Num output units. Equal to # data classes.
        std: float. Standard deviation of the normal distribution of weights

        Returns:
        -----------
        No return

        TODO:
        - Initialize self.y_wts, self.y_b and self.z_wts, self.z_b
        with the appropriate size according to the normal distribution with standard deviation
        `std` and mean of 0.
          - For wt shapes, they should be be equal to (#prev layer units, #associated layer units)
            for example: self.y_wts has shape (M, H)
          - For bias shapes, they should equal the number of units in the associated layer.
            for example: self.y_b has shape (H,)
        '''
        # keep the random seed for debugging/test code purposes
        np.random.seed(0)

        self.y_wts = np.random.normal(0, std, (M, H))
        self.y_b = np.random.normal(0, std, (H,))

        self.z_wts = np.random.normal(0, std, (H, C))
        self.z_b = np.random.normal(0, std, (C,))

        return

    def accuracy(self, y, y_pred):
        ''' Computes the accuracy of classified samples. Proportion correct

        Parameters:
        -----------
        y: ndarray. int-coded true classes. shape=(Num samps,)
        y_pred: ndarray. int-coded predicted classes by the network. shape=(Num samps,)

        Returns:
        -----------
        float. accuracy in range [0, 1]
        '''
        y_diff = y - y_pred
        acc = np.count_nonzero(y_diff==0)/y.size

        return acc

    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
            e.g. if y = [0, 2, 1] and num_classes = 4 we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''
        y_one_hot = np.zeros((y.shape[0], num_classes))
        for i in range(y.shape[0]):
            y_one_hot[i, y[i]] = 1
        return y_one_hot

    def predict(self, features):
        ''' Predicts the int-coded class value for network inputs ('features').

        Parameters:
        -----------
        features: ndarray. shape=(mini-batch size, num features)

        Returns:
        -----------
        y_pred: ndarray. shape=(mini-batch size,).
            This is the int-coded predicted class values for the inputs passed in.
            Note: You can figure out the predicted class assignments without applying the
            softmax net activation function — it will not affect the most active neuron.
        '''
        num_classes = self.z_wts.shape[1]
        N = features.shape[0]

        y_net_in = features @ self.y_wts + self.y_b
        y_net_act = np.where(y_net_in < 0, 0, y_net_in)
        z_net_in = y_net_act @ self.z_wts + self.z_b

        return np.argmax(z_net_in, axis=1)

    def forward(self, features, y, reg=0):
        '''
        Performs a forward pass of the net (input -> hidden -> output).
        This should start with the features and progate the activity
        to the output layer, ending with the cross-entropy loss computation.
        Don't forget to add the regularization to the loss!

        NOTE: Implement all forward computations within this function
        (don't divide up into separate functions for net_in, net_act). Doing this all in one method
        is not good design, but as you will discover, having the
        forward computations (y_net_in, y_net_act, etc) easily accessible in one place makes the
        backward pass a lot easier to track during implementation. In future projects, we will
        rely on better OO design.

        NOTE: Loops of any kind are NOT ALLOWED!

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size, Num features)
        y: ndarray. int coded class labels. shape=(mini-batch-size,)
        reg: float. regularization strength.

        Returns:
        -----------
        y_net_in: ndarray. shape=(N, H). hidden layer "net in"
        y_net_act: ndarray. shape=(N, H). hidden layer activation
        z_net_in: ndarray. shape=(N, C). output layer "net in"
        z_net_act: ndarray. shape=(N, C). output layer activation
        loss: float. REGULARIZED loss derived from output layer, averaged over all input samples

        NOTE:
        - To regularize loss for multiple layers, you add the usual regularization to the loss
          from each set of weights (i.e. 2 in this case).
        '''
        num_classes = self.z_wts.shape[1]
        N = features.shape[0]
        # y_one_hot = self.one_hot(y, num_classes) 


        y_net_in = (features @ self.y_wts) + self.y_b

        y_net_act = np.where(y_net_in < 0, 0, y_net_in)

        z_net_in = (y_net_act @ self.z_wts) + self.z_b

        # implement z_net_act using softmax activation
        z_net_in_reduced = z_net_in - np.max(z_net_in, keepdims=True, axis=1)
        exp_sum = np.sum(np.exp(z_net_in_reduced), keepdims=True, axis=1)
        z_net_act = np.exp(z_net_in_reduced) / exp_sum
        
        # implement cross-entropy loss function
        log_act = np.log(z_net_act)
        correct_loss = log_act[np.arange(N), y.astype(np.integer)]
        loss = ((-1/N) * np.sum(correct_loss)) + (1/2) * reg * np.sum(np.square(self.y_wts)) + (1/2) * reg * np.sum(np.square(self.z_wts))

        return y_net_in, y_net_act, z_net_in, z_net_act, loss

    def backward(self, features, y, y_net_in, y_net_act, z_net_in, z_net_act, reg=0):
        '''
        Performs a backward pass (output -> hidden -> input) during training to update the
        weights. This function implements the backpropogation algorithm.

        This should start with the loss and progate the activity
        backwards through the net to the input-hidden weights.

        I added the first few gradients, assuming dz_net_act is your softmax
        activations. Next, tackle dz_net_in and so on.

        I suggest numbering your forward flow equations and process each for
        relevant gradients in reverse order until you hit the first set of weights.

        Don't forget to backpropogate the regularization to the weights!
        (I suggest worrying about this last)

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size, Num features)
        y: ndarray. int coded class labels. shape=(mini-batch-size,)
        y_net_in: ndarray. shape=(N, H). hidden layer "net in"
        y_net_act: ndarray. shape=(N, H). hidden layer activation
        z_net_in: ndarray. shape=(N, C). output layer "net in"
        z_net_act: ndarray. shape=(N, C). output layer activation
        reg: float. regularization strength.

        Returns:
        -----------
        dy_wts, dy_b, dz_wts, dz_b: The following backwards gradients
        (1) hidden wts, (2) hidden bias, (3) output weights, (4) output bias
        Shapes should match the respective wt/bias instance vars.

        NOTE:
        - Regularize each layer's weights like usual.
        '''
        # Loss -> z_net_act
        dz_net_act = -1/(len(z_net_act) * z_net_act)

        # z_net_act -> z_net_in
        y_one_hot = self.one_hot(y, self.num_output_units)
        dz_net_in = dz_net_act * z_net_act * (y_one_hot - z_net_act)

        # --------------------------------
        # TODO: Fill in gradients here

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

        return dy_wts, dy_b, dz_wts, dz_b

    def fit(self, features, y, x_validation, y_validation,
            resume_training=False, n_epochs=500, lr=0.0001, mini_batch_sz=256, reg=0, verbose=2):
        ''' Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Parameters:
        -----------
        features: ndarray. shape=(Num samples N, num features). Features over N inputs.
        y: ndarray. int-coded class assignments of training samples. 0,...,numClasses-1
        x_validation: ndarray. shape=(Num samples in validation set, num features).
            This is used for computing/printing the accuracy on the validation set at the end of each
            epoch.
        y_validation: ndarray. int-coded class assignments of validation samples. 0,...,numClasses-1
        resume_training: bool. True: we clear the network weights and do fresh training, False,
            we continue training based on the previous state of the network. This is handy if runs
            of training get interupted and you'd like to continue later.
        n_epochs: int. Number of training epochs
        lr: float. Learning rate
        mini_batch_sz: int. Batch size per epoch. i.e. How many samples we draw from features to pass
            through the model per training epoch before we do gradient descent and update the wts.
        reg: float. Regularization strength used when computing the loss and gradient.
        verbose: int. 0 means no print outs. Any value > 0 prints Current epoch number and
            training loss every 1000 epochs.

        Returns:
        -----------
        loss_history: Python list of floats. Recorded training loss on every epoch.
        train_acc_history: Python list of floats. Recorded accuracy on every training epoch.
        validation_acc_history: Python list of floats. Recorded accuracy on every epoch when
            tested on the validation set.

        TODO:
        -----------
        The flow of this method should follow the one that you wrote in single_layer_net.py.
        The main differences are:
        1) Remember to update weights and biases for all layers!
        2) At the end of an epoch (calculated from iterations), compute the trainng and
            validation set accuracy. This is only done at the end of an epoch because "peeking" slows
            down the training.
        '''
        num_samps, num_features = features.shape
        num_classes = len(np.unique(y))

        iter_per_epoch = max(int(num_samps / mini_batch_sz), 1)
        n_iter = n_epochs * iter_per_epoch

        loss_history = []
        train_acc_history = []
        validation_acc_history = []

        if not resume_training:
            self.initialize_wts(self.num_input_units, self.num_hidden_units, self.num_output_units)

        if verbose > 0:
            print(f'Starting to train network...There will be {n_epochs} epochs', end='')
            print(f' and {n_iter} iterations total, {iter_per_epoch} iter/epoch.')

        for i in range(n_iter):

            idx = np.random.randint(num_samps, size = mini_batch_sz)
            batch = features[idx, :]
            y_batch = y[idx]

            y_net_in, y_net_act, z_net_in, z_net_act, loss = self.forward(batch, y_batch)

            dy_wts, dy_b, dz_wts, dz_b = self.backward(batch, y_batch, y_net_in, y_net_act, z_net_in, z_net_act)

            self.y_wts = self.y_wts - dy_wts * lr
            self.y_b = self.y_b - dy_b * lr

            self.z_wts = self.z_wts - dz_wts * lr
            self.z_b = self.z_b - dz_b * lr


            if i % 100 == 0 and verbose > 0 and i > 0:
                print(f'  Completed iter {i}/{n_iter}. Training loss: {loss_history[-1]:.2f}.')

            if (i) % iter_per_epoch == 0:
                loss_history.append(loss)

                train_acc = self.accuracy(y, self.predict(features))
                val_acc = self.accuracy(y_validation, self.predict(x_validation))
                
                train_acc_history.append(train_acc)
                validation_acc_history.append(val_acc)
                
                if verbose > 0:
                    print(f"Completed epoch {i/iter_per_epoch}, train_acc: {train_acc}, validation_acc: {val_acc}")


        if verbose > 0:
            print('Finished training!')

        return loss_history, train_acc_history, validation_acc_history
