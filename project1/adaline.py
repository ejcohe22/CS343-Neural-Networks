'''adaline.py
Sawyer Strong & Erik Cohen
CS343: Neural Networks
Project 1: Single Layer Networks
ADALINE (ADaptive LInear NEuron) neural network for classification and regression
'''
import numpy as np


class Adaline():
    ''' Single-layer neural network

    Network weights are organized [bias, wt1, wt2, wt3, ..., wtM] for a net with M input neurons.
    '''
    def __init__(self, n_epochs=1000, lr=0.001):
        '''
        Parameters:
        ----------
        n_epochs: (int)
            Number of epochs to use for training the network
        lr: (float)
            Learning rate used in weight updates during training
        '''
        self.n_epochs = n_epochs
        self.lr = lr

        # Network weights: Bias is stored in self.wts[0], wt for neuron 1 is at self.wts[1],
        # wt for neuron 2 is at self.wts[2], ...
        self.wts = None
        # Record of training loss. Will be a list. Value at index i corresponds to loss on epoch i.
        self.loss_history = []
        # Record of training accuracy. Will be a list. Value at index i corresponds to acc. on epoch i.
        self.accuracy_history = []
        # Net Activation values
        self.net_act = None

    def get_wts(self):
        ''' Returns a copy of the network weight array'''
        return self.wts.copy()

    def get_num_epochs(self):
        ''' Returns the number of training epochs'''
        return self.n_epochs

    def get_learning_rate(self):
        ''' Returns the learning rate'''
        return self.lr

    def set_learning_rate(self, lr):
        '''Updates the value of self.lr (learning rate)'''
        self.lr = lr

    def net_input(self, features):
        ''' Computes the net_input (weighted sum of input features,  wts, bias)

        NOTE: bias is the 1st element of self.wts. Wts for input neurons 1, 2, 3, ..., M occupy
        the remaining positions.

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples, Num features]
            Collection of input vectors.

        Returns:
        ----------
        The net_input. Shape = [Num samples,]
        '''
        return (np.dot(features,self.wts[1:].T)+self.wts[0])

    def activation(self, net_in):
        '''
        Applies the activation function to the net input and returns the output neuron's activation.
        It is simply the identify function for vanilla ADALINE: f(x) = x

        Parameters:
        ----------
        net_in: ndarray. Shape = [Num samples,]

        Returns:
        ----------
        net_act. ndarray. Shape = [Num samples,]
        '''
        return net_in.copy()

    def compute_loss(self, y, net_act):
        ''' Computes the Sum of Squared Error (SSE) loss (over a single training epoch)

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples,]
            True classes corresponding to each input sample in a training epoch (coded as -1 or +1).
        net_act: ndarray. Shape = [Num samples,]
            Output neuron's activation value (after activation function is applied)

        Returns:
        ----------
        float. The SSE loss (across a single training epoch).
        '''
        squared_error = ((y - net_act)**2)
        sum_squared_error = np.sum(squared_error,axis=0)
        loss = sum_squared_error*0.5
        return loss

    def compute_accuracy(self, y, y_pred):
        ''' Computes accuracy (proportion correct) (across a single training epoch)

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples,]
            True classes corresponding to each input sample in a training epoch  (coded as -1 or +1).
        y_pred: ndarray. Shape = [Num samples,]
            Predicted classes corresponding to each input sample (coded as -1 or +1).

        Returns:
        ----------
        The accuracy for each input sample in the epoch. ndarray. Shape = [Num samples,]
            Expressed as proportions in [0.0, 1.0]
        '''
        y_diff = y - y_pred
        acc_array = np.where(y_diff==0,1,0)
        return np.sum(acc_array)/acc_array.shape[0]

    def gradient(self, errors, features):
        ''' Computes the error gradient of the loss function (for a single epoch).
        Used for backpropogation.

        Parameters:
        ----------
        errors: ndarray. Shape = [Num samples,]
            Difference between class and output neuron's activation value
        features: ndarray. Shape = [Num samples, Num features]
            Collection of input vectors.

        Returns:
        ----------
        grad_bias: float.
            Gradient with respect to the bias term
        grad_wts: ndarray. shape=(Num features,).
            Gradient with respect to the neuron weights in the input feature layer
        '''
        grad_bias = np.sum(errors,axis=0)
        grad_wts = np.dot(errors,features)
        return (grad_bias, grad_wts)

    def predict(self, features):
        '''Predicts the class of each test input sample

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples, Num features]
            Collection of input vectors.

        Returns:
        ----------
        The predicted classes (-1 or +1) for each input feature vector. Shape = [Num samples,]

        NOTE: Remember to apply the activation function!
        '''

        net_in = self.net_input(features)
        net_act = self.activation(net_in)
        self.net_act = net_act.copy()
        net_act[net_act >= 0] = 1
        net_act[net_act < 0] = -1
        return net_act

    def fit(self, features, y,early_stopping=False, loss_tol=0.1):
        ''' Trains the network on the input features for self.n_epochs number of epochs

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples, Num features]
            Collection of input vectors.
        y: ndarray. Shape = [Num samples,]
            Classees corresponding to each input sample (coded -1 or +1).

        Returns:
        ----------
        self.loss_history: Python list of network loss values for each epoch of training.
            Each loss value is the loss over a training epoch.
        self.acc_history: Python list of network accuracy values for each epoch of training
            Each accuracy value is the accuracy over a training epoch.

        TODO:
        1. Initialize the weights according to a Gaussian distribution centered
            at 0 with standard deviation of 0.01. Remember to initialize the bias in the same way.
        2. Write the main training loop where you:
            - Pass the inputs in each training epoch through the net.
            - Compute the error, loss, and accuracy (across the entire epoch).
            - Do backprop to update the weights and bias.
        '''
        self.wts = np.random.normal(0, 0.01, features.shape[1]+1)


        if (early_stopping):
            i = 0
            y_pred = self.predict(features)
            loss = self.compute_loss(y, self.net_act)
            acc = self.compute_accuracy(y, y_pred)
            self.loss_history.append(loss)
            self.accuracy_history.append(acc)

            errors = y - self.net_act
            grad_bias, grad_wts = self.gradient(errors, features)

            self.wts[0] += self.lr * grad_bias
            self.wts[1:] += self.lr * grad_wts
            i=i+1
            loss_diff = loss_tol+1
            while(loss_diff>loss_tol and i < self.n_epochs):
                y_pred = self.predict(features)
                loss = self.compute_loss(y, self.net_act)
                loss_diff = self.loss_history[-1]-loss
                acc = self.compute_accuracy(y, y_pred)
                self.loss_history.append(loss)
                self.accuracy_history.append(acc)

                errors = y - self.net_act
                grad_bias, grad_wts = self.gradient(errors, features)

                self.wts[0] += self.lr * grad_bias
                self.wts[1:] += self.lr * grad_wts
                i=i+1



        else:
            for i in range(self.get_num_epochs()):
                y_pred = self.predict(features)
                loss = self.compute_loss(y, self.net_act)
                acc = self.compute_accuracy(y, y_pred)
                self.loss_history.append(loss)
                self.accuracy_history.append(acc)

                errors = y - self.net_act
                grad_bias, grad_wts = self.gradient(errors, features)

                self.wts[0] += self.lr * grad_bias
                self.wts[1:] += self.lr * grad_wts

        return self.loss_history, self.accuracy_history
