'''accelerated_layer.py
Much faster Conv2, Maxpooling layers of a neural network
Oliver W. Layton
CS343: Neural Networks
Project 3: Convolutional Neural Networks
'''
import numpy as np

import im2col_cython
import layer
import filter_ops


class Conv2DAccel(layer.Conv2D):
    def compute_net_in(self):
        '''Computes fast convolution net-in using the im2col algorithm, C-compiled via Cython
        '''
        batch_sz, n_chans, img_y, img_x = self.input.shape
        n_kers, n_ker_chans, ker_x, ker_y = self.wts.shape
        ker_sz = ker_x

        stride = 1
        pad = int(np.ceil((ker_sz - 1) / 2))

        # padded input spatial dims
        img_y_p, img_x_p = img_y + 2*pad, img_x + 2*pad

        self.input_cols = im2col_cython.im2col_cython(self.input, ker_sz, ker_sz, pad, stride)

        self.net_in = self.wts.reshape(len(self.wts), -1) @ self.input_cols + self.b.reshape(-1, 1)
        self.net_in = self.net_in.reshape(n_kers, img_y, img_x, batch_sz)
        self.net_in = self.net_in.transpose(3, 0, 1, 2)

    def backward_netIn_to_prevLayer_netAct(self, d_upstream):
        '''Computes fast convolution backward pass using the im2col algorithm, C-compiled via Cython

        Parameters:
        -----------
        d_upstream: Upstream gradient

        Returns:
        -----------
        dprev_net_act: Gradient with respect to layer below
        d_wts: Wt gradient of current layer
        d_b: bias gradient of current layer
        '''
        batch_sz, n_chans, img_y, img_x = self.input.shape
        n_kers, n_ker_chans, ker_x, ker_y = self.wts.shape
        ker_sz = ker_x

        stride = 1
        pad = int(np.ceil((ker_sz - 1) / 2))

        # bias gradient
        d_b = np.sum(d_upstream, axis=(0, 2, 3))

        # wt gradient
        #
        # reshape upstream grad to be compatible with im2col format
        d_upstream = d_upstream.transpose(1, 2, 3, 0)
        d_upstream = d_upstream.reshape(n_kers, -1)

        d_wts = d_upstream @ self.input_cols.T
        d_wts = d_wts.reshape(self.wts.shape)

        # prev layer net_act grad
        #
        #
        dprev_net_act_cols = self.wts.reshape(n_kers, -1).T @ d_upstream
        dprev_net_act = im2col_cython.col2im_cython(dprev_net_act_cols, batch_sz, n_chans,
                                                    img_y, img_x, ker_x, ker_y,
                                                    pad, stride)
        return dprev_net_act, d_wts, d_b


class MaxPooling2DAccel(layer.MaxPooling2D):
    def compute_net_in(self):
        '''Computes fast 2D max pooling net-in using reshaping (partitioning input into small windows).
        '''
        mini_batch_sz, n_chans, img_y, img_x = self.input.shape

        out_x = filter_ops.get_pooling_out_shape(img_x, self.pool_size, self.strides)
        out_y = filter_ops.get_pooling_out_shape(img_y, self.pool_size, self.strides)

        # Partition the input into pool_sz chunks, then take the max within each chunk
        self.input_reshaped = self.input.reshape(mini_batch_sz, n_chans, out_y, self.pool_size,
                                                 out_x, self.pool_size)
        self.net_in = self.input_reshaped.max(axis=3).max(axis=4)

    def backward_netIn_to_prevLayer_netAct(self, d_upstream):
        '''Computes fast max pooling backward pass using the im2col algorithm, C-compiled via Cython
        Algorithm from Fei-Fei Li, Andrej Karpathy, & Justin Johnson (2015)

        Parameters:
        -----------
        d_upstream: Upstream gradient

        Returns:
        -----------
        dprev_net_act: Gradient with respect to layer below
        d_wts: None
        d_b: None
        '''
        # Find where net_in matches input (where the max vals are)
        maxes = self.input_reshaped == self.net_in[:, :, :, np.newaxis, :, np.newaxis]

        # Find matching values in upstream gradient and in destination gradient passed to prev layer
        d_upstream = d_upstream[:, :, :, np.newaxis, :, np.newaxis]
        dprev_net_act_rs = np.zeros_like(self.input_reshaped)
        d_upstream_broadcast, _ = np.broadcast_arrays(d_upstream, dprev_net_act_rs)
        dprev_net_act_rs[maxes] = d_upstream_broadcast[maxes]

        dprev_net_act_rs /= np.sum(maxes, axis=(3, 5), keepdims=True)
        dprev_net_act = dprev_net_act_rs.reshape(self.input.shape)

        return dprev_net_act, None, None
