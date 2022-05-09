'''preprocess_data.py
Preprocessing data in STL-10 image dataset
YOUR NAMES HERE
CS343: Neural Networks
Project 2: Multilayer Perceptrons
'''
import numpy as np


def preprocess_stl(imgs, labels):
    '''Preprocesses stl image data for training by a MLP neural network

    Parameters:
    ----------
    imgs: unint8 ndarray  [0, 255]. shape=(Num imgs, height, width, RGB color chans)

    Returns:
    ----------
    imgs: float64 ndarray [0, 1]. shape=(Num imgs N,)
    Labels: int ndarray. shape=(Num imgs N,). Contains int-coded class values 0,1,...,9

    OLD TODO:
    1) Cast imgs to float64, normalize to the range [0,1]
    2) Flatten height, width, color chan dims. New shape will be (num imgs, height*width*chans)
    3) Compute the mean image across the dataset, subtract it from the dataset
    4) Fix class labeling. Should span 0, 1, ..., 9 NOT 1,2,...10
    '''

    imgs = imgs.astype('float64')

    imgs[:, :, :, 0]= (imgs[:, :, :, 0]-imgs[:,:,:,0].min())/imgs[:, :, :, 0].max()
    imgs[:, :, :, 1]= (imgs[:, :, :, 1]-imgs[:,:,:,1].min())/imgs[:, :, :, 1].max()
    imgs[:, :, :, 2]= (imgs[:, :, :, 2]-imgs[:,:,:,2].min())/imgs[:, :, :, 2].max()

    # imgs = imgs.reshape(imgs.shape[0], np.prod(imgs.shape[1:]))
    print("imgs shape before transpose", imgs.shape)
    imgs = imgs.transpose((0, 3, 1, 2))
    print("imgs shape after transpose", imgs.shape)
    mean_img = imgs.mean(axis=0)

    imgs = imgs - mean_img

    labels = labels-1

    return imgs, labels


def create_splits(data, y, n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500):
    '''Divides the dataset up into train/test/validation/development "splits" (disjoint partitions)
    Parameters:
    ----------
    data: float64 ndarray. Image data. shape=(Num imgs, height*width*chans)
    y: ndarray. int-coded labels.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)

    TODO:
    1) Divvy up the images into train/test/validation/development non-overlapping subsets (see return vars)
    '''

    if n_train_samps + n_test_samps + n_valid_samps + n_dev_samps != len(data):
        samps = n_train_samps + n_test_samps + n_valid_samps + n_dev_samps
        print(f'Error! Num samples {samps} does not equal num images {len(data)}!')
        return

    x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev = [], [], [], [], [], [], [], []

    j = 0

    while j < (n_train_samps):
        x_train.append(data[j])
        y_train.append(y[j])
        j += 1

    while j < (n_train_samps + n_test_samps):
        x_test.append(data[j])
        y_test.append(y[j])
        j += 1

    while j < (n_train_samps + n_test_samps + n_valid_samps):
        x_val.append(data[j])
        y_val.append(y[j])
        j += 1

    while j < (n_train_samps + n_test_samps + n_valid_samps + n_dev_samps):
        x_dev.append(data[j])
        y_dev.append(y[j])
        j += 1

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), np.array(x_val), np.array(y_val), np.array(x_dev), np.array(y_dev)



