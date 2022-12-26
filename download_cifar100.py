import pickle
import tarfile
import os
import wget
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# define a fucntion to download cifar100 dataset
def download_cifar100():
    """
    Download CIFAR-100 dataset
    """
    if not os.path.exists('cifar-100-python.tar.gz'):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        wget.download(url, 'cifar-100-python.tar.gz')
        print('Downloaded')
    else:
        print('Already downloaded') 

    # extract the tar file
    if not os.path.exists('cifar-100-python'):
        tar = tarfile.open('cifar-100-python.tar.gz')
        tar.extractall()
        tar.close()
        print('Extracted')
    else:
        print('Already extracted')

def load_cifar100():
    """
    Load CIFAR-100 dataset
    """

    train_files = ['train']
    training_images = np.array([],dtype=np.uint8).reshape((0,3072))
    training_labels = np.array([])
    for tf in train_files:
        data_dict = unpickle('cifar-100-python/'+tf)
        data = data_dict[b'data']
        training_images = np.append(training_images,data,axis=0)
        training_labels = np.append(training_labels,data_dict[b'fine_labels'])
    print('Train data loaded!')
    print('Train data shape: ' + str(training_images.shape) + ' | Train labels shape: ' + str(training_labels.shape))

    test_images = np.array([],dtype=np.uint8).reshape((0,3072))
    test_labels = np.array([])
    data_dict = unpickle('cifar-100-python/test')
    data = data_dict[b'data']
    test_images = np.append(test_images,data,axis=0)
    test_labels = np.append(test_labels,data_dict[b'fine_labels'])
    print('Test data loaded!')
    print('Test data shape: ' + str(test_images.shape) + ' | Test labels shape: ' + str(test_labels.shape))
    return training_images, training_labels, test_images, test_labels


if __name__ == '__main__':
    download_cifar100()
    load_cifar100()