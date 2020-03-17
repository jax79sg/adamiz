from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import h5py
from PIL import Image
import numpy as np
from scipy.io import loadmat as load
import torch
import os
import urllib
import gzip
import pickle


def load_mnist_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset_train = datasets.MNIST(
        root='./dataset/',
        train=True,
        transform=transform,
        download=True
    )
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)

    dataset_test = datasets.MNIST(
        root='./dataset/',
        train=False,
        transform=transform,
        download=True
    )
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)
    return dataloader_train, dataloader_test

'''
class USPS_data_train(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        with h5py.File('./dataset/usps/usps.h5', 'r') as hf:
            train = hf.get('train')
            # format:(7291, 256)
            self.train_samples = train.get('data')[:]
            # format:(7291,)
            self.train_labels = train.get('target')[:]

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):
        img = self.train_samples[index]
        img = img.reshape(16, 16)
        img = img[:, :, np.newaxis]
        if self.transform is not None:
            img = self.transform(img)
        sample = [img, self.train_labels[index]]
        return sample


class USPS_data_Test(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        with h5py.File('./dataset/usps/usps.h5', 'r') as hf:
            test = hf.get('test')
            # format:(2007, 256)
            self.test_samples = test.get('data')[:]
            # format:(2007,)
            self.test_labels = test.get('target')[:]

    def __len__(self):
        return len(self.test_labels)

    def __getitem__(self, index):
        img = self.test_samples[index]
        img = img.reshape(16, 16)
        img = img[:, :, np.newaxis]
        if self.transform is not None:
            img = self.transform(img)
        sample = [img, self.test_labels[index]]
        return sample

'''
class USPS(Dataset):
    """USPS Dataset.

    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        #self.train_data *= 255.0
        self.train_data = self.train_data.transpose(
            (0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels

def load_usps_data(batch_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.Resize((28, 28), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    #dataset_train = USPS_data_train(transform=transform)
    dataset_train = USPS(root='./dataset/',
                        train=True,
                        transform=transform,
                        download=True)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    #dataset_test = USPS_data_Test(transform=transform)
    dataset_test = USPS(root='./dataset/',
                        train=False,
                        transform=transform,
                        download=True)

    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader_train, dataloader_test


#class SVHN_data_train(Dataset):
#    def __init__(self, transform=None):
#        self.transform = transform
#        traindata = load('./dataset/svhn/train_32x32.mat')
#        # format:(32, 32, 3, 73257)
#        self.train_samples = traindata['X']
#        # format:(73257, 1)
#        self.train_labels = traindata['y']

#    def __len__(self):
#        return len(self.train_labels)

#    def __getitem__(self, index):
#        img = self.train_samples[:, :, :, index]
#        if self.transform is not None:
#            img = self.transform(img)
#        sample = [img, self.train_labels[index, 0] % 10]
#        return sample

def SVHN_data(train,transform):
    """Get svhn dataset loader."""
    split = 'train' if train == True else 'test'

    # dataset and data loader
    svhn_dataset = datasets.SVHN(root='./dataset/',
                                   split=split,
                                   transform=transform,
                                   download=True)


    return svhn_dataset


class SVHN_data_test(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        testdata = load('./dataset/svhn/test_32x32.mat')
        # format:(32, 32, 3, 26032)
        self.test_samples = testdata['X']
        # format:(26032, 1)
        self.test_labels = testdata['y']

    def __len__(self):
        return len(self.test_labels)

    def __getitem__(self, index):
        img = self.test_samples[:, :, :, index]
        if self.transform is not None:
            img = self.transform(img)
        sample = [img, self.test_labels[index, 0] % 10]
        return sample


def load_svhn_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset_train = SVHN_data('train', transform=transform)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dataset_test = SVHN_data('test', transform=transform)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader_train, dataloader_test


def load_data(dataset, batch_size):
    if dataset == 'mnist':
        return load_mnist_data(batch_size)
    elif dataset == 'usps':
        return load_usps_data(batch_size)
    elif dataset == 'svhn':
        return load_svhn_data(batch_size)


if __name__ == '__main__':
    x, y = load_svhn_data(128)
    print(len(y.dataset))
    for i, sample in enumerate(y, 0):
        if i > 1:
            break
        print(i)
        print(sample[0].size())
        print(sample[1])
