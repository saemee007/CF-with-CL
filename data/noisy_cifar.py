from __future__ import print_function

import os
import os.path
import sys

import numpy as np
from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch
import torch.utils.data as data
from .utils import download_url, check_integrity, noisify_dataset, noisify_instance

class NoisyCIFAR10(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                  ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                  ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                  ['data_batch_4', '634d18415352ddfa80567beed471001a'],
                  ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],]

    test_list = [['test_batch', '40351d587109b95175f43aff81a1287e'],]
    
    def __init__(self, root, train=True, download=False,
                 transform=None, target_transform=None,
                 noise_type=None, noise_rate=0.2, closeset_ratio=0.0, openset_ratio=0.0,
                 random_state=0, verbose=True, synthetic=False):
        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.noise_type = noise_type
        self.noise_file = '../data/datasets/cifar10_human.pt'
        self.nb_classes = 10
        self.closeset_noise_rate = closeset_ratio
        self.openset_noise_ratio = openset_ratio
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        if self.train:
            # Load files
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()
            
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            
            if noise_type != 'clean':
                if noise_type in ['symmetric', 'pairflip']:
                    train_labels = np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                    
                    # Noise - Transition matrix
                    noisy_labels, self.actual_noise_rate = noisify_dataset(self.nb_classes, train_labels, noise_type, closeset_ratio, openset_ratio, random_state, verbose)
                    
                    self.train_noisy_labels = [i[0] for i in noisy_labels]
                    self.train_labels = [i[0] for i in train_labels]
                    
                elif noise_type == 'instance':
                    self.train_noisy_labels, self.actual_noise_rate = noisify_instance(self.train_data, self.train_labels, noise_rate=closeset_ratio)
                    print(f'Noise Type: {noise_type} (noise_ratio: {closeset_ratio})\n'
              f'Actual Total Noise Ratio: {self.actual_noise_rate:.3f}')
                
                elif noise_type == 'manual':  # manual noise
                    # load noise label
                    train_noisy_labels = self.load_label()
                    self.train_noisy_labels = train_noisy_labels.numpy().tolist()

                    noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
                    self.actual_noise_rate = np.sum(noise_or_not) / 50000
                    print(f'Noise Type: {noise_type} (noisy labels loaded from {self.noise_file})\n'
              f'Actual Total Noise Ratio: {self.actual_noise_rate:.3f}')
                    
                else:
                    raise NotImplementedError
                    
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
        
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type != 'clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'index': index, 'data': img, 'label': target}
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        
    def get_sets(self):
        assert self.train == True
        if self.noise_type == 'clean':
            return None, None, None
        closed_set, open_set, clean_set = [], [], []
        closeset_nb_classes = int(self.nb_classes * (1 - self.openset_noise_ratio))
        openset_label_list = [i for i in range(closeset_nb_classes, self.nb_classes)]

        for idx in range(self.train_data.shape[0]):
            if self.train_labels[idx] in openset_label_list:
                assert self.train_labels[idx] != self.train_noisy_labels[idx]
                open_set.append(idx)
            elif self.train_labels[idx] != self.train_noisy_labels[idx]:
                assert self.train_labels[idx] not in openset_label_list
                closed_set.append(idx)
            else:
                clean_set.append(idx)

        return closed_set, open_set, clean_set
    
    

    def load_label(self):
        '''
        I adopt .pt rather .pth according to this discussion:
        https://github.com/pytorch/pytorch/issues/14864
        '''
        # NOTE presently only use for load manual training label
        assert self.noise_file != 'None'
        noise_label = torch.load(self.noise_file)
        if isinstance(noise_label, dict):
            if "clean_label_train" in noise_label.keys():
                clean_label = noise_label['clean_label_train']
                assert torch.sum(torch.tensor(
                    self.train_labels) - clean_label) == 0  # commented for noise identification (NID) since we need to replace labels
            if "clean_label" in noise_label.keys() and 'raw_index' in noise_label.keys():
                assert torch.sum(
                    torch.tensor(self.train_labels)[noise_label['raw_index']] != noise_label['clean_label']) == 0
                noise_level = torch.sum(noise_label['clean_label'] == noise_label['noisy_label']) * 1.0 / (
                noise_label['clean_label'].shape[0])
                print(f'the overall noise level is {noise_level}')
                self.train_data = self.train_data[noise_label['raw_index']]
            return noise_label['noise_label_train'].view(-1).long() if 'noise_label_train' in noise_label.keys() else \
            noise_label['noisy_label'].view(-1).long()  # % 10

        else:
            return noise_label.view(-1).long()    

    
    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
    
    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    
class NoisyCIFAR100(data.Dataset):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [['train', '16019d7e3df5f24257cddd939b257f8d'],]

    test_list = [['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],]
    
    def __init__(self, root, train=True, download=False,
                 transform=None, target_transform=None,
                 noise_type=None, noise_rate=0.2, closeset_ratio=0.0, openset_ratio=0.0,
                 random_state=0, verbose=True, synthetic=False):
        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.noise_type = noise_type
        self.noise_file = '../data/datasets/cifar100_human.pt'
        self.nb_classes = 100
        self.closeset_noise_rate = closeset_ratio
        self.openset_noise_ratio = openset_ratio
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        if self.train:
            # Load files
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()
            
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            
            if noise_type != 'clean':
                if noise_type in ['symmetric', 'pairflip']:
                    train_labels = np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                    
                    # Noise - Transition matrix
                    noisy_labels, self.actual_noise_rate = noisify_dataset(self.nb_classes, train_labels, noise_type, closeset_ratio, openset_ratio, random_state, verbose)
                    
                    self.train_noisy_labels = [i[0] for i in noisy_labels]
                    self.train_labels = [i[0] for i in train_labels]
                    
                elif noise_type == 'instance':
                    self.train_noisy_labels, self.actual_noise_rate = noisify_instance(self.train_data, self.train_labels, noise_rate=closeset_ratio)
                    print(f'Noise Type: {noise_type} (noise_ratio: {closeset_ratio})\n'
              f'Actual Total Noise Ratio: {self.actual_noise_rate:.3f}')
                
                elif noise_type == 'manual':  # manual noise
                    # load noise label
                    train_noisy_labels = self.load_label()
                    self.train_noisy_labels = train_noisy_labels.numpy().tolist()

                    noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
                    self.actual_noise_rate = np.sum(noise_or_not) / 50000
                    print(f'Noise Type: {noise_type} (noisy labels loaded from {self.noise_file})\n'
              f'Actual Total Noise Ratio: {self.actual_noise_rate:.3f}')
                    
                else:
                    print(noise_type)
                    raise NotImplementedError
                    
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
        
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type != 'clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'index': index, 'data': img, 'label': target}
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        
    def get_sets(self):
        assert self.train == True
        if self.noise_type == 'clean':
            return None, None, None
        closed_set, open_set, clean_set = [], [], []
        closeset_nb_classes = int(self.nb_classes * (1 - self.openset_noise_ratio))
        openset_label_list = [i for i in range(closeset_nb_classes, self.nb_classes)]

        for idx in range(self.train_data.shape[0]):
            if self.train_labels[idx] in openset_label_list:
                assert self.train_labels[idx] != self.train_noisy_labels[idx]
                open_set.append(idx)
            elif self.train_labels[idx] != self.train_noisy_labels[idx]:
                assert self.train_labels[idx] not in openset_label_list
                closed_set.append(idx)
            else:
                clean_set.append(idx)

        return closed_set, open_set, clean_set

    def load_label(self):
        '''
        I adopt .pt rather .pth according to this discussion:
        https://github.com/pytorch/pytorch/issues/14864
        '''
        # NOTE presently only use for load manual training label
        assert self.noise_file != 'None'
        noise_label = torch.load(self.noise_file)
        if isinstance(noise_label, dict):
            if "clean_label_train" in noise_label.keys():
                clean_label = noise_label['clean_label_train']
                assert torch.sum(torch.tensor(
                    self.train_labels) - clean_label) == 0  # commented for noise identification (NID) since we need to replace labels
            if "clean_label" in noise_label.keys() and 'raw_index' in noise_label.keys():
                assert torch.sum(
                    torch.tensor(self.train_labels)[noise_label['raw_index']] != noise_label['clean_label']) == 0
                noise_level = torch.sum(noise_label['clean_label'] == noise_label['noisy_label']) * 1.0 / (
                noise_label['clean_label'].shape[0])
                print(f'the overall noise level is {noise_level}')
                self.train_data = self.train_data[noise_label['raw_index']]
            return noise_label['noise_label_train'].view(-1).long() if 'noise_label_train' in noise_label.keys() else \
            noise_label['noisy_label'].view(-1).long()  # % 10

        else:
            return noise_label.view(-1).long()    

    
    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
    
    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str