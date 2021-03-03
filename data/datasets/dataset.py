# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, data_dict, transforms):
        self.data_dict = data_dict
        self.transform = transforms
        self.len = len(self.data_dict['sst'])

    def __getitem__(self, idx):
        idx = idx%self.len
        return  (self.data_dict['sst'][idx], self.data_dict['t300'][idx], self.data_dict['ua'][idx], self.data_dict['va'][idx]), self.data_dict['name'][idx]

    def __len__(self):
        return self.len

    def data_preproccess(self, data):
        for i in range(4):
            data[i] = self.transform(data[i])
        return data

class EarthDataset(Dataset):
    """
    Example dataset class for loading images from folder and converting them to monochromatic.
    Can be used to perform inference with trained MNIST model.
    'Dataset' type classes can also be used to create 'DataLoader' type classes which are used by datamodules.
    """

    def __init__(self, data_dict, transforms):
        self.data_dict = data_dict
        self.transform = transforms
        self.len = len(self.data_dict['sst'])

    def __getitem__(self, idx):
        idx = idx % self.len
        return  (self.data_dict['sst'][idx], self.data_dict['t300'][idx], self.data_dict['ua'][idx], self.data_dict['va'][idx]), self.data_dict['label'][idx]

    def __len__(self):
        return self.len

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        for i in range(4):
            data[i] = self.transform(data[i])
        return data
