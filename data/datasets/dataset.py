# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    """
    Example dataset class for loading images from folder and converting them to monochromatic.
    Can be used to perform inference with trained MNIST model.
    'Dataset' type classes can also be used to create 'DataLoader' type classes which are used by datamodules.
    """

    def __init__(self, data_list, transforms):
        self.data_list = data_list
        self.transform = transforms
        self.len = len(self.data_list)

    def __getitem__(self, idx):
        index = idx % self.len
        img = self.data_list[index][0]
        label = self.data_list[index][1]
        return img, label

    def __len__(self):
        return self.len

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        data = self.transform(data)
        return data
