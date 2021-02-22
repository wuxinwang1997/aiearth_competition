# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import os
import numpy as np
import zipfile
import torch
from torch import nn
import torchvision.models as models

class SimpleResnet(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(48, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 24)
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):

        return self.model(x)

model = SimpleResnet()

model.load_state_dict(torch.load('../usr_data/model_data/soda/best-model.bin', map_location=torch.device('cpu'))['model_state_dict'])
model.eval()

test_path = '../tcdata/enso_round1_test_20210201/'

### 1. 测试数据读取
files = os.listdir(test_path)
test_feas_dict = {}
for file in files:
    test_feas_dict[file] = np.load(test_path + file)

### 2. 结果预测
test_predicts_dict = {}
for file_name, val in test_feas_dict.items():
    val1, val2, val3, val4 = np.split(val, 4, axis=3)
    val1.squeeze(), val2.squeeze(), val3.squeeze(), val4.squeeze()
    val = np.concatenate((val1, val2, val3, val4), axis=0, out=None)
    val = val.transpose(3, 0, 1, 2)
    test_predicts_dict[file_name] = model(torch.from_numpy(val).float()).reshape(-1, )
#     test_predicts_dict[file_name] = model.predict(val.reshape([-1,12])[0,:])

### 3.存储预测结果
for file_name, val in test_predicts_dict.items():
    np.save('../result/' + file_name, val.detach().numpy())

#打包目录为zip文件（未压缩）
def make_zip(source_dir='../result/', output_filename = '../result.zip'):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    source_dirs = os.walk(source_dir)
    print(source_dirs)
    for parent, dirnames, filenames in source_dirs:
        print(parent, dirnames)
        for filename in filenames:
            if '.npy' not in filename:
                continue
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)   #相对路径
            zipf.write(pathfile, arcname)
    zipf.close()
make_zip()