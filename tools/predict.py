# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import argparse
import os
import sys
sys.path.append('.')
from os import mkdir
from config import cfg
import torch
import numpy as np
from utils.logger import setup_logger
from modeling import MultiResnet
from torchvision import transforms
import zipfile

def main():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 0

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.RESULT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("resnet18", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = MultiResnet(cfg)

    model.load_state_dict(
        torch.load(cfg.TEST.WEIGHT, map_location=cfg.MODEL.DEVICE)['model_state_dict'])
    model.eval()

    test_path = '../tcdata/enso_round1_test_20210201/'

    ### 1. 测试数据读取
    files = os.listdir(test_path)

    mean = []
    std = []
    vals = np.zeros((len(files), 48, 24, 72))

    for i in range(len(files)):
        val = np.load(test_path + files[i])
        val1, val2, val3, val4 = np.split(val, 4, axis=3)
        val1.squeeze(), val2.squeeze(), val3.squeeze(), val4.squeeze()
        val = np.concatenate((val1, val2, val3, val4), axis=0, out=None)
        val = val.transpose(3, 0, 1, 2)
        vals[i, :, :, :] = val

    for i in range(48):
        mean.append(np.mean(vals[:, i, :, :]))
        std.append(np.std(vals[:, i, :, :]))

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])

    test_feas_dict = {}
    for file in files:
        test_feas_dict[file] = np.load(test_path + file)

    ### 2. 结果预测
    test_predicts_dict = {}
    for file_name, val in test_feas_dict.items():
        val1, val2, val3, val4 = np.split(val, 4, axis=3)
        val1.squeeze(), val2.squeeze(), val3.squeeze(), val4.squeeze()
        val = np.concatenate((val1, val2, val3, val4), axis=0, out=None)
        val = transform(val.transpose(3, 2, 1, 0).squeeze(0)).unsqueeze(0).float()
        test_predicts_dict[file_name] = model(val).reshape(-1, )
    #     test_predicts_dict[file_name] = model.predict(val.reshape([-1,12])[0,:])

    ### 3.存储预测结果
    for file_name, val in test_predicts_dict.items():
        np.save('../result/' + file_name, val.detach().numpy())

    make_zip()

# 打包目录为zip文件（未压缩）
def make_zip(source_dir='../result/', output_filename='../result.zip'):
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
            arcname = pathfile[pre_len:].strip(os.path.sep)  # 相对路径
            zipf.write(pathfile, arcname)
    zipf.close()

if __name__ == '__main__':
    main()