# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""
from .calculate_score import score

def evaluate(val_label, prediction):
    evaludate_score = score(y_true = val_label, y_preds = prediction)

    return evaludate_score