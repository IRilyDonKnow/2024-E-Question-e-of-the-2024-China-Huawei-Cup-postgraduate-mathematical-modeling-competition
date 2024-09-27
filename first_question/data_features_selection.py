# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:53:30 2024

@author: Realunknown
"""

import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from collections import Counter
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.contingency_tables import Table

if __name__ == '__main__' :
    data_path = "./features/"
    #headers = ["index","shape_ex_point","shape_area","shape_de_mean","shape_de_var","shape_curve_smooth","shape_angle_max","shape_angle_min","time_mean","time_var","time_std","time_rms","time_skew","time_kurt","time_form","time_peak","time_impulse","time_clearance","freq_S1","freq_S2","freq_S3","freq_S4","target"]
    headers_new = ["shape_area","shape_de_mean","shape_de_var","shape_curve_smooth","shape_angle_max","shape_angle_min","time_mean","time_var","time_std","time_rms","time_skew","time_kurt","time_form","time_peak","time_impulse","time_clearance","freq_S1","freq_S2","freq_S3","freq_S4"]
    df = pd.read_csv(data_path + 'features.csv', encoding='utf-8')
    # 设置颜色主题
    antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864'] 

    # 绘制  Violinplot
    f, axes = plt.subplots(4, 5, figsize=(20, 20), sharex=True)
    sns.despine(left=True)

    for i in range(4):
        for j in range(5):
            sns.violinplot(x='target', y=headers_new[i*5+j] , data=df, palette=antV, ax=axes[(i, j)])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8)
    plt.show()
    
    plt.figure(figsize=(8,8),dpi = 600)
    sns.displot(data=df, x="time_kurt",palette=antV, kind="kde", hue="target")
    plt.show()
    