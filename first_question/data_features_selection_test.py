# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:24:02 2024

@author: Realunknown
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def diagfunc(x, **kws):
  ax = plt.gca()
  ax.annotate(x.name, xy=(0.2, 0.5),size = 12, xycoords=ax.transAxes)

if __name__ == '__main__' :
    data_path = "./features/"
    df = pd.read_csv(data_path + 'features_values_test.csv', encoding='utf-8')
    headers_new = ['time_skew','time_kurt','time_form','time_peak','time_impulse','time_clearance','target']
    df_ready = df.iloc[:, 12:18]
    # 设置颜色主题
    antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864'] 

    # 绘制  Pairplot
    sns.pairplot(df_ready)