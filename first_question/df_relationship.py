# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:04:32 2024

@author: Realunknown
"""
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__' :
    data_path = "./features/"
    df = pd.read_csv(data_path + 'features.csv', encoding='utf-8')
    df_used = df.iloc[:, -3:]
    #df_used = df_used[:20]
    g = sns.PairGrid(df_used, hue='target')
    g = g.map_upper(sns.scatterplot)
    g = g.map_lower(sns.kdeplot, colors="C0")
    g = g.map_diag(sns.kdeplot, lw=2)#3绘制核密度图
    g = g.add_legend()#添加图例
    sns.set(style='whitegrid',font_scale=1.5)
    
    df_pure = df.iloc[:, 2:-1]
    colors = [(0,  "#9a5a65"),(0.25,  "#de9f8e"),(0.5,  "#baa9ad"),(0.75,  "#87adc1"),(1,  "#345e75")]
    colors = [(0,  "#2b4a5e"),(0.2,  "#4b6b82"),(0.4,  "#8ea2b7"),(0.6,  "#b0c0d1"),(0.8,  "#e2e4eb"),(1,  "#84a5d3")]


    cmap = LinearSegmentedColormap.from_list("fudan",colors,N = 256)

    

    plt.figure(figsize=(25, 25),dpi = 300)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)

    corrMatt = df_pure.corr()
    mask = np.array(corrMatt)
    mask[np.triu_indices_from(mask)] = False #下三角
    h = sns.heatmap(corrMatt, mask=mask,vmax=1,cbar=False, square=True,annot=True,annot_kws={"size": 20},fmt='.2f',cmap="viridis",cbar_kws={'ticks': [-1, 0, 1]})
    cb = h.figure.colorbar(h.collections[0]) #显示colorbar
    cb.ax.tick_params(labelsize=40)  # 设置colorbar刻度字体大小
    plt.xticks(rotation=75)
    