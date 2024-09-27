# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 06:46:48 2024

@author: Realunknown
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import shapiro
from scipy import stats
from sklearn.decomposition import PCA

if __name__ == '__main__' :
    data_path = './features/'
    df = pd.read_csv(data_path + 'features.csv', encoding='utf-8')
    wave_data = df.iloc[:, 2:-1]
    wave_data = wave_data.T
    pca = PCA()
    pca.fit(wave_data)
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    #pc1 = [ele[0] for ele in pca.components_]
    #pc2 = [ele[1] for ele in pca.components_]
    print('特征向量\n',pca.components_)
    print('各个成分各自的方差百分比（贡献率）\n',pca.explained_variance_ratio_)
    plt.figure(figsize=(10,10))

    colour = ['#ff2121']
    plt.scatter(pc1,pc2 ,c=colour,edgecolors='#000000')
    plt.ylabel("Glucose",size=20)
    plt.xlabel('Age',size=20)
    plt.yticks(size=20)
    plt.xticks(size=20)
    plt.xlabel('PC1')
    plt.ylabel('PC2')