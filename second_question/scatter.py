# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:25:36 2024

@author: Realunknown
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == '__main__' :
    data_path = "./"
    df = pd.read_csv(data_path + "features_regress_same_material_wave_shape.csv")
    plt.figure(dpi = 600)
    sns.scatterplot(data= df,
                x="temp",
                y="core_loss",
                hue = "materials")
    plt.show()
    
    plt.figure(dpi = 600)
    sns.scatterplot(data= df,
                x="freq",
                y="core_loss",
                hue = "materials")
    plt.show()
    
    plt.figure(dpi = 600)
    sns.scatterplot(data= df,
                x="max_al",
                y="core_loss",
                hue = "materials")
    plt.show()