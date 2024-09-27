# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 23:12:40 2024

@author: Realunknown
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_box(df,names):
    plt.figure(figsize=(6,6),dpi = 600)
    data=df.loc[:, ['频率，Hz', '磁芯损耗，w/m3']]
    headers = ['Frequency/log(Hz)','Core Loss/log(w/m3)']
    data_log10 = data.applymap(np.log10)
    data_log10.columns = headers
    sns.boxplot(data_log10)
    plt.title('Polt Box for ' + names)
    #plt.title('Polt Box for Test Set')
    plt.show()
    return 0

if __name__ == '__main__' :
    data_path = './'
    file_names = ["trainning_material_1","trainning_material_2","trainning_material_3","trainning_material_4","test_q1"]
    #file_names = ["trainning_material_1"]
    for names in file_names:
        df = pd.read_csv(data_path + 'csv/' + file_names[0] + '.csv',encoding='utf-8')
        #缺失值
        missing = df.isnull()  
        print("The number of missing values is {}".format(len(df[missing.any(axis=1)])))
        #重复值
        duplicates = df.duplicated().any()
        if duplicates == False:
            print("There are no duplicate values here!")
        #异常值
        #Frequency and Core Loss
        plot_box(df,"Material " + names[-1])
        #Magnetic Flux Density
        df_mfd = df.iloc[:,4:]
        df_mfd_list = df_mfd.values.tolist()
        count = 0
        for i in range(len(df_mfd_list)):
            for j in range(len(df_mfd_list[0])):
                if abs(df_mfd_list[i][j]) > 0.3:
                    count += 1
        if count == 0:
            print("There are no outlier here!")