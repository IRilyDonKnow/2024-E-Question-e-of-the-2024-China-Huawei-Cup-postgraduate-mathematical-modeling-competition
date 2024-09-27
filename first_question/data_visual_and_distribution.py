# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 00:25:33 2024

@author: Realunknown
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import shapiro
from scipy import stats

def painting_waves(wave_data):
    data = [wd[4:] for wd in wave_data]
    data = data[:250]
    x = np.linspace(0, len(data[0]), len(data[0]))

    plt.rc('lines', linewidth=2.5)
    fig, ax = plt.subplots(dpi = 600)

    for i in range(len(data)):
        ax.plot(x, data[i], color = 'darkgreen',alpha = i/len(data),linewidth = 1) #, label = "Temp: " + str(wave_data.iloc[0]["温度，oC"]))

    ax.legend(handlelength=4)
    ax.axhline(y=0, color='gray', linestyle='--',linewidth = 1)
    
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    return 0

def painting_distribution(wave_data):
    wave_form = ['Sine','Triangular','Trapezoidal']
    data = [wd[4:] for wd in wave_data]
    data = data[:250]
    all_data = []
    for i in range(len(data)):
        all_data += data[i]
    plt.figure(figsize=(8, 8))
    sns.violinplot(data=all_data, palette="viridis", cut=1, linewidth=1.5, bw_adjust=.5,)
    # 添加标题和标签
    plt.title('Violin Plot for '+ wave_form[1], fontsize=20)
    plt.xlabel('Frequency', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)
    plt.yticks(fontsize = 20)
    plt.text(-0.4,0.2,"Mean: {:.3f}".format(np.mean(all_data)) + "\n" + "Std: {:.3f}".format(np.std(all_data)) ,bbox = {'facecolor': 'white', 'alpha':  1.0},fontsize = 16 )
    plt.grid()
    plt.show()
    
    
    plt.figure(figsize=(8, 8))
    sns.boxplot(data=all_data)
    plt.title('Box Plot for '+ wave_form[1], fontsize=20)
    plt.xlabel('Frequency', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)
    plt.yticks(fontsize = 20)
    plt.text(-0.4,0.2,"Mean: {:.3f}".format(np.mean(all_data)) + "\n" + "Std: {:.3f}".format(np.std(all_data)) ,bbox = {'facecolor': 'white', 'alpha':  1.0},fontsize = 16 )
    plt.grid()
    plt.show()
    return 0

def painting_normal(wave_data):
    data = [wd[4:] for wd in wave_data]
    data = data[:250]
    all_data = []
    for i in range(len(data)):
        all_data += data[i]
    
    #S-W正态性检验
    stat_sw, p_value_sw = shapiro(data)
    print("Shapiro-Wilk’s统计量：", stat_sw)
    print("P值：", p_value_sw)
    
    #K-S检验
    stat_ks, p_value_ks =stats.kstest(all_data, 'norm', (np.mean(all_data), np.std(all_data)))
    print("Kolmogorov-Smirnov’s统计量：", stat_ks)
    print("P值：", p_value_ks)
    
    #A-D检验
    stat_ad = stats.anderson(all_data, dist='norm')[0]
    print("Anderson-Darling’s统计量：", stat_ad)
    #print("P值：", p_value_ad)
    
    plt.figure(figsize=(8, 8),dpi=120)
    sns.set(style='dark')
    sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
    g = sns.distplot(all_data,
               hist=True,
               kde=True,#开启核密度曲线kernel density estimate (KDE)
               kde_kws={'linestyle':'--','linewidth':'1'},#设置外框线属性
               axlabel='Xlabel'#设置x轴标题
               )
    plt.xlabel('Amplitude', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.text(-0.275,6,"K-S’s Stat: {:.3f}".format(stat_ks) + "\n" + "K-S’s P-Value: {:.3f}".format(p_value_ks) + "\n" +"A-D’s Stat: {:.3f}".format(stat_ad),bbox = {'facecolor': 'white', 'alpha':  1.0},fontsize = 16 )
    plt.yticks(fontsize = 20)
    plt.xticks(fontsize = 20)
    return 0

if __name__ == '__main__' :
    data_path = './'
    count_list = []
    file_names = ["trainning_material_1","trainning_material_2","trainning_material_3","trainning_material_4"]
    #file_names = ["trainning_material_1"]
    wave_data = []
    for names in file_names:
        df = pd.read_csv(data_path + 'csv/' + names + '.csv',encoding='utf-8')
        for i in range(len(df)):
            #if df.loc[i]["励磁波形"] == "正弦波" and df.loc[i]["温度，oC"] == 50:
            if df.loc[i]["励磁波形"] == "三角波":
                #print(df.loc[i])
                wave_data.append(df.loc[i].tolist())
    painting_waves(wave_data)
    painting_distribution(wave_data)
    painting_normal(wave_data)