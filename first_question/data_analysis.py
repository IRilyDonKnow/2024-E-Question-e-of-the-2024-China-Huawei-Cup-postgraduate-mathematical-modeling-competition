# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 23:12:40 2024

@author: Realunknown
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def painting_waveform(count_list):
    species = ("Material 1", "Material 2", "Material 3","Material 4")
    sine_wave = []
    Triangular_wave = []
    Trapezoidal_wave = []
    
    for count in count_list:
        sine_wave.append(count[0])
        Triangular_wave.append(count[1])
        Trapezoidal_wave.append(count[2])
        
    penguin_means = {
        'Sine': sine_wave,
        'Triangular':Triangular_wave ,
        'Trapezoidal': Trapezoidal_wave,
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10,8),layout='constrained',dpi = 600)

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Waveform Statistics in Each Material')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, 1800)
    plt.show()
    return 0

def painting_material(count_list):
    fruits = ['Material 1', 'Material 2', 'Material 3', 'Material 4']
    counts = [sum(count_list[0]), sum(count_list[1]), sum(count_list[2]),sum(count_list[3])]
    bar_labels = ['red', 'blue', '_red', 'orange']
    #bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

    fig, ax = plt.subplots(figsize = (10,8),dpi = 600)
    rects =ax.bar(fruits, counts, label=counts)
    ax.bar_label(rects, padding=3)

    ax.set_ylabel('Count')
    ax.set_title('Number of Wave Trains in Each Material')
    ax.set_ylim(0, 3800)
    #ax.legend(title='Fruit color')

    plt.show()
    return 0

if __name__ == '__main__' :
    data_path = './'
    count_list = []
    file_names = ["trainning_material_1","trainning_material_2","trainning_material_3","trainning_material_4"]
    #file_names = ["trainning_material_1"]
    for names in file_names:
        count = [0]*3
        df = pd.read_csv(data_path + 'csv/' + names + '.csv',encoding='utf-8')
        for i in range(len(df)):
            if df.loc[i]["励磁波形"] == "正弦波":
                count[0] += 1
            elif df.loc[i]["励磁波形"] == "三角波":
                count[1] += 1
            elif df.loc[i]["励磁波形"] == "梯形波":
                count[2] += 1
        count_list.append(count)
    painting_waveform(count_list)
    painting_material(count_list)