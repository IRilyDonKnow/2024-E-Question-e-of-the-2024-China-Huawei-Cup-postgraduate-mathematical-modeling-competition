# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 22:35:13 2024

@author: Realunknown
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def draw_MAE():
    fig, ax = plt.subplots(figsize=(5, 5),dpi = 300)
    names = dt["models"]
    x = range(len(names))
    
    ax.plot(x, dt["MAE"], color = 'darkred', marker = 'o', linestyle = '-',markersize=8, label = 'MAE(all features)')
    
    ax.set(xlabel='Model', ylabel='MAE of Core Loss (W/m^3)',
           title='Predicted MAE in Different Models')
    #plt.ylim(26,48)
    plt.legend(prop={'size': 10}) # 显示图例
    plt.xticks(x, names)
    ax.grid(True)

    #fig.savefig("test.png")
    plt.show()
    return 0

def draw_R_square_and_RMSE():
   fig, ax = plt.subplots(figsize=(5, 5),dpi = 300)
   names = dt["models"]
   x = range(len(names))
   
   ax.plot(x, dt["RMSE"], color = 'salmon', marker = 'o', linestyle = '-',markersize=8, label = 'RMSE(all features)')
   ax.plot(x, dt["R_square"], color = 'steelblue', marker = 'o', linestyle = '-',markersize=8, label = 'R-square(all features)')
   
   ax.set(xlabel='Model', ylabel='RMSE and R-square of Core Loss',
          title='Predicted RMSE and R-square in Different Models')
   #plt.ylim(32,60)
   plt.legend(prop={'size': 10}) # 显示图例
   plt.xticks(x, names)
   ax.grid(True)

   #fig.savefig("test.png")
   plt.show()
   return 0

if __name__ == '__main__' :
    #data_path = "D:/新建文件夹/经历：课题组事务/科研任务：机器学习/4. HEA原子特性回归/more_data/final_data/"
    data_path = "./"
    dt = pd.read_csv(data_path + "MAE_RMSE_R_square.csv")
    #dt_last_except = pd.read_csv(data_path + "last_3_except.csv")
    draw_MAE()
    draw_R_square_and_RMSE()