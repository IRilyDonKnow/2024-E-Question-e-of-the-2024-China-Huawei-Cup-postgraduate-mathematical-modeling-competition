# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 02:16:45 2024

@author: Realunknown
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def draw_MAE():
    fig, ax = plt.subplots(figsize=(7, 5),dpi = 300)
    names = dt["features"]
    x = range(len(names))
    
    ax.plot(x, dt["MAE"], color = 'darkred', marker = 'o', linestyle = '-',markersize=8, label = 'MAE(ablation)')
    
    ax.set(xlabel='Features', ylabel='MAE of Core Loss (W/m^3)',
           title='Predicted MAE with Different Features')
    #plt.ylim(26,48)
    ax.grid()
    plt.legend(prop={'size': 10}) # 显示图例
    plt.xticks(x, names,rotation = 28)
    plt.axhline(y=2045.284817,ls = "--",color = "red")
    plt.text(-0.2, 10000, "baseline: 2.513", fontsize = 12,color ="red" )

    #fig.savefig("test.png")
    plt.show()
    return 0

def draw_R_square_and_RMSE():
   fig, ax = plt.subplots(figsize=(7, 5),dpi = 300)
   names = dt["features"]
   x = range(len(names))
   
   ax.plot(x, dt["R_square"], color = 'steelblue', marker = 'o', linestyle = '-',markersize=8, label = 'R-square(ablation)')
   ax.plot(x, dt["RMSE"], color = 'salmon', marker = 'o', linestyle = '-',markersize=8, label = 'RMSE(ablation)')
   
   ax.set(xlabel='Features', ylabel='RMSE and R-square of Core Loss',
          title='Predicted RMSE and R-square with Different Features')
   #plt.ylim(32,60)
   ax.grid()
   plt.legend(prop={'size': 10}) # 显示图例
   plt.xticks(x, names,rotation = 28)
   plt.axhline(y=0.002237868,ls = "--",color = "red")
   plt.text(3.8, 0.018, "baseline: 0.00224", fontsize = 12,color ="red" )
   plt.axhline(y=0.9998713,ls = "--",color = "blue")
   plt.text(3.8, 0.95, "baseline: 0.99987", fontsize = 12,color ="blue" )
   
   #fig.savefig("test.png")
   plt.show()
   return 0

if __name__ == '__main__' :
    #data_path = "D:/新建文件夹/经历：课题组事务/科研任务：机器学习/4. HEA原子特性回归/more_data/final_data/"
    data_path = "./"
    dt = pd.read_csv(data_path + "ablation.csv")
    #dt_last_except = pd.read_csv(data_path + "last_3_except.csv")
    draw_MAE()
    draw_R_square_and_RMSE()