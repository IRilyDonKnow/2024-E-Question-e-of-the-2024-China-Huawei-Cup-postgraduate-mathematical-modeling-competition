# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:35:25 2024

@author: Realunknown
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
import math
from scipy.optimize import curve_fit

def linear_regression(endog, exog):
    mod = sm.OLS(endog, exog).fit()
    return mod

def draw_r_square(x_observed,y_predicted_linear, y_predicted_nonlinear,rmse_linear,rmse_nonlinear):
    fig, ax = plt.subplots(figsize=(5,5),dpi = 600)
    r_squared_linear = r2_score(x_observed, y_predicted_linear)
    r_squared_nonlinear = r2_score(x_observed, y_predicted_nonlinear)
    x_max = math.ceil(max(x_observed))
    x_min = math.floor(min(x_observed))
    y_max = math.ceil(max(y_predicted_linear))
    y_min = math.floor(min(y_predicted_linear))
    Max = max(x_max,y_max)
    Min = min(x_min,y_min)
    plt.xlabel('Observed/log(W/m^3)')
    plt.ylabel('Predicted/log(W/m^3)')
    x = range(Min,Max + 1)
    y = range(Min,Max + 1)
    plt.xlim(Min,Max)
    plt.ylim(Min,Max)
    plt.plot(x,y,color = 'darkorange')
    font = {"size":10}
    ax.scatter(x_observed, y_predicted_linear, s = 80, marker='o',edgecolors='steelblue',facecolors='skyblue', label='Linear Fit-SE(R^2 = {:.3f} , RMSE = {:.3F})'.format(r_squared_linear,rmse_linear))
    ax.scatter(x_observed, y_predicted_nonlinear, s = 80, marker='o',edgecolors='firebrick',facecolors='lightcoral', label='Noninear Fit-SE-corrected\n                    (R^2 = {:.3f} , RMSE = {:.3F})'.format(r_squared_nonlinear,rmse_nonlinear))
    plt.legend(loc='best',prop = font)
    plt.grid(linestyle="-.")
    plt.show()
    return 0

if __name__ == '__main__' :
    data_path = "./"
    df = pd.read_csv(data_path + "features_regress_same_material_wave_shape.csv")
    df_temp = df["temp"]
    df_freq_log = df["freq"].apply(np.log)
    df_max_al_log = df["max_al"].apply(np.log)
    y = df["core_loss"].apply(np.log)
    
    #Linear Fitting SE
    exog = sm.add_constant(np.c_[df_freq_log, df_max_al_log])
    # 执行多元线性回归
    model = linear_regression(y, exog)
    # 输出模型结果
    print(model.summary())
    
    y_fit_linear = []
    for i in range(len(df_freq_log)):
        x1 = df_freq_log[i]
        x2 = df_max_al_log[i]
        y_pred = 1.6340*x1 + 2.5245*x2 - 1.8763
        y_fit_linear.append(y_pred)
    y_real = y.tolist()
    
    #Non-linear-SE-corrected
    def nonlinear_model(variable, a1, a2, a3, a4, a5, a6):
        lnf,lnB,T = variable
        result = a1 + a2*lnf + a3*lnB + np.log(a4 - a5*T + a6*T**2)
        return result
    
    
    popt, pcov = curve_fit(nonlinear_model, (df_freq_log, df_max_al_log,df_temp), y)
    a1_fit, a2_fit, a3_fit, a4_fit, a5_fit, a6_fit = popt
    y_fit_nonlinear = nonlinear_model((df_freq_log, df_max_al_log,df_temp), a1_fit, a2_fit, a3_fit, a4_fit, a5_fit, a6_fit)
    
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_reg = np.sum((y_fit_nonlinear - np.mean(y)) ** 2)
    r_squared_nonlinear = ss_reg / ss_total
    rmse_nonlinear = np.sqrt(np.mean((y - y_fit_nonlinear) ** 2))
 
    print("non_linear-R-square:", r_squared_nonlinear)
    print("nonlinear-RMSE:", rmse_nonlinear)
    
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_reg = np.sum((y_fit_linear - np.mean(y)) ** 2)
    r_squared_linear = ss_reg / ss_total
    rmse_linear = np.sqrt(np.mean((y - y_fit_linear) ** 2))
 
    print("linear-R-square:", r_squared_linear)
    print("linear-RMSE:", rmse_linear)
    
    draw_r_square(y_real, y_fit_linear,y_fit_nonlinear,rmse_linear,rmse_nonlinear)
    