# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:31:27 2024

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
from scipy.optimize import minimize

def linear_regression(endog, exog):
    mod = sm.OLS(endog, exog).fit()
    return mod

def draw_r_square(x_observed, y_predicted_nonlinear,rmse_nonlinear):
    fig, ax = plt.subplots(figsize=(5,5),dpi = 600)
    r_squared_nonlinear = r2_score(x_observed, y_predicted_nonlinear)
    x_max = math.ceil(max(x_observed))
    x_min = math.floor(min(x_observed))
    y_max = math.ceil(max(y_predicted_nonlinear))
    y_min = math.floor(min(y_predicted_nonlinear))
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
    ax.scatter(x_observed, y_predicted_nonlinear, s = 80, marker='o',edgecolors='royalblue',facecolors='cornflowerblue', label='Noninear Fit-SE-corrected(T) in Material 4\n                    (R^2 = {:.3f} , RMSE = {:.3F})'.format(r_squared_nonlinear,rmse_nonlinear))
    plt.legend(loc='best',prop = font)
    plt.grid(linestyle="-.")
    plt.show()
    return 0

if __name__ == '__main__' :
    data_path = "./"
    df = pd.read_csv(data_path + "features_regress_same_material_wave_shape_2.csv")
    df_temp = df["temp"]
    df_freq_log = df["freq"].apply(np.log)
    df_max_al_log = df["max_al"].apply(np.log)
    y = df["core_loss"].apply(np.log)
    
    #Non-linear-SE-corrected
    def nonlinear_model(variable, a1, a2, a3, a4, a5, a6, a7, a8,a9,a10,a11):
        lnf,lnB,T = variable
        result = a1 + a2*lnf + a3*lnB + a4*T + a5*T*lnf + a6*T*lnB + a7*lnf*lnB + a8*T*lnf*lnB + np.log(a9 - a10*T + a11*T**2)
        return result
    
    y_real = y.tolist()
    
    popt, pcov = curve_fit(nonlinear_model, (df_freq_log, df_max_al_log,df_temp), y)
    a1_fit, a2_fit, a3_fit, a4_fit, a5_fit, a6_fit, a7_fit, a8_fit, a9_fit, a10_fit, a11_fit = popt
    y_fit_nonlinear = nonlinear_model((df_freq_log, df_max_al_log,df_temp), a1_fit, a2_fit, a3_fit, a4_fit, a5_fit, a6_fit, a7_fit, a8_fit, a9_fit, a10_fit, a11_fit)
    
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_reg = np.sum((y_fit_nonlinear - np.mean(y)) ** 2)
    r_squared_nonlinear = ss_reg / ss_total
    rmse_nonlinear = np.sqrt(np.mean((y - y_fit_nonlinear) ** 2))
 
    print("non_linear-R-square:", r_squared_nonlinear)
    print("nonlinear-RMSE:", rmse_nonlinear)
    
    draw_r_square(y_real,y_fit_nonlinear,rmse_nonlinear)
    
    #optimization
    def f(x):
        return a1_fit + a2_fit*x[0] + a3_fit*x[1] + a4_fit*x[2] + a5_fit*x[2]*x[0] + a6_fit*x[2]*x[1] + a7_fit*x[0]*x[1] + a8_fit*x[2]*x[0]*x[1] + np.log(a9_fit - a10_fit*x[2] + a11_fit*x[2]**2)
        
    x0 = np.array([5.0, 5.0, 20.0])
    result = minimize(f, x0, method='Nelder-Mead')
    print(result)