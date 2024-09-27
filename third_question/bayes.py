# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 22:20:57 2024

@author: Realunknown
"""
import xgboost as xgb
import optuna
import sys
sys.path.append("./")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

def paint_trend(trend):
    x = list(range(len(trend[0])))
    fig, axes = plt.subplots(10, 1, figsize=(8,24), dpi = 600)
    y_name = ['time_skew','time_kurt','time_form','time_peak/max_al','time_impulse','time_clearance','temp','materials','freq','target_mag']
    for i in range(len(trend) -1):
        axes[i].plot(x, trend[i],color = "darkblue")
        axes[i].set_xlabel('Steps')
        axes[i].set_ylabel(y_name[i])
    axes[9].plot(x, trend[-1],color = "darkred")
    axes[9].set_xlabel('Steps')
    axes[9].set_ylabel("Mag_target")
    plt.show()
    return 0

def calculate_y(test_x):
    clf = xgb.XGBRegressor()
    booster = xgb.Booster()
    booster.load_model('models/model_xgboost.bin')
    clf._Booster = booster
    model_xgb = clf
    min_value = 2.6186892424572115
    delta_value = 3.9395550971001576
    
    y_pred = model_xgb.predict(test_x)[0]
    y_real = np.power(10,y_pred *delta_value + min_value)
    return y_real

def objective(trial):
    time_skew = trial.suggest_float('time_skew', -0.05, 0.07)
    time_peak = trial.suggest_float('time_peak/max_al', 1.0, 2.0)
    time_kurt = trial.suggest_float('time_kurt', -2.0, -1.0)
    time_impulse = trial.suggest_float('time_impulse',1.2 , 2.2)
    time_form = trial.suggest_float('time_form', 1.1,1.7 )
    time_clearance = trial.suggest_float('time_clearance', 1.5, 2.5)
    temp = trial.suggest_categorical('temp', [25, 50, 70, 90])
    materials = trial.suggest_categorical('materials', [1, 2, 3, 4])
    freq = trial.suggest_int('freq', 0, 55000)

    
    data = [time_skew,time_peak,time_kurt,time_impulse,time_form,time_clearance,temp,materials,freq]
    datas = []
    datas.append(data)
    datas.append(data)
    df = pd.DataFrame(datas)
    
    if len(study.trials) > 1:
        trial = study.best_trial
        y_old = trial.value
    else:
        y_old = 100

    y_pred =  calculate_y(df)
    
    if y_pred < y_old:
        y.append(y_pred)
        time_skew_hist.append(time_skew)
        time_peak_hist.append(time_peak)
        time_kurt_hist.append(time_kurt)
        time_impulse_hist.append(time_impulse)
        time_form_hist.append(time_form)
        time_clearance_hist.append(time_clearance)
        temp_hist.append(temp)
        materials_hist.append(materials) 
        freq_hist.append(freq)
    return y_pred

if __name__ == '__main__' :
    y = []
    time_skew_hist = []
    time_peak_hist = []
    time_kurt_hist = []
    time_impulse_hist = []
    time_form_hist = []
    time_clearance_hist = []
    temp_hist = []
    materials_hist =[]
    freq_hist = []
    study = optuna.create_study(direction='minimize')#searching mode
    study.optimize(objective, n_trials=1000)
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    trend = []
    trend.append(time_skew_hist)
    trend.append(time_kurt_hist)
    trend.append(time_form_hist)
    trend.append(time_peak_hist)
    trend.append(time_impulse_hist)
    trend.append(time_clearance_hist)
    trend.append(temp_hist)
    trend.append(materials_hist)
    trend.append(freq_hist)
    trend.append(y)
    paint_trend(trend)