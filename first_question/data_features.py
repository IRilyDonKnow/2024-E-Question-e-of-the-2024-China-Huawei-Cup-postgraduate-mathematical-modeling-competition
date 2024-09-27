# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 07:30:19 2024

@author: Realunknown

More Details: https://mp.weixin.qq.com/s/aYbIweGalUxIOkWxpDR6lA.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import math
from scipy.fftpack import fft
#from pycwt import wavelet
#from pycwt.helpers import find
from matplotlib.pyplot import MultipleLocator

def time_domain(data_line):
    #均值
    df_mean = data_line.mean()
    #方差
    df_var = data_line.var()
    #标准差
    df_std = data_line.std()
    #均方根
    df_rms = math.sqrt(pow(df_mean,2) + pow(df_std,2))
    #偏度
    df_skew = data_line.skew()
    #峭度
    df_kurt = data_line.kurt()
    Sum=0
    for i in range(len(data_line)):
        Sum += math.sqrt(abs(data_line[i]))
    #波形因子
    df_form = df_rms / (abs(data_line).mean())
    #峰值因子
    df_peak = (max(data_line)) / df_rms
    #脉冲因子
    df_impulse = (max(data_line)) / (abs(data_line).mean())
    #裕度因子
    df_clearance= (max(data_line)) / pow((Sum/(len(data_line))),2)
    feature_time_list = [df_mean,df_var,df_std,df_rms,df_skew,df_kurt,df_form,df_peak,df_impulse,df_clearance]
    return feature_time_list

def get_fft_power_spectrum(y_values, N, f_s, f):
    y_values = y_values.tolist()
    f_values = np.linspace(0, f_s/f, int(N/f))
    fft_values_ = np.abs(fft(y_values))
    fft_values = 2.0/N * (fft_values_[0:int(N/2)])
 
    # power spectrum 直接周期法
    ps_values = fft_values**2 / N
 
    # 自相关傅里叶变换法
    cor_x = np.correlate(y_values, y_values, 'same')    # 自相关
    cor_X = fft(cor_x, N)                 
    ps_cor = np.abs(cor_X)
    ps_cor_values = 10*np.log10(ps_cor[0:int(N/2)] / np.max(ps_cor))
    
    return f_values, fft_values, ps_values, ps_cor_values

def integrate(data_line_list_abs,x):
    result = 0
    for data in data_line_list_abs:
        result += data*0.01 
    return result

def wave_shape(data_line):
    data_line_list = data_line.tolist()
    data_line_list = [data*100 for data in data_line_list]
    data_line_differ = [data_line_list[i+1] - data_line_list[i] for i in range(len(data_line_list) -1)]
    #极值点个数extreme_value_points
    ex_value = [abs(data_line_differ[i] - data_line_differ[i + 1]) for i in range(len(data_line_differ)-1)]
    ex_value_num = len([num for num in ex_value if num > 0.03])
    #绝对面积
    data_line_list_abs = [abs(ele) for ele in data_line_list]
    x = np.linspace(0, len(data_line_list),len(data_line_list))
    area = integrate(data_line_list_abs,x)
    #导数绝对值的平均值
    de_sqr = np.mean(data_line_list_abs)
    #导数方差
    de_var = np.std(data_line_list)
    #曲线平滑度
    curve_smoothness = np.std(ex_value)
    #最大角度
    max_angle = max(data_line_list_abs)
    
    #最小角度
    min_angle = min(data_line_list_abs)
    
    features_shape_list = [ex_value_num,area,de_sqr,de_var,curve_smoothness,max_angle,min_angle]
    return features_shape_list

def freq_domain(data_line):
    N = len(data_line)
    f_s = 2048
    f_values, fft_values, ps_values, ps_cor_values = get_fft_power_spectrum(data_line, N, f_s, 2)
    
    P = ps_values
    f = fft_values
 
    S = []
    for i in range(N//2):
        P1 = P[i]
        f1 = fft_values[i]
        s1 = P1*f1
        S.append(s1)
 
    # 求取重心频率
    S1 = np.sum(S)/np.sum(P)
    # 求平均频率
    S2 = np.sum(P)/len(P)
    #频率标准差
    S = []
    for i in range(N//2):
        P1 = P[i]
        f1 = fft_values[i]
        s2 = P1*((f1-S1)**2)
        S.append(s2) 
    S3 = np.sqrt(np.sum(S) / np.sum(P))
    #均方根频率
    S = []
    for i in range(N//2):
       P1 = P[i]
       f1 = fft_values[i]
       s2 = P1*(f1**2)
       S.append(s2) 
    S4 = np.sqrt(np.sum(S) / np.sum(P))
    
    features_freq_domain = [S1,S2,S3,S4]
    return features_freq_domain

def wavelet(data_line):
    features_wavelet = []
    return features_wavelet

if __name__ == '__main__' :
    data_path = './'
    count_list = []
    headers_train = ["index","shape_ex_point","shape_area","shape_de_mean","shape_de_var","shape_curve_smooth","shape_angle_max","shape_angle_min","time_mean","time_var","time_std","time_rms","time_skew","time_kurt","time_form","time_peak","time_impulse","time_clearance","freq_S1","freq_S2","freq_S3","freq_S4","target"]
    headers_test = ["index","shape_ex_point","shape_area","shape_de_mean","shape_de_var","shape_curve_smooth","shape_angle_max","shape_angle_min","time_mean","time_var","time_std","time_rms","time_skew","time_kurt","time_form","time_peak","time_impulse","time_clearance","freq_S1","freq_S2","freq_S3","freq_S4"]
    #file_names = ["trainning_material_1","trainning_material_2","trainning_material_3","trainning_material_4"]
    #file_names = ["trainning_material_1"]
    file_names = ["test_q3"]
    wave_data = []
    features_list = []
    index = 0
    for names in file_names:
        df = pd.read_csv(data_path + 'csv/' + names + '.csv', encoding='utf-8')
        df_mfd = df.iloc[:,5:]
        for i in range(len(df_mfd)):
            print(index)
            #if df["励磁波形"][i] == "正弦波":
            #    target = 1
            #elif df["励磁波形"][i] == "三角波":
            #    target = 2
            #elif df["励磁波形"][i] == "梯形波":
            #    target = 3
            features = []
            data_line = df_mfd.iloc[i]
            #Wave shape
            features_shape = wave_shape(data_line)
            #Time Domain
            features_time_domain = time_domain(data_line)
            
            #Frequency Domain
            features_freq_domain = freq_domain(data_line)
            #Wavelet
            #features_wavelet = wavelet(data_line)
            #features = features_shape + features_time_domain + features_freq_domain + features_wavelet
            #features = [i] + features_shape + features_time_domain + features_freq_domain + [target]
            features = [i] + features_shape + features_time_domain + features_freq_domain
            features_list.append(features)
            index += 1
    df_final = pd.DataFrame(features_list, columns=headers_test)
    df_final.to_csv("./features_values_test_q3.csv",index = False)
            
        
        