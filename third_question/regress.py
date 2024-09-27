# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:16:31 2024

@author: Realunknown
"""
from sklearn import preprocessing
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.metrics import make_scorer, mean_squared_error
import shap
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score
import math
import numpy as np
import torch

def draw_r_square(x_observed,y_predicted,model_name):
    x_observed = [ele*delta_value + min_value for ele in x_observed]
    y_predicted = [ele*delta_value + min_value for ele in y_predicted]
    fig, ax = plt.subplots(figsize=(5,5),dpi = 600)
    r_squared = r2_score(x_observed, y_predicted)
    x_max = math.ceil(max(x_observed))
    x_min = math.floor(min(x_observed))
    y_max = math.ceil(max(y_predicted))
    y_min = math.floor(min(y_predicted))
    Max = max(x_max,y_max)
    Min = min(x_min,y_min)
    plt.xlabel('Observed log(W/m^3)')
    plt.ylabel('Predicted log(W/m^3)')
    x = range(Min,Max + 1)
    y = range(Min,Max + 1)
    plt.xlim(Min,Max)
    plt.ylim(Min,Max)
    plt.plot(x,y,color = 'darkorange')
    font = {"size":10}
    ax.scatter(x_observed, y_predicted, s = 80, marker='o',edgecolors='steelblue',facecolors='skyblue', label='All Features in {} (R^2 = {:.5f})'.format(model_name,r_squared))
    #ax.scatter(x_observed, y_predicted, s = 80, marker='o',edgecolors='steelblue',facecolors='skyblue', label='Except Vv and Ad (R = {:.3f})'.format(np.sqrt(r_squared)))
    #ax.scatter(x_observed, y_predicted, s = 80, marker='o',edgecolors='steelblue',facecolors='skyblue', label='Features except Voronoi Volume (R = {:.3f})'.format(np.sqrt(r_squared)))
    #plt.savefig(save_path + '/r_square_curve.png', dpi=600)
    plt.legend(loc='best',prop = font)
    plt.grid(linestyle="-.")
    plt.show()
    return r_squared

def xgboost_pred(train_x,train_y,test_x,test_y):
    #xgb_boost
    model_xgb = xgb.XGBRegressor(max_depth= 15, learning_rate=0.01, n_estimators=2200)
    model_xgb.fit(train_x,train_y)

    y_pred = model_xgb.predict(test_x)
    y_pred = list(y_pred)
    y_test = list(test_y)

    MAE_list = [abs(np.power(10,(y_pred[i]*delta_value + min_value))- np.power(10,(y_test[i]*delta_value + min_value))) for i in range(len(y_pred))]
    MAE = sum(MAE_list)/len(MAE_list)
    print('mae:{}'.format(MAE))
    RMSE_list = [(y_pred[i]-y_test[i])**2 for i in range(len(y_pred))]
    RMSE = np.sqrt(sum(RMSE_list)/len(RMSE_list))
    print('rmse:{}'.format(RMSE))
    return model_xgb,y_pred,y_test,MAE,RMSE

def rf_pred(train_x,train_y,test_x,test_y):
    #random_forest
    model_rf = RandomForestRegressor(n_estimators=1000, random_state=62,max_features = 1.0)
    model_rf.fit(train_x,train_y)

    y_pred = model_rf.predict(test_x)
    y_pred = list(y_pred)
    y_test = list(test_y)

    MAE_list = [abs(np.power(10,(y_pred[i]*delta_value + min_value))- np.power(10,(y_test[i]*delta_value + min_value))) for i in range(len(y_pred))]
    MAE = sum(MAE_list)/len(MAE_list)
    print('mae:{}'.format(MAE))
    RMSE_list = [(y_pred[i]-y_test[i])**2 for i in range(len(y_pred))]
    RMSE = np.sqrt(sum(RMSE_list)/len(RMSE_list))
    print('rmse:{}'.format(RMSE))
    return model_rf,y_pred,y_test,MAE,RMSE

def dt_pred(train_x,train_y,test_x,test_y):
    #decision tree
    model_dt = DecisionTreeRegressor(max_depth=10)
    model_dt.fit(train_x,train_y)

    y_pred = model_dt.predict(test_x)
    y_pred = list(y_pred)
    y_test = list(test_y)

    MAE_list = [abs(np.power(10,(y_pred[i]*delta_value + min_value))- np.power(10,(y_test[i]*delta_value + min_value))) for i in range(len(y_pred))]
    MAE = sum(MAE_list)/len(MAE_list)
    print('mae:{}'.format(MAE))
    RMSE_list = [(y_pred[i]-y_test[i])**2 for i in range(len(y_pred))]
    RMSE = np.sqrt(sum(RMSE_list)/len(RMSE_list))
    print('rmse:{}'.format(RMSE))
    return model_dt,y_pred,y_test,MAE,RMSE


def svr_pred(train_x,train_y,test_x,test_y):
    #decision tree
    model_svr = SVR(kernel='rbf', degree=2, C=1000, epsilon=0.1, gamma=0.01)
    model_svr.fit(train_x, train_y)

    y_pred = model_svr.predict(test_x)
    y_pred = list(y_pred)
    y_test = list(test_y)

    MAE_list = [abs(np.power(10,(y_pred[i]*delta_value + min_value))- np.power(10,(y_test[i]*delta_value + min_value))) for i in range(len(y_pred))]
    MAE = sum(MAE_list)/len(MAE_list)
    print('mae:{}'.format(MAE))
    RMSE_list = [(y_pred[i]-y_test[i])**2 for i in range(len(y_pred))]
    RMSE = np.sqrt(sum(RMSE_list)/len(RMSE_list))
    print('rmse:{}'.format(RMSE))
    return model_svr,y_pred,y_test,MAE,RMSE
    
def krr_pred(train_x,train_y,test_x,test_y):
    model_krr = KernelRidge(kernel='rbf', gamma=0.1 ,alpha = 0.1)
    model_krr.fit(train_x, train_y)

    y_pred = model_krr.predict(test_x)
    y_pred = list(y_pred)
    y_test = list(test_y)

    MAE_list = [abs(np.power(10,(y_pred[i]*delta_value + min_value))- np.power(10,(y_test[i]*delta_value + min_value))) for i in range(len(y_pred))]
    MAE = sum(MAE_list)/len(MAE_list)
    print('mae:{}'.format(MAE))
    RMSE_list = [(y_pred[i]-y_test[i])**2 for i in range(len(y_pred))]
    RMSE = np.sqrt(sum(RMSE_list)/len(RMSE_list))
    print('rmse:{}'.format(RMSE))
    return model_krr,y_pred,y_test,MAE,RMSE

def gpr_pred(train_x,train_y,test_x,test_y):
    kernel = 1.0 * RBF(1.0)
    model_gpr = GaussianProcessRegressor(kernel=kernel,
        random_state=0)
    model_gpr.fit(train_x, train_y)

    y_pred = model_gpr.predict(test_x)
    y_pred = list(y_pred)
    y_test = list(test_y)

    MAE_list = [abs(np.power(10,(y_pred[i]*delta_value + min_value))- np.power(10,(y_test[i]*delta_value + min_value))) for i in range(len(y_pred))]
    MAE = sum(MAE_list)/len(MAE_list)
    print('mae:{}'.format(MAE))
    RMSE_list = [(y_pred[i]-y_test[i])**2 for i in range(len(y_pred))]
    RMSE = np.sqrt(sum(RMSE_list)/len(RMSE_list))
    print('rmse:{}'.format(RMSE))
    return model_gpr,y_pred,y_test,MAE,RMSE

def lgr_pred(train_x,train_y,test_x,test_y):
    model_lgr = LinearRegression()
    model_lgr.fit(train_x, train_y)

    y_pred = model_lgr.predict(test_x)
    y_pred = list(y_pred)
    y_test = list(test_y)

    MAE_list = [abs(np.power(10,(y_pred[i]*delta_value + min_value))- np.power(10,(y_test[i]*delta_value + min_value))) for i in range(len(y_pred))]
    MAE = sum(MAE_list)/len(MAE_list)
    print('mae:{}'.format(MAE))
    RMSE_list = [(y_pred[i]-y_test[i])**2 for i in range(len(y_pred))]
    RMSE = np.sqrt(sum(RMSE_list)/len(RMSE_list))
    print('rmse:{}'.format(RMSE))
    return model_lgr ,y_pred,y_test,MAE,RMSE

def shap_tree(model,test_x):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_x)

    #cmap = 
    #特征总体对SHAP值的影响
    fig,ax = plt.subplots(figsize = (5,5), dpi = 300)
    shap.summary_plot(shap_values, test_x, plot_type="bar")

    #样本对于每个特征的SHAP值
    fig,ax = plt.subplots(figsize = (5,5), dpi = 300)
    shap.summary_plot(shap_values, test_x)
    return 0


def shap_kernel(model,test_x):
    explainer = shap.KernelExplainer(model.predict, test_x)
    shap_values = explainer.shap_values(test_x)
    shap_values_2 = explainer(test_x)

    #cmap = 
    #特征总体对SHAP值的影响
    plt.figure(figsize=(5, 5),dpi=1200)
    shap.summary_plot(shap_values, test_x, plot_type="bar")
    plt.show()

    #样本对于每个特征的SHAP值
    plt.figure(figsize=(5, 5),dpi=1200)
    shap.summary_plot(shap_values, test_x)
    plt.show()
    
    expected_value = explainer.expected_value
    shap_values = explainer.shap_values(test_x)[1]
    shap.decision_plot(expected_value, shap_values, 
                   test_x)
    
    shap.plots.heatmap(shap_values_2)
    
    shap.plots.waterfall(shap_values_2[7])
    
    plt.show()
    print(shap_values)
    
    torch.save(test_x,"./test_x.pt")
    torch.save(shap_values,"shap_values.pt")
    torch.save(shap_values_2,"shap_values_2.pt")
    torch.save(expected_value,"expected_value.pt")
    torch.save(model,"model.pt")
    torch.save(explainer,"explainer.pt")
    return 0

if __name__ == '__main__' :
    data_path = "../第五题/"
    headers = ['time_skew','time_kurt','time_form','max_al','time_impulse','time_clearance','temp','materials','freq','target_mag']
    df = pd.read_csv(data_path + "features_regress_for_opt.csv")
    #df.head().append(df.tail())
    df = df.drop("index",1)
    df["target_mag"] = df["target_mag"].apply(np.log10)#对target求log10
    max_value = df['target_mag'].max()
    min_value = df['target_mag'].min()
    delta_value = max_value - min_value
    df['target_mag'] = (df['target_mag'] - min_value) / delta_value
    
    #df_random = df.sample(frac=1)
    df_random = df
    
    train_df = df_random.iloc[:int(len(df * 0.8))]
    test_df = df_random.iloc[int(len(df) * 0.8):]
    
    train_x = train_df.drop('target_mag', 1)
    train_y = train_df['target_mag']
    
    test_x = test_df.drop('target_mag', 1)
    test_y = test_df['target_mag']
    
    #ss = preprocessing.StandardScaler()
    #train_x = ss.fit_transform(train_x)
    #test_x = ss.fit_transform(test_x)
    
    MAE = []
    RMSE = []
    r = []
    
    #xgb
    
    model_xgb,y_pred,y_real,MAE_xgb,RMSE_xgb = xgboost_pred(train_x,train_y,test_x,test_y)#xgboost
    MAE.append(MAE_xgb)
    RMSE.append(RMSE_xgb)
    #x_headers = ("Electronegativity","Charge Transfer","Bader Volume","Voronoi Volume","t$_{2g}$","e$_{g}$")
    x_headers = ('time_skew','time_kurt','time_form','time_peak/max_al','time_impulse','time_clearance','temp','materials','freq')
    test_x = pd.DataFrame(test_x)
    test_x.columns = x_headers
    shap_tree(model_xgb, test_x)
    explainer = shap.Explainer(model_xgb)
    shap_values = explainer(test_x)
    shap.plots.waterfall(shap_values[0])
    r_squared_xgb = r2_score(y_real, y_pred)
    r.append(r_squared_xgb)
    print('r_squared: {}'.format(r_squared_xgb))
    draw_r_square(y_real, y_pred, "Xgboost")
    
    
    #rf
    
    model_rf,y_pred,y_real,MAE_rf,RMSE_rf = rf_pred(train_x,train_y,test_x,test_y)#xgboost
    MAE.append(MAE_rf)
    RMSE.append(RMSE_rf)
    
    x_headers = ('time_skew','time_kurt','time_form','time_peak/max_al','time_impulse','time_clearance','temp','materials','freq')
    
    test_x = pd.DataFrame(test_x)
    test_x.columns = x_headers
    #shap_tree(model_rf, test_x)
    r_squared_rf = r2_score(y_real, y_pred)
    r.append(r_squared_rf)
    print('r_squared: {}'.format(r_squared_rf))
    draw_r_square(y_real, y_pred, "RF")
    
    #dt
    model_dt,y_pred,y_real,MAE_dt,RMSE_dt = dt_pred(train_x,train_y,test_x,test_y)#decision tree
    MAE.append(MAE_dt)
    RMSE.append(RMSE_dt)
    
    x_headers = ('time_skew','time_kurt','time_form','time_peak/max_al','time_impulse','time_clearance','temp','materials','freq')
            
    test_x = pd.DataFrame(test_x)
    test_x.columns = x_headers
    #shap_tree(model_dt, test_x)
    r_squared_dt = r2_score(y_real, y_pred)
    r.append(r_squared_dt)
    print('r_squared: {}'.format(r_squared_dt))
    draw_r_square(y_real, y_pred ,"DT")

    
    #svr
    models_svr,y_pred,y_real,MAE_svr,RMSE_svr = svr_pred(train_x,train_y,test_x,test_y)#decision tree
    MAE.append(MAE_svr)
    RMSE.append(RMSE_svr)
    
    x_headers = ('time_skew','time_kurt','time_form','time_peak/max_al','time_impulse','time_clearance','temp','materials','freq')
    
    test_x = pd.DataFrame(test_x)
    test_x.columns = x_headers
    #shap_kernel(models_svr, test_x)
    r_squared_svr = r2_score(y_real, y_pred)
    r.append(r_squared_svr)
    print('r_squared: {}'.format(r_squared_svr))
    draw_r_square(y_real, y_pred,"SVR" )
    

    #krr
    model_krr,y_pred,y_real,MAE_krr,RMSE_krr = krr_pred(train_x,train_y,test_x,test_y)#decision tree
    MAE.append(MAE_krr)
    RMSE.append(RMSE_krr)
   
    x_headers = ('time_skew','time_kurt','time_form','time_peak/max_al','time_impulse','time_clearance','temp','materials','freq')
    
    test_x = pd.DataFrame(test_x)
    test_x.columns = x_headers
    #shap_kernel(model_krr, test_x)
    r_squared_krr = r2_score(y_real, y_pred) 
    r.append(r_squared_krr)
    print('r_squared: {}'.format(r_squared_krr))
    draw_r_square(y_real, y_pred ,"KRR")

    """
    #gpr
    
    model_gpr,y_pred,y_real,MAE_gpr,RMSE_gpr = gpr_pred(train_x,train_y,test_x,test_y)#decision tree
    MAE.append(MAE_gpr)
    RMSE.append(RMSE_gpr)
    
    x_headers = ('time_skew','time_kurt','time_form','time_peak/max_al','time_impulse','time_clearance','temp','freq','materials')
            
    test_x = pd.DataFrame(test_x)
    test_x.columns = x_headers
    #shap_kernel(model_gpr, test_x)
    r_squared_gpr = r2_score(y_real, y_pred)
    r.append(r_squared_gpr)
    print('r_squared: {}'.format(r_squared_gpr))
    draw_r_square(y_real, y_pred,"GPR" )
    """
    
    #lgr

    r_squared_lgr = -1
    #while(r_squared_lgr < 0):
    model_lgr,y_pred,y_real,MAE_lgr,RMSE_lgr = lgr_pred(train_x,train_y,test_x,test_y)#decision tree
    MAE.append(MAE_lgr)
    RMSE.append(RMSE_lgr)
        
    x_headers = ('time_skew','time_kurt','time_form','time_peak/max_al','time_impulse','time_clearance','temp','materials','freq')
        
    test_x = pd.DataFrame(test_x)
    test_x.columns = x_headers
    #shap_kernel(model_lgr, test_x)
    r_squared_lgr = r2_score(y_real, y_pred)
    print('r_squared: {}'.format(r_squared_lgr))
    draw_r_square(y_real, y_pred ,"LGR")
    r.append(r_squared_lgr)
    
    