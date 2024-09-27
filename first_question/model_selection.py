# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:53:34 2024

@author: Realunknown
"""

from sklearn import preprocessing

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import shap
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score
import math
import numpy as np
import torch

def draw_acc(ACC):
    fig, ax = plt.subplots(figsize=(5, 5),dpi = 300)
    names = ["Xgboost","RF","DT","SVC","KNN"]
    x = range(len(names))
        
    ax.plot(x, ACC, color = 'darkblue', marker = '+', linestyle = '-',markersize=8, label = 'Accuracy')
    ax.set(xlabel='Model', ylabel='Accuracy in Models',
               )
    plt.ylim(0,1.2)
    plt.legend(prop={'size': 10}) # 显示图例
    plt.xticks(x, names)

        #fig.savefig("test.png")
    plt.grid(True)
    plt.show()
    return 0

def draw_cm(cm):
    plt.figure(figsize=(8, 8))
    plt.tight_layout()
    xtick=["1","2","3"]
    ytick=["1","2","3"]
    sns.heatmap(cm,cmap='Blues',fmt='3d',annot=True,annot_kws={"fontsize":20},linecolor='black',linewidths=0,cbar=False,xticklabels=xtick, yticklabels=ytick)
    plt.xlabel("Predicted",fontsize = 20)
    plt.ylabel("Actual",fontsize = 20)
    return 0

def xgboost_pred(train_x,train_y,test_x,test_y):
    #xgb_boost
    model_xgb = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=12, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
    model_xgb.fit(train_x,train_y)
    print(train_x)

    y_pred = model_xgb.predict(test_x)
    y_pred = list(y_pred)
    y_test = list(test_y)
    
    ACC = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
    draw_cm(cm)
    print(f"cm: \n{cm}")
    cr = classification_report(y_test, y_pred) # 分类报告
    print(f"cr:  \n{cr}")
    xgb.plot_importance(model_xgb,max_num_features = 6)
    return model_xgb,y_pred,y_test, ACC

def rf_pred(train_x,train_y,test_x,test_y):
    #random_forest
    model_rf = RandomForestClassifier(n_estimators=1000, random_state=62,max_features = 1.0)
    model_rf.fit(train_x,train_y)

    y_pred = model_rf.predict(test_x)
    y_pred = list(y_pred)
    y_test = list(test_y)

    ACC = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
    draw_cm(cm)
    print(f"cm: \n{cm}")
    cr = classification_report(y_test, y_pred) # 分类报告
    print(f"cr:  \n{cr}")
    return model_xgb,y_pred,y_test, ACC

def dt_pred(train_x,train_y,test_x,test_y):
    #decision tree
    model_dt = DecisionTreeClassifier(max_depth=10)
    model_dt.fit(train_x,train_y)

    y_pred = model_dt.predict(test_x)
    y_pred = list(y_pred)
    y_test = list(test_y)

    ACC = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
    draw_cm(cm)
    print(f"cm: \n{cm}")
    cr = classification_report(y_test, y_pred) # 分类报告
    print(f"cr:  \n{cr}")
    return model_xgb,y_pred,y_test, ACC


def svc_pred(train_x,train_y,test_x,test_y):
    #decision tree
    model_svc = SVC(kernel='rbf', degree=2, C=1000, gamma=0.01)
    model_svc.fit(train_x, train_y)

    y_pred = model_svc.predict(test_x)
    y_pred = list(y_pred)
    y_test = list(test_y)

    ACC = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
    draw_cm(cm)
    print(f"cm: \n{cm}")
    cr = classification_report(y_test, y_pred) # 分类报告
    print(f"cr:  \n{cr}")
    return model_svc,y_pred,y_test, ACC
    
def knn_pred(train_x,train_y,test_x,test_y):
    model_knn = KNeighborsClassifier(n_neighbors=2,p=2,metric="minkowski")
    model_knn.fit(train_x,train_y)

    y_pred = model_knn.predict(test_x)
    y_pred = list(y_pred)
    y_test = list(test_y)
    
    ACC = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
    draw_cm(cm)
    print(f"cm: \n{cm}")
    cr = classification_report(y_test, y_pred) # 分类报告
    print(f"cr:  \n{cr}")
    return model_knn,y_pred,y_test, ACC

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
    data_path = "./"
    models_save = "./models/"
    df = pd.read_csv(data_path + "features/features.csv")
    headers = ['time_skew','time_kurt','time_form','time_peak','time_impulse','time_clearance','target']
    df_ready = df.iloc[:, 12:18]
    df_ready["target"] = df["target"]
    df_random = df_ready.sample(frac=1)
    
    train_df = df_random.iloc[:int(len(df_ready * 0.8))]
    test_df = df_random.iloc[int(len(df_ready) * 0.8):]
    
    train_x = train_df.drop('target', 1)
    train_y = train_df['target']
    train_y = train_y.sub(1)
    test_x = test_df.drop('target', 1)
    test_y = test_df['target']
    test_y = test_y.sub(1)
    
    #ss = preprocessing.StandardScaler()
    #train_x = ss.fit_transform(train_x)
    #test_x = ss.fit_transform(test_x)
    
    ACC = []
    AUC = []
    
    #xgb
    
    model_xgb,y_pred_xgb,y_real_xgb,ACC_xgb = xgboost_pred(train_x,train_y,test_x,test_y)#xgboost
    ACC.append(ACC_xgb)
    x_headers = ( 'time_skew','time_kurt','time_form','time_peak','time_impulse','time_clearance','target')
    test_x = pd.DataFrame(test_x)
    #test_x.columns = x_headers
    #shap_tree(model_xgb, test_x)
    
    
    #rf
    
    model_rf,y_pred_rf,y_real_rf,ACC_rf = rf_pred(train_x,train_y,test_x,test_y)#xgboost
    ACC.append(ACC_rf)
    
    x_headers = ("Electronegativity","Electronegativity Difference","Charge Transfer","Bader Volume","Voronoi Volume","t$_{2g}$","e$_{g}$","Atomic Displacement")
    
    test_x = pd.DataFrame(test_x)
    #test_x.columns = x_headers
    
    #dt
    model_dt,y_pred_dt,y_real_dt,ACC_dt = dt_pred(train_x,train_y,test_x,test_y)#decision tree
    ACC.append(ACC_dt)
    
    x_headers = ("Electronegativity","Electronegativity Difference","Charge Transfer","Bader Volume","Voronoi Volume","t$_{2g}$","e$_{g}$","Atomic Displacement")
            
    test_x = pd.DataFrame(test_x)
    #test_x.columns = x_headers

    
    #svc
    model_svc,y_pred_svc,y_real_svc,ACC_svc= svc_pred(train_x,train_y,test_x,test_y)#decision tree
    ACC.append(ACC_svc)
    
    x_headers = ("Electronegativity","Electronegativity Difference","Charge Transfer","Bader Volume","Voronoi Volume","t$_{2g}$","e$_{g}$","Atomic Displacement")
    
    test_x = pd.DataFrame(test_x)
    #test_x.columns = x_headers
    #shap_kernel(models_svc, test_x)
    
    #knn
    model_knn,y_pred_knn,y_real_knn,ACC_knn = knn_pred(train_x,train_y,test_x,test_y)#decision tree
    ACC.append(ACC_knn)
    
    x_headers = ("Electronegativity","Electronegativity Difference","Charge Transfer","Bader Volume","Voronoi Volume","t$_{2g}$","e$_{g}$","Atomic Displacement")
    
    test_x = pd.DataFrame(test_x)
    #test_x.columns = x_headers
    draw_acc(ACC)
    
    #Test data
    df = pd.read_csv(data_path + "features/features_values_test.csv")
    headers_test = ['time_skew','time_kurt','time_form','time_peak','time_impulse','time_clearance']
    df_ready_test = df.iloc[:, 12:18]
    y_pred_test_xgb = model_xgb.predict(df_ready_test)
    y_pred_test_rf = model_rf.predict(df_ready_test)
    y_pred_test_svc = model_svc.predict(df_ready_test)
    y_pred_test_knn = model_knn.predict(df_ready_test)
    y_pred_test_dt = model_dt.predict(df_ready_test)