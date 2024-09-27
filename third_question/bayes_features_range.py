# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 23:43:42 2024

@author: Realunknown
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data_path = "./"
headers = ['time_skew','time_kurt','time_form','time_peak/max_al','time_impulse','time_clearance','temp','freq','materials','target_mag']
df = pd.read_csv(data_path + "features_regress.csv")
df = df.drop("index",1)

antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864'] 

f, axes = plt.subplots(3, 3, figsize=(16, 16), sharex=True,dpi = 600)
sns.despine(left=True)

for i in range(3):
    for j in range(3):
        b = sns.violinplot( y=headers[i*3+j] , data=df, palette=antV, ax=axes[(i, j)])
        #b.set_yticklabels(b.get_yticks(), size = 12)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8)
plt.show()