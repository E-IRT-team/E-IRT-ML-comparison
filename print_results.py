# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:11:19 2020

@author: konst
"""

import numpy as np
from pprint import pprint        

datalist = ['noise30','noise60','sizesmall','sizemedium'] #'latentfeatures4','latentfeatures30'
methodlist = ['MLP','KNN','DT','RF','GB','QDA'] # EIRT
settings = ['rows','columns','items']

path = 'C:/Users/u0135479/Documents/GitHub/E-IRT-ML-comparison/Results/' 

final_auroc,final_aupr,final_mse = [],[],[]
for s in settings: # 3 settings: rows, columns, 
    auroc,aupr,mse = [],[],[]
    for data in datalist: # length: 4
        temp_auroc,temp_aupr,temp_mse = [],[],[]
        for method in methodlist: #length 6 or 7
            filename = path + data + method + s + '.npy'
            resmethod = np.nanmean(np.load(filename), 0).round(3)
            temp_auroc.extend([resmethod[0]])
            temp_aupr.extend([resmethod[1]])
            temp_mse.extend([resmethod[2]])
            
        auroc.append(temp_auroc)
        aupr.append(temp_aupr)
        mse.append(temp_mse)
    pprint(auroc) 
    print(" ")

    final_auroc.append(auroc)
    final_aupr.append(aupr)
    final_mse.append(mse)


''' we store AUROC performance for further statistical testing'''
import pandas as pd 

df = pd.DataFrame()
for line in final_auroc: ## the user can anyother measure 
    df = df.append(line)
df.columns = methodlist #synchronise column names with methodlist list
    
pd.DataFrame(df).to_csv("results_auroc.csv", index=False)
            