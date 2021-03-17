# -*- coding: utf-8 -*-
"""
Visualization of precision-recall curves from K-fold cross-validation and testing
Created on Thu Oct 11 11:25:10 2018

@author: ys587
"""
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_accu_all(data_path):
    data_list = glob.glob(data_path+'/Run*.txt')
    
    print(data_list)

    accu_list = []
    for dd in data_list:
        run_df = pd.read_csv(dd,sep='\t')
        run_df['recall'] = run_df['TP']/run_df['TotP']
        run_df['precision'] = run_df['TP.1']/(run_df['TP.1']+run_df['FP'])
        
        #accu = np.array(run_df[['Thre', 'precision','recall']])
        #accu_list.append(accu)
        accu_list.append(run_df[['Thre', 'precision','recall']])
    
    accu_all_runs = pd.concat(accu_list)
    
    return accu_all_runs


accu_all_runs_lenet_dropout = get_accu_all(r'/home/ys587/tensorflow3/tmp/x-validate-lenet_dropout_conv/__full_data/')    
accu_all_runs_birdnet_dropout = get_accu_all(r'/home/ys587/tensorflow3/tmp/x-validate-birdnet_dropout_conv/__full_data/')



#data_path = r'/home/ys587/tensorflow3/tmp/x-validate-lenet_dropout_conv/__full_data/'
#data_path = r'/home/ys587/tensorflow3/tmp/x-validate-birdnet_dropout_conv/__full_data/'
#accu_all_runs = np.vstack(accu_list)

import seaborn as sns
sns.set(style="darkgrid")

#fig, ax = plt.subplots()
#g = sns.jointplot(x="recall", y="precision", data=accu_all_runs, kind='scatter',
#                  xlim=(0, 1.0), ylim=(0, 1.0), color='k')
g1 = sns.relplot(x="recall", y="precision", data=accu_all_runs_lenet_dropout, color='k')
g1.axes[0,0].set_ylim(0,1.0)
g1.axes[0,0].set_xlim(0,1.0)
g1.set_titles("Lenet Dropout Conv-layers")

g2 = sns.relplot(x="recall", y="precision", data=accu_all_runs_birdnet_dropout, color='b')
g2.axes[0,0].set_ylim(0,1.0)
g2.axes[0,0].set_xlim(0,1.0)
g2.set_titles("Birdnet Dropout Conv-layers")