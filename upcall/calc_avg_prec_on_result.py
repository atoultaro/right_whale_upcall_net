# -*- coding: utf-8 -*-
"""
calculate mAP, mean average precision

Modified from 
https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py

Created on Wed Jan  9 18:36:39 2019

@author: ys587
"""
from __future__ import print_function
import os, sys
ROOT_DIR = os.path.abspath("../../upcall-basic-net")
sys.path.append(ROOT_DIR)
import glob
import pandas as pd
import numpy as np
from upcall.accuracy_measure import map_build_day_file, get_date_str
from upcall.config import Config
from collections import Counter
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_DIR = r'/home/ys587/__Data' # Data
TRAIN_RESULT_FOLDER = r'/home/ys587/__ExptResult' # Result


def timestamp_to_date(timestamp):
    return datetime.datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S.%f').date() # date object


def ElevenPointInterpolatedAP(rec, prec):
    # def CalculateAveragePrecision2(rec, prec):
    mrec = []
    # mrec.append(0)
    [mrec.append(e) for e in rec]
    # mrec.append(1)
    mpre = []
    # mpre.append(0)
    [mpre.append(e) for e in prec]
    # mpre.append(0)
    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(mrec[:-1] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / 11
    # Generating values for the plot
    rvals = []
    rvals.append(recallValid[0])
    [rvals.append(e) for e in recallValid]
    rvals.append(0)
    pvals = []
    pvals.append(0)
    [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recallValues = [i[0] for i in cc]
    rhoInterp = [i[1] for i in cc]
    
    return [ap, rhoInterp, recallValues, None]


def CalculateAveragePrecision(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


def calc_avg_prec(truth_folder, seltab_detect_path, config):
    map_day_file = map_build_day_file(truth_folder)
    day_list = sorted(glob.glob(os.path.join(seltab_detect_path,'*.txt'))) # find selection table for all days

    # turn off pd "cope of a slice" warning
    pd.set_option('mode.chained_assignment', None)

    df_detect_list = []
    df_truth_list = []
    for dd in day_list:
        basename = os.path.basename(dd)
        print('Current day: '+basename)
        #print('Corresponding truth file: ', end='')
        date_str = get_date_str(basename)
        truth_file = map_day_file[date_str]
        #print(truth_file)
        
        # (a) dd, the detection file versus (b) truth_file
        df_detect0 = pd.read_table(dd)
        df_truth0 = pd.read_table(truth_file)
        
        # sort by the column "Begin Time (s)"
        df_detect0 = df_detect0.sort_values(r'Begin Time (s)')
        df_detect0 = df_detect0.reset_index(drop=True)
        df_detect0[r'Center Time (s)'] = 0.5*df_detect0[r'Begin Time (s)'] + 0.5*df_detect0[r'End Time (s)']
        df_detect0[r'date'] = date_str
        df_detect_list.append(df_detect0)
            
        df_truth0 = df_truth0.sort_values(r'Begin Time (s)')
        df_truth0 = df_truth0.reset_index(drop=True)
        df_truth0[r'Center Time (s)'] = 0.5*df_truth0[r'Begin Time (s)'] + 0.5*df_truth0[r'End Time (s)']
        df_truth0[r'date'] = date_str
        df_truth_list.append(df_truth0)

    df_detect = pd.concat(df_detect_list)
    del df_detect_list
    df_truth = pd.concat(df_truth_list)
    del df_truth_list
    #df_truth = df_truth.reset_index(drop=True)

    #dects = []
    #[dects.append(row) for index, row in df_detect.iterrows()]
    gts = []
    [gts.append(row) for index, row in df_truth.iterrows()]
            
    # dects => df_detect; gts => df_truth
    npos = df_truth.shape[0]
    # sort detections by decreasing confidence
    df_detect = df_detect.sort_values(by=['Score'], ascending=False) # sort by score
    TP = np.zeros(df_detect.shape[0])
    FP = np.zeros(df_detect.shape[0])
    
    det0 = Counter([cc[r'date'] for cc in gts])
    det = {}
    for key, val in det0.items():
        det[key] = np.zeros(val)
        
    for dd in range(df_detect.shape[0]):
    # for dd in range(500):
        df_truth_same_date = df_truth.loc[(df_truth['date']==df_detect.iloc[dd]['date']) & ( np.abs(df_truth[r'Center Time (s)'] - df_detect.iloc[dd][r'Center Time (s)']) < config.SEP_THRE)]
        df_truth_same_date[r'delta time'] = np.abs(df_truth_same_date[r'Center Time (s)'] - df_detect.iloc[dd][r'Center Time (s)'])
        if dd % 200 == 0:
            print('#', end='', flush=True)
        
        hit_num = df_truth_same_date.shape[0]
        if hit_num > 0:
            if hit_num == 1:
                ind_hit = df_truth_same_date.index[0]
            else: # >= 2
                ind_hit = df_truth_same_date[r'delta time'].idxmin(axis=0)
                
            if det[df_detect.iloc[dd]['date']][ind_hit] == 0:
                TP[dd] = 1
                det[df_detect.iloc[dd]['date']][ind_hit] = 1
            else:
                FP[dd] = 1
        else:
            FP[dd] = 1
    print('')

    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    
    [ap_11pt, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)
    print('AP_11pt is: '+str(ap_11pt))
    [ap_avg, mpre, mrec, _] = CalculateAveragePrecision(rec, prec)
    print('AP_avg is: ' + str(ap_avg))
    # plt.plot(mrec, mpre)
    
    return ap_11pt, ap_avg, mpre, mrec


def calc_ap_over_runs(seltab_detect_path, truth_folder, config):
    ap_list = []
    ap11_list = []
    pre_rec_list = []
    for rr in range(config.NUM_RUNS):
    #for rr in range(1):
        print('Run '+str(rr))
        seltab_detect_run_path = os.path.join(seltab_detect_path, 'Run'+str(rr))
        #print('seltab_detect_run_path:')
        #print(seltab_detect_run_path)
        
        ap_11, ap_avg, mpre, mrec = calc_avg_prec(truth_folder, seltab_detect_run_path, config)
        ap11_list.append(ap_11)
        ap_list.append(ap_avg)
        pre_rec_list.append([mpre, mrec])
        
        # draw a fig
        fig, ax = plt.subplots()
        ax.plot(mrec, mpre, '-b')
        
        ax.set_xlim([0, 1.0])
        ax.set_ylim([0, 1.0])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        #ax.legend(loc='lower left', shadow=True, fontsize='x-large')
        plt.grid()
    
        plt.savefig( os.path.join(seltab_detect_path, 'prec_rec_Run'+str(rr)+'.png'),dpi=300) # 300 dpi
    
    # save ap values in a file
    ap_array = np.array(ap_list)
    ap11_array = np.array(ap11_list)
    np.savetxt(os.path.join(seltab_detect_path, 'avg_prec.txt'), ap_array, fmt='%.6f', delimiter='/t')
    np.savetxt(os.path.join(seltab_detect_path, 'avg_prec_11pt.txt'), ap11_array, fmt='%.6f', delimiter='/t')

    # create a datafram to hold precision-recall cuver of all runs; for the seaborn
    pd_prc_list = []
    for ii in range(len(pre_rec_list)):
        #pre_rec_array_list.append(np.array(pre_rec_list[ii]))
        pd_prc = pd.DataFrame(np.array(pre_rec_list[ii]).T, columns=['mpre','mrec'])
        pd_prc['run'] = ii
        pd_prc_list.append(pd_prc)
    pd_prc_all_runs = pd.concat(pd_prc_list)
    pd_prc_all_runs.to_csv(os.path.join(seltab_detect_path, 'pre_rec.txt'), sep='\t')    
    

if __name__ == "__main__":
    config = Config()
    config.NUM_RUNS = 10

    # DCL13
    truth_folder = os.path.join(DATA_DIR, 'DCL_St_Andrew/Sound_3_days_seltab') # DCL13 test dataset

    func_folder = '__full_data' # could be __full_data_large as well
    # seltab_detect_path = r'/home/ys587/__ExptResult/__V4_Paper'
    seltab_detect_path = r'/home/ys587/__ExptResult/__V4_Paper/__result_only_20190114/__negtive_mining'
    model_path = glob.glob(seltab_detect_path+'/cv_*')
    model_path.sort()
    
    for mm in model_path:
        print(os.path.basename(mm))
        seltab_detect_path = os.path.join(mm, func_folder)
        calc_ap_over_runs(seltab_detect_path, truth_folder, config)
        
    # # large_test_dataset
    # # /home/ys587/__Data/__large_test_dataset/__truth_seltab/
    # truth_folder = os.path.join(DATA_DIR, '__large_test_dataset/__truth_seltab')
    # seltab_detect_path = r'/home/ys587/__ExptResult/__V4_Paper/__full_data_large/'
    # model_path = glob.glob(seltab_detect_path+'/__full*')
    # model_path.sort()
    #
    # for mm in model_path:
    #     print(os.path.basename(mm))
    #     calc_ap_over_runs(mm, truth_folder, config)
    
    
    
    
    

    

    

    if False:
        # Calculate mAP on a single model
        seltab_detect_path = r'/home/ys587/__ExptResult/cv_lenet_dropout_conv_train_Data_6_augment'
        func_folder = '__full_data' # could be __full_data_large as well
    
        seltab_detect_path = os.path.join(seltab_detect_path, func_folder)
        calc_ap_over_runs(seltab_detect_path, config)
        
        # Playground
        import pandas as pd
        pd_prc_all_runs = pd.read_csv('/home/ys587/__ExptResult/cv_lenet_dropout_conv_train_Data_6_augment/__full_data/pre_rec.txt',sep='\t')
        
        #import seaborn as sns
        sns.set()
        #sns.lineplot(x='mrec',y='mpre', data=pd_prc_all_runs)
        
        sns.set(style='darkgrid')
        sns.relplot(x='mrec',y='mpre', hue='run', data=pd_prc_all_runs)
        #np.savetxt(os.path.join(seltab_detect_path, func_folder, 'pre_rec.txt'), ap_array, fmt='%.6f', delimiter='/t')

