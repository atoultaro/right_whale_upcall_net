# -*- coding: utf-8 -*-
"""
measuring the detection accuracy 
by comparing the detection selection table 
against the truth selection table. The accuracy measurement consists of
number of tru posisitve, number of false negative and number of false 
positive

Created on Wed Mar 14 10:44:10 2018

@author: ys587
"""
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import re
import glob
import matplotlib.pyplot as plt
import sys
import datetime
from collections import Counter


def get_date_str(the_string):
    """get the date string in the filename
    args:
        the_string: the filename of YYYYMMDD format
    return:
        the data (string)
    """
    m = re.search("_(\d{8})", the_string) # find the pattern _YYYYMMDD
    return m.groups()[0]

def dataframe2seltab(DF_input, seltab_path):
    """write out detection results in dataframe to seletion table file
    """
    with open(seltab_path,'w') as f:
        f.write('Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow \
        Freq (Hz)\tHigh Freq (Hz)\tScore\n')
        for index, row in DF_input.iterrows():
            f.write(str(row[u'Selection'])+'\t'+'Spectrogram'+'\t'+
            str(row['Channel'])+'\t'+ str.format("{0:=.4f}",
                row[u'Begin Time (s)']) + '\t' + str.format("{0:<.4f}",
                row[u'End Time (s)'])+ '\t50.0\t800.0\t'+str.format("{0:<.4f}",
                row[u'Score'])+'\n')
    return
    
def calc_TP_FP_FN(detect_file, truth_file, FP_output_path, config):
    """
    count the number of TP, FN & FP
    args:
        detect_file: path to the detection selection table file
        truth_file: path to the truth selection table file
        FP_output_path: the output path of FP & TP selection tables
        config: configuration
    return:
        df_detect_perform: dataframe of detection performance, i.e., TP, 
        FN & FP
    """
    # read the detection result
    try:
        df_detect = pd.read_table(detect_file)
        
        # Check if read properly imported the file
        if type(df_detect['Selection'].iloc[0]) == str:
            
            # then the dataframe was improperly improted
            df_detect = pd.read_table(detect_file, index_col= False)        
    except OSError:
        print('Detecton selection table is not found.')
        sys.exit(1) # terminate the program
    # Read the truth
    try:
        df_truth = pd.read_table(truth_file)
        # Check if read properly imported the file
        if type(df_truth['Selection'].iloc[0]) == str:
            
            # then the dataframe was improperly improted
            df_truth = pd.read_table(truth_file, index_col= False)        
    except OSError:
        print('Truth selection table is not found.')
        sys.exit(1)

    # sort by the column "Begin Time (s)"
    df_detect = df_detect.sort_values(r'Begin Time (s)')
    df_detect = df_detect.reset_index(drop=True)
    try:
        df_detect[r'Center Time (s)'] = 0.5*df_detect[r'Begin Time (s)'] + 0.5*df_detect[r'End Time (s)']
    except:
        print()
        
    df_truth = df_truth.sort_values(r'Begin Time (s)')
    df_truth = df_truth.reset_index(drop=True)
    try:
        df_truth[r'Center Time (s)'] = 0.5*df_truth[r'Begin Time (s)'] + 0.5*df_truth[r'End Time (s)']
    except:
        print()
            
    
    print('Collect TP, FP & FN......')
    # check every sound events in truth label to see if it is detected.
    # an array for truth: TP & FN
    ResultTP_FN = np.zeros(len(df_truth)) # 0: FN; 1: TP
    # an array for detected: TP & FP
    ResultTP_FP = np.zeros(len(df_detect)) # 0: FP; others: Score
    
    #NumThre = int((1-config.START_THRE)/config.STEP_THRE)
    NumThre = round((1-config.START_THRE)/config.STEP_THRE)
    detect_perform = np.zeros([NumThre, 6])
    for tt in range(NumThre):
        detect_perform[tt, 0] = tt*config.STEP_THRE + config.START_THRE
        
    for ii in range(detect_perform.shape[0]):
        detect_perform[ii, 1] = len(df_truth)*1.0 # how many positive events are
    
    for IndTru in range(len(df_truth)): # examine each event in the truth table to determine if it's TP or FN
        if IndTru % 100 == 0:
            print('Truth sound event: '+str(IndTru))
        
        RowTru = df_truth.iloc[IndTru]
        TargetDetect = df_detect[(df_detect[r'Channel']==RowTru[r'Channel']) & (np.abs(df_detect[r'Center Time (s)'] - RowTru['Center Time (s)']) < config.SEP_THRE)]
        #IndDetect = np.where((df_detect[r'Channel']==RowTru[r'Channel']) & (np.abs(df_detect[r'Center Time (s)'] - RowTru['Center Time (s)']) < config.SEP_THRE)>0)
        if len(TargetDetect) == 0: # FN
            #print('')
            for ScoreInd in range(0, detect_perform.shape[0]): # 15 since starting at 0.25 with step 0.05
                detect_perform[ScoreInd, 3] += 1.0 # FN
        else: # len(IndDetect)>=1; TP or FP if larger than score threshold, FN if smaller.
            if(len(TargetDetect)>=2):
                IndClosest = np.argmin(np.abs(TargetDetect[r'Center Time (s)']-RowTru['Center Time (s)']).values) # closer one wins
                TargetDetect = TargetDetect.iloc[[IndClosest]] # make sure the output is Dataframe instead of Series!
                #.to_frame()

            ResultTP_FP[int(TargetDetect.index[0])] = 1.0
            
            for ScoreInd in range(0, detect_perform.shape[0]): # 0.25 to 1.0 with 0.05 step            
                #if( float(TargetDetect['Score']) >= (ScoreInd*config.STEP_THRE + config.START_THRE) ):
                if( float(TargetDetect['Score']) > (ScoreInd*config.STEP_THRE + config.START_THRE) ):
                    detect_perform[ScoreInd, 2] += 1.0 # TP
                    ResultTP_FN[IndTru] = 1.0
                    
                else:
                    detect_perform[ScoreInd, 3] += 1.0 # FN
    df_detect['TP_FP']=pd.Series(ResultTP_FP) # add ResultTP_FP as a new column to the dataframe
    
    # outputting potential FPs to the selection tables
    df_detectFP = df_detect[df_detect['TP_FP']==0] # output all potential FP; assuming score threshold is the smallest 0.25
    # os.path.splitext(os.path.basename(detect_file))[0]
    dataframe2seltab(df_detectFP, os.path.join(FP_output_path, os.path.splitext(os.path.basename(detect_file))[0]+'_FP.txt'))
    
    # outputting potential TPs
    df_detectTP = df_detect[df_detect['TP_FP']==1]
    dataframe2seltab(df_detectTP, os.path.join(FP_output_path, os.path.splitext(os.path.basename(detect_file))[0]+'_TP.txt'))
    
    # calculate TP & FP                    
    for ScoreInd in range(0, NumThre):
        detect_perform[ScoreInd, 4] = (df_detect[df_detect['TP_FP']==1.0]['Score'] >=(ScoreInd*config.STEP_THRE + config.START_THRE)).sum() # TP
        detect_perform[ScoreInd, 5] = (df_detect[df_detect['TP_FP']==0.0]['Score'] >=(ScoreInd*config.STEP_THRE + config.START_THRE)).sum() # FP
    df_detect_perform = pd.DataFrame(detect_perform, columns=['Thre', 'TotP','TP', 'FN', 'TP','FP'])
    
    return df_detect_perform, df_detectFP

def calc_precision_recall(accu_tab):
    """ 
    Calculate precision and recall, given number of TP, FP and FN
    args:
        accu_tab: accuracy table (thresholds, TP, FN & FP)
    returns:
        recall
        precision
    """
    Recall = accu_tab[:,2]/accu_tab[:,1]
    Precision = accu_tab[:,4]/(accu_tab[:,4]+accu_tab[:,5])
    return Recall, Precision
    
def plot_precision_recall(accu_tab, expt_label, plot_path, fig_filename='Precision_Recall_Curve.png'):
    """
    Plot precision-recall curve, given accu_tab
    args:
        accu_tab: accuracy table (thresholds, TP, FN & FP)
        expt_label: text label for the plot
        plot_path: path of output plot 
        fig_filename: filename of the output figure
    """
    r1, p1 = calc_precision_recall(accu_tab)
    
    fig, ax = plt.subplots()
    ax.plot(r1, p1, '-ob', label=expt_label)
    
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left', shadow=True, fontsize='x-large')
    plt.grid()
    plt.savefig(plot_path+'/'+fig_filename, dpi=300) # 300 dpi


def accu_days(map_day_file, seltab_detect_path, accu_result_path, expt_label, config):
    """
    Measure accuracy performance numbers such true positive (TP), false 
    positive (FP) & false negative (FN) for a set of days
    args:
        map_day_file: map between the name of sound day folder and its path
        seltab_detect_path: folder that has multiple day selection tables
        accu_result_path: path of the folder having output results
        config: configuration
    return:
        TP, FN & FP numbers for multiple targeted days        
    """
    day_list = sorted(glob.glob(os.path.join(seltab_detect_path,'*.txt'))) # find selection table for all days
    
    accu_result_total_array = np.zeros([round((1.-config.START_THRE)/config.STEP_THRE), 5])
    for dd in day_list:
        basename = os.path.basename(dd)
        print('\nCurrent day: '+basename)
        print('Corresponding truth file: ', end='')
        truth_file = map_day_file[get_date_str(basename)]
        print(truth_file)
        
        # Calculate AP
        #AP, PreArr, RecArr = AvgPrecision(dd, truth_file)

        # Calculate TP, FN & FP, given steps of score threshold
        accu_result, _ = calc_TP_FP_FN(dd, truth_file, accu_result_path, config) # TP, FP, TP2, FN
        # output each seltab
        accu_result_total_array += (accu_result.values)[:,1:]
        accu_result.to_csv(os.path.join(accu_result_path, basename), index=False, sep="\t")
    # combine TP, FP & FN from days and write it out
    accu_result_total_array0 = np.hstack(( np.reshape(accu_result.values[:,0],[-1,1]), accu_result_total_array))
    accu_result_tot = pd.DataFrame(accu_result_total_array0, columns=['Thre','TotP','TP', 'FN', 'TP','FP'])
    accu_result_tot.to_csv(os.path.join(accu_result_path, expt_label+'_TP_FN_FP.txt'), index=False, sep="\t")  
    
    #return accu_result_tot.values
    return accu_result_tot
    
def map_build_day_file(truth_seltab_path):
    """
    Build the map from the date string (YYYYMMDD) to the corresponding 
    file path
    args:
        truth_seltab_path: path to the truth selection tables
    return:
        the built map
    """
    truth_seltab_list = sorted(glob.glob(os.path.join(truth_seltab_path,'*.txt')))
    map_day_file = {}
    for dd in truth_seltab_list:
        map_day_file[get_date_str(os.path.basename(dd))] = dd
    return map_day_file


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

def calc_avg_prec(map_day_file, seltab_detect_path, config):
    #map_day_file = map_build_day_file(truth_folder)
    day_list = sorted(glob.glob(os.path.join(seltab_detect_path,'*.txt'))) # find selection table for all days
    
    df_detect_list = []
    df_truth_list = []
    for dd in day_list:
        basename = os.path.basename(dd)
        #print('\nCurrent day: '+basename)
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
    #for dd in range(100):
        df_truth_same_date = df_truth.loc[(df_truth['date']==df_detect.iloc[dd]['date']) & ( np.abs(df_truth[r'Center Time (s)'] - df_detect.iloc[dd][r'Center Time (s)']) < config.SEP_THRE)]
        df_truth_same_date[r'delta time'] = np.abs(df_truth_same_date[r'Center Time (s)'] - df_detect.iloc[dd][r'Center Time (s)'])
        #if dd % 1000 == 0:
        #    print(dd)
        
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

    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    
    [ap_11pt, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)
    print('AP is: '+str(ap_11pt))
    [ap_average, mpre, mrec, _] = CalculateAveragePrecision(rec, prec)
    
    plt.plot(mrec, mpre)
    
    return ap_11pt, mpre, mrec

#def calc_ap_over_runs(seltab_detect_path, truth_folder, config):
def calc_ap_over_runs(seltab_detect_path, map_day_file, config):
    #map_day_file = map_build_day_file(truth_folder)
    
    ap_list = []
    pre_rec_list = []
    for rr in range(config.NUM_RUNS):
    #for rr in range(1):
        print('Run '+str(rr))
        seltab_detect_run_path = os.path.join(seltab_detect_path, 'Run'+str(rr))
        #print('seltab_detect_run_path:')
        #print(seltab_detect_run_path)
        
        ap, mpre, mrec = calc_avg_prec(map_day_file, seltab_detect_run_path, config)
        ap_list.append(ap)
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
    np.savetxt(os.path.join(seltab_detect_path, 'avg_prec.txt'), ap_array, fmt='%.6f', delimiter='/t')
    

    # create a datafram to hold precision-recall cuver of all runs; for the seaborn
    pd_prc_list = []
    for ii in range(len(pre_rec_list)):
        #pre_rec_array_list.append(np.array(pre_rec_list[ii]))
        pd_prc = pd.DataFrame(np.array(pre_rec_list[ii]).T, columns=['mpre','mrec'])
        pd_prc['run'] = ii
        pd_prc_list.append(pd_prc)
    pd_prc_all_runs = pd.concat(pd_prc_list)
    pd_prc_all_runs.to_csv(os.path.join(seltab_detect_path, 'pre_rec.txt'), sep='\t')    
    
#def AvgPrecision(detect_file, truth_file, config.SEP_THRE = 1.0):
#    """
#    single-day average precision/AP
#    """
        
    
#    # read the detection result
#    try:
#        df_detect = pd.read_table(detect_file)
#    except NameError:
#        print('invalid results file type, must use .txt or .csv')
#        return
#    
#    # Read the truth
#    try:
#        df_truth = pd.read_table(truth_file)
#    except NameError:
#        print('invalid truth file type, must use .txt or .csv')
#        return
#    
#    df_truth = df_truth.sort_values(r'Begin Time (s)')
#    df_truth = df_truth.reset_index(drop=True)
#    df_truth[r'Center Time (s)'] = 0.5*df_truth[r'Begin Time (s)'] + 0.5*df_truth[r'End Time (s)']
#    NumPos = len(df_truth)
#
#    # sort df_detect by score
#    df_detect = df_detect.sort_values(r'Score')
#    df_detect = df_detect.reset_index(drop=True)
#    df_detect[r'Center Time (s)'] = 0.5*df_detect[r'Begin Time (s)'] + 0.5*df_detect[r'End Time (s)']
#    
#    # calculate TP by comparing over all events in truth table
#    for IndDet in range(len(df_detect)):
#        if IndDet % 100 == 0:
#            print('Detected sound event: '+str(IndDet))
#        RowDet = df_detect.iloc[IndDet]
#        TargetDetect = df_detect[(df_truth[r'Channel']==RowDet[r'Channel']) & (np.abs(df_truth[r'Center Time (s)'] - RowDet['Center Time (s)']) < config.SEP_THRE)]
#        print()
#
#    return AP, PreArr, RecArr


#    print('Collect TP, FP & FN......')
#    # check every sound events in truth label to see if it is detected.
#    # an array for truth: TP & FN
#    ResultTP_FN = np.zeros(len(df_truth)) # 0: FN; 1: TP
#    # an array for detected: TP & FP
#    ResultTP_FP = np.zeros(len(df_detect)) # 0: FP; others: Score
    
#    for IndTru in range(len(df_truth)): # examine each event in the truth table to determine if it's TP or FN
#    #for IndTru in range(28, 30):
#        if IndTru % 100 == 0:
#            print('Truth sound event: '+str(IndTru))
#        
#        RowTru = df_truth.iloc[IndTru]
#        TargetDetect = df_detect[(df_detect[r'Channel']==RowTru[r'Channel']) & (np.abs(df_detect[r'Center Time (s)'] - RowTru['Center Time (s)']) < config.SEP_THRE)]


#        if len(TargetDetect) == 0: # FN
#            #print('')
#            for ScoreInd in range(0, DetectPerform.shape[0]): # 15 since starting at 0.25 with step 0.05
#                DetectPerform[ScoreInd, 3] += 1 # FN
#        else: # len(IndDetect)>=1; TP or FP if larger than score threshold, FN if smaller.
#            if(len(TargetDetect)>=2):
#                IndClosest = np.argmin(np.abs(TargetDetect[r'Center Time (s)']-RowTru['Center Time (s)']).values) # closer one wins
#                TargetDetect = TargetDetect.iloc[[IndClosest]] # make sure the output is Dataframe instead of Series!
#                #.to_frame()
#
#            #ResultTP_FP[int(TargetDetect.index[0])] = float(TargetDetect['Score']) # TP if the score is larger than threshold!
#            ResultTP_FP[int(TargetDetect.index[0])] = 1.0