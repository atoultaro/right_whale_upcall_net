# -*- coding: utf-8 -*-
"""
Copy the files of accuracy performance numbers such as true positive, false 
positive and false negative to a new place, in order to share them easily 
without including the big fat trained model files.

Created on Mon Nov 12 14:55:11 2018

@author: ys587
"""
import os, glob
from shutil import copyfile

result_folder = r'/home/ys587/__ExptResult/__V4_Paper'
result_only_folder = r'/home/ys587/__ExptResult/__V4_Paper/__result_only_temp'
model_folder_list = sorted(glob.glob(os.path.join(result_folder+'/','cv*')))

for mm in model_folder_list:
    #mm1 = mm+'_result_only'
    mm1 = os.path.join(result_only_folder, os.path.split(mm)[-1])
    os.makedirs(mm1, exist_ok=True)
    
    # on model folder level: copy txt files
    file_list = sorted(glob.glob(os.path.join(mm,'*.txt')))
    for ff in file_list:
        copyfile(ff, os.path.join(mm1, os.path.basename(ff)))

    
    # __full_data folder
    os.makedirs(os.path.join(mm1, '__full_data'), exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(mm, '__full_data','*.txt')))
    for ff in file_list:
        copyfile(ff, os.path.join(mm1, '__full_data',os.path.basename(ff)))

    os.makedirs(os.path.join(mm1, '__full_data'), exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(mm, '__full_data','*.png')))
    for ff in file_list:
        copyfile(ff, os.path.join(mm1, '__full_data',os.path.basename(ff)))
    
    # copy stuff in each run, excluding the model hdf5 file
    for rr in range(10):
        file_list = sorted(glob.glob(os.path.join(mm, '__full_data','Run'+str(rr),'*.txt')))
        os.makedirs(os.path.join(mm1, '__full_data','Run'+str(rr)), exist_ok=True)
        for ff in file_list:
            copyfile(ff, os.path.join(mm1, '__full_data','Run'+str(rr),os.path.basename(ff)) )
        #copyfile(os.path.join(mm, '__full_data','Run'+str(rr),'Precision_Recall_Curve.png'), os.path.join(mm1, '__full_data','Run'+str(rr),'Precision_Recall_Curve.png') )


    # __split_data folder
    os.makedirs(os.path.join(mm1, '__split_data'), exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(mm, '__split_data','*.txt')))
    for ff in file_list:
        copyfile(ff, os.path.join(mm1, '__split_data',os.path.basename(ff)))

    # copy stuff in each run, excluding the model hdf5 file
    for rr in range(10):
        file_list = sorted(glob.glob(os.path.join(mm, '__split_data','Run'+str(rr),'*.txt')))
        os.makedirs(os.path.join(mm1, '__split_data','Run'+str(rr)), exist_ok=True)
        for ff in file_list:
            copyfile(ff, os.path.join(mm1, '__split_data','Run'+str(rr),os.path.basename(ff)) )
        #copyfile(os.path.join(mm, '__split_data','Run'+str(rr),'Precision_Recall_Curve.png'), os.path.join(mm1, '__split_data','Run'+str(rr),'Precision_Recall_Curve.png') )

# __full_data_large
result_folder = r'/home/ys587/__ExptResult/__V4_Paper/__full_data_large/'
result_only_folder = os.path.join(result_only_folder,r'__full_data_large')
model_folder_list = sorted(glob.glob(os.path.join(result_folder+'/','__full_data_large*')))

for mm in model_folder_list:
    #mm1 = mm+'_result_only'
    mm1 = os.path.join(result_only_folder, os.path.split(mm)[-1])
    os.makedirs(mm1, exist_ok=True)

    # __full_data folder
    file_list = sorted(glob.glob(os.path.join(mm,'*.txt')))
    for ff in file_list:
        copyfile(ff, os.path.join(mm1, os.path.basename(ff)))

    file_list = sorted(glob.glob(os.path.join(mm,'*.png')))
    for ff in file_list:
        copyfile(ff, os.path.join(mm1, os.path.basename(ff)))

    # copy stuff in each run, excluding the model hdf5 file
    for rr in range(10):
        file_list = sorted(glob.glob(os.path.join(mm, 'Run'+str(rr),'*.txt')))
        os.makedirs(os.path.join(mm1, 'Run'+str(rr)), exist_ok=True)
        for ff in file_list:
            copyfile(ff, os.path.join(mm1, 'Run'+str(rr),os.path.basename(ff)) )















        
