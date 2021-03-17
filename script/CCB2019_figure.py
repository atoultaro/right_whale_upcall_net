#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
draw figure for upcall spectrogram with respect to score

Created on 7/10/19
@author: atoultaro
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import random

import soundfile as sf

from upcall.run_detector_dsp import make_sound_stream

# data
sound_path = r"/mnt/drive_W/projects/2018_ORStateU_CCB_85941/Dep01/AIFFs_UTCz"
seltab_path = r"/mnt/drive_W/projects/2018_ORStateU_CCB_85941/Dep01/detection_results/__seltab_test_drive"
sound_folder = r"85941_CCB27_20190224"
seltab_name = r"85941_CCB27_SelTab_20190224.txt"
sound_save_path = r"/home/ys587/__Data/__CCB/__upcall_clips"

num_whale_per_score = 40
num_score = 5

N_read = 4000 # 2 sec of 2,000 sampling rate

# selection tables
sound_files = os.path.join(sound_path, sound_folder)
# read into Pandas dataframe
seltab = os.path.join(seltab_path, seltab_name)
seltab_df = pd.read_csv(seltab, sep='\t')

# select events that satisfy the requirement of scores
df_list = []
for ss in np.arange(0.95, 0.55-0.1, -0.1):
    seltab_df_score = seltab_df[(seltab_df["Score"] > ss) & (seltab_df["Score"] <= ss+0.1)]
    chosen_ones = random.sample(range(len(seltab_df_score)), num_whale_per_score)
    df_list.append(seltab_df_score.iloc[chosen_ones][['Begin Time (s)', 'Channel', 'Score']])

# download the sound files
sample_stream = make_sound_stream(sound_files)
timestamp_0 = sample_stream.get_current_timesamp()
count = 0
for dd in df_list:
    for index, event in dd.iterrows():
        # print(event)
        # print(pd.Timestamp(event["Begin Time (s)"], unit='s'))
        print(timestamp_0 + pd.DateOffset(seconds=event["Begin Time (s)"]))
        sample_stream.set_time(timestamp_0 + pd.DateOffset(seconds=event["Begin Time (s)"]))
        # sample_stream.set_channel(event["Channel"]-1)
        samps_all_channels = sample_stream.read(N=N_read, previousN=0)[0]
        samps = samps_all_channels[:, int(event["Channel"]-1)]
        sf.write(os.path.join(sound_save_path, "upcall_"+str(count).zfill(4)+"_"+str(int(np.floor(event["Score"]*100.0)))+".wav"), samps, 2000)
        count += 1


# plot
fig, ax_arr = plt.subplots(num_score, num_whale_per_score, figsize=[16., 8.])

# plt.axis('off')
