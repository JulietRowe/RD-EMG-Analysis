# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 13:29:46 2022

@author: julie
"""

from IPython import get_ipython
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

get_ipython().magic('reset -f')
get_ipython().magic('clear')

#Importing data
File = 'C:/Users/julie/Year 5 Kin/RD-EMG-Analysis/Gait Motion Capture.csv'
MC_data = pd.read_csv(File, header = 0)

File_emg = 'C:/Users/julie/Year 5 Kin/RD-EMG-Analysis/RD_Walk_EMG.csv'
emg = pd.read_csv(File_emg, header = 0)

#Resampling LASI from 100Hz to 2000Hz
x = MC_data['LLAN Y']
x_resampled = signal.resample(x, 102840)

plt.figure()
fig, ax = plt.subplots(1, figsize = (15,10))
ax.plot(x_resampled)

#Sampling frequency
sfreq = 2000

#Setting time column

emg['delta_t'] = 1/sfreq
emg['real_time'] = np.cumsum(emg['delta_t'])
emg = emg.set_index('real_time')
emg = emg.iloc[25010:98423]

#Determining side to analyze 
val = input("Please type right to analyze the right side OR left to analyze the left side: ")
   
if val == 'right':
    GM = 'R_GM'
    ST = 'R_ST'
    BF = 'R_BF'
elif val == 'left':
    GM = 'L_GM'
    ST = 'L_ST'
    BF = 'L_BF'

#Visualizing data
def custom_plot(suptitle, x,
                y1, y2, y3):
    """Subplots of EMG data"""
    fig, axs = plt.subplots(3, figsize =(15,10))
    fig.suptitle(suptitle)
    axs[0].plot(x, y1)
    axs[0].set_title('Gluteus maximus')

    axs[1].plot(x, y2)
    axs[1].set_title('Semitendinosis')

    axs[2].plot(x, y3)
    axs[2].set_title('Biceps femoris')

    for ax in axs.flat:
        ax.set(xlabel = 'Time(sec)', ylabel = 'EMG(uV)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    for ax in axs.flat:
        ax.label_outer()

#Visualizing data
custom_plot('Raw EMG', emg.index, emg[GM], emg[ST], emg[BF])

#Filtering data
from scipy import signal 

#Band Pass Filtering
def Band_Pass(Muscle):
    """Band Pass filter for EMG data"""
    High_fc = 30
    Low_fc = 300
    HighPass = High_fc/(sfreq/2)
    LowPass = Low_fc/(sfreq/2)
    b, a = signal.butter(4, [HighPass, LowPass], btype = 'bandpass')
    emg_filter = signal.filtfilt(b, a, Muscle, padlen = 0)
    return emg_filter

NewEMG = emg.apply(Band_Pass)

#Fullwave rectify
emg_rectified = abs(NewEMG)
custom_plot('Rectified and filtered EMG', emg_rectified.index, emg_rectified[GM], emg_rectified[ST], emg_rectified[BF])

#EMG envelope
def Envelope(Muscle):
    """Low Pass filter to create EMG envelope"""
    Low_fc = 5
    b_e, a_e = signal.butter(4, Low_fc/(sfreq/2), btype = 'lowpass')
    emg_filter = signal.filtfilt(b_e, a_e, Muscle, padlen = 0)
    return emg_filter

#Threshold Detection
EMG_envelope = emg_rectified.apply(Envelope)
custom_plot('Envelope EMG', EMG_envelope.index, EMG_envelope[GM], EMG_envelope[ST],EMG_envelope[BF])
EMG_envelope['sample'] = (np.linspace(1, len(EMG_envelope), num = len(EMG_envelope))).astype(int)
EMG_envelope = EMG_envelope.set_index('sample')

#Peak detection for sit to stand
from scipy.signal import find_peaks
x_resampled = pd.DataFrame(x_resampled)
x_resampled = x_resampled.rename(columns = {0: 'LLAN Y'})
x_resampled['LLAN Y'] = x_resampled['LLAN Y']*-1
x_resampled['LLAN Y'] = x_resampled.loc[27000:, 'LLAN Y'].replace(np.nan)
x_resampled['LLAN Y'] = x_resampled['LLAN Y'][:-4000].replace(np.nan)
plt.figure()
plt.plot(x_resampled)

peaks = find_peaks(x_resampled['LLAN Y'], height = -130, distance = 1500)
height = peaks[1]['peak_heights'] #list containing the height of the peaks
peak_pos = x_resampled.index[peaks[0]] #list of the peaks positions
    
fig = plt.figure()
ax = fig.subplots()
ax.plot(x_resampled['LLAN Y'])
ax.scatter(peak_pos, height, color = 'r', s = 15, marker = 'D')
plt.show()

#Phase averaging the gait cycle based on ST activation
array_ST = EMG_envelope[ST].to_numpy()
array_GM = EMG_envelope[GM].to_numpy()
array_BF = EMG_envelope[BF].to_numpy()
subarray_ST = []
subarray_GM = []
subarray_BF = []


for i in range(1, len(peak_pos)):
    subarray_ST.append(array_ST[peak_pos[i - 1]:peak_pos[i]])
    subarray_GM.append(array_GM[peak_pos[i - 1]:peak_pos[i]])
    subarray_BF.append(array_BF[peak_pos[i - 1]:peak_pos[i]])
    
plt.figure()
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (15,10))
fig.suptitle('sit to squat')

for i in range(len(subarray_ST)):
    k = list(range(0, len(subarray_GM[i])))
    GM_X = []
    for number in k:
        GM_X.append(number/len(subarray_GM[i]))
        
    l = list(range(0, len(subarray_ST[i])))
    ST_X = []
    for number in l:
        ST_X.append(number/len(subarray_ST[i]))
        
    m = list(range(0, len(subarray_BF[i])))
    BF_X = []
    for number in m:
        BF_X.append(number/len(subarray_BF[i]))
    
    ax1.plot(GM_X, subarray_GM[i])
    ax2.plot(ST_X, subarray_ST[i])
    ax3.plot(BF_X, subarray_BF[i])
