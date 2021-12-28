# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:33:41 2021

@author: julie
"""


from IPython import get_ipython
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().magic('reset -f')
get_ipython().magic('clear')

# Import Data
emg = pd.read_csv('C:/Users/julie/Year 5 Kin/RD_Sit_Stand_EMG.csv', header = 0)

#Sampling frequency
sfreq = 1000

#Setting time column
emg['delta_t'] = 1/sfreq
emg['real_time'] = np.cumsum(emg['delta_t'])
emg = emg.set_index('real_time')
# emg = emg.iloc[25010:98423]

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
custom_plot('Raw EMG', emg.index, emg['R_GM'], emg['R_ST'], emg['R_BF'])

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
custom_plot('Rectified and filtered EMG', emg_rectified.index, emg_rectified['R_GM'], emg_rectified['R_ST'], emg_rectified['R_BF'])

#EMG envelope
def Envelope(Muscle):
    """Low Pass filter to create EMG envelope"""
    Low_fc = 5
    b_e, a_e = signal.butter(4, Low_fc/(sfreq/2), btype = 'lowpass')
    emg_filter = signal.filtfilt(b_e, a_e, Muscle, padlen = 0)
    return emg_filter

#Threshold Detection
EMG_envelope = emg_rectified.apply(Envelope)
custom_plot('Envelope EMG', EMG_envelope.index, EMG_envelope['R_GM'], EMG_envelope['R_ST'],EMG_envelope['R_BF'])

EMG_envelope['sample'] = (np.linspace(1, len(EMG_envelope), num = len(EMG_envelope))).astype(int)
EMG_envelope = EMG_envelope.set_index('sample')

from scipy.signal import find_peaks

#Finding the peaks of the rest sections
peaks = find_peaks(EMG_envelope['R_BF'], height = 0.000016, distance = 8000)
height = peaks[1]['peak_heights'] #list containing the height of the peaks
peak_pos = EMG_envelope.index[peaks[0]] #list of the peaks positions

fig = plt.figure()
ax = fig.subplots()
ax.plot(EMG_envelope['R_BF'])
ax.scatter(peak_pos, height, color = 'r', s = 15, marker = 'D')
plt.show()

#Phase averaging the gait cycle based on ST activation
array_ST = EMG_envelope["R_ST"].to_numpy()
array_GM = EMG_envelope["R_GM"].to_numpy()
array_BF = EMG_envelope["R_BF"].to_numpy()
subarray_ST = []
subarray_GM = []
subarray_BF = []

for i in range(0, len(peak_pos)):
    subarray_ST.append(array_ST[peak_pos[i]-1000:peak_pos[i]+7000])
    subarray_GM.append(array_GM[peak_pos[i]-1000:peak_pos[i]+7000])
    subarray_BF.append(array_BF[peak_pos[i]-1000:peak_pos[i]+7000])
    #R is -1000, +7000; L is  -6000, +4000
    #R is based on R_BF, whereas L is L_ST

#Setting time column
EMG_envelope['delta_t'] = 1/sfreq
EMG_envelope['real_time'] = np.cumsum(EMG_envelope['delta_t'])

#Plotting muscle activation
plt.figure()
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (15,10))
fig.suptitle('Stacked Sit to Stand')

for i in range(len(subarray_ST)):
    ax1.plot(subarray_GM[i])
    ax2.plot(subarray_ST[i])
    ax3.plot(subarray_BF[i])

def simpleaxis(ax, title, xlabel = None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title)
    ax.title.set_size(12)
    ax.set_xlabel(xlabel, fontsize = 12)
    ax.set_ylabel('EMG (uV)', fontsize = 12)
    ax.set_ylim(0, 0.00008)
    #R side 0.000035; all was 0.00008; all L is 0.0002
    ax.set_xticklabels([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    

simpleaxis(ax1, 'R Gluteus maximus')
simpleaxis(ax2, 'R Semitendinosus')
simpleaxis(ax3, 'R Biceps femoris', xlabel = 'Time (s)')
    
plt.show()
