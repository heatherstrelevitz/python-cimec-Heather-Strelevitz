# -*- coding: utf-8 -*-
"""

@author: heather.strelevitz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
from skimage import filters

#import data
# LP: please always specify an input data directory, so that the code can be run on any machine
# regardless of where I've put the data!
from pathlib import Path
input_data_dir = Path("/Users/vigji/Downloads/session_0")
x_eye = np.load(input_data_dir / 'eye.xPos.npy')
y_eye = np.load(input_data_dir / 'eye.yPos.npy')
eye_time = np.load(input_data_dir / 'eye.times.npy')
run_speed = np.load(input_data_dir / 'running.speed.npy')
run_time = np.load(input_data_dir / 'running.times.npy')
pupil_size = np.load(input_data_dir / 'eye.size.npy')
neural_activity = np.load(input_data_dir / 'frame.neuralActivity.npy')
#immediately zscore neural signal so I don't forget later
neural_activity = stats.zscore(neural_activity)
neural_times = np.load(input_data_dir / 'frame.times.npy')
names_file = open(input_data_dir / 'neuron.ttype.txt')
names = names_file.readlines()
#create df for neurons
neurons_df = pd.DataFrame(dict(type=names))
neurons_df.index = [f"neuron_{i}" for i in range(len(names))]

# LP: be more generous with empty lines! Also, if you want to have nicely formatted code 
# for free, you can check out black (https://black.readthedocs.io/en/stable/)
#first we need to define saccades, which will be our events of interest.
def find_saccades(x_eye,y_eye,min_dist):
    """
Find sufficiently high peaks in the velocity trace to define saccadic eye movements.
    Parameters
    ----------
    x_eye : array of float
        Eye position in x throughout the session.
    y_eye : array of float
        Eye position in y throughout the session.
    mind_dist : int
        A minimum separation between consecutive saccades.

    """
    
    #find velocity
    diffX = np.diff(x_eye, 1, 0)  # LP: small, but keep variable names lowercase
    diffY = np.diff(y_eye, 1, 0)
    velocity = np.squeeze(np.sqrt(diffX**2 + diffY**2))
    #choose velocity threshold
    velocity_nonan = velocity[~np.isnan(velocity)]
    thresh = filters.threshold_otsu(velocity_nonan)
    #use the threshold to find peaks in velocity trace
    peaks = find_peaks(velocity, height = thresh)[0]
    #adding a minimum separation of min_dist to avoid redundant peaks
    saccade_index_diffs = np.diff(peaks)
    non_overlaps = np.where(saccade_index_diffs>min_dist)
    peaks = peaks[non_overlaps]
    #we have the peaks of velocity, which are probably in the center-ish of the saccades. So let's find the beginnings and ends
    #detect saccade onsets (check which preceding points fall below half threshold), in index distance from the peak
    onsets = np.zeros((np.size(peaks)))
    for index, value in enumerate(peaks):
        velocity_slice = velocity[value-min_dist:value]
        onsets[index] = min_dist - (np.where(velocity_slice<thresh/2))[0][-1]
    onsets = onsets.astype(int)  # LP: could have initialized as int specifying dtype
    #detect saccade offsets (check which following points fall below half threshold), in index distance from the peak
    offsets = np.zeros((np.size(peaks)))
    for index, value in enumerate(peaks):
        velocity_slice = velocity[value:value+min_dist]
        offsets[index] = (np.where(velocity_slice<thresh/2))[0][0]
    offsets = offsets.astype(int)  # LP: I would suggest use a single loop for both onsets and offsets

    sacc_indexes = peaks[:, None] + np.arange(-5, 5) #just for sanity check purposes
    sacc_velos = np.squeeze(velocity[sacc_indexes]) #just for sanity check purposes  # LP: don't leave it around in a function! either return or don't compute
    duration=offsets+onsets
    #turn into times
    peak_time = np.squeeze(eye_time[peaks])
    onset_time = np.squeeze(eye_time[peaks - onsets])
    offset_time = np.squeeze(eye_time[peaks + offsets])
    #calculate amplitudes via pairwise distance between each sample during a saccade
    amplitudes = []
    for p in np.arange(len(peaks)):
        x_during = x_eye[peaks[p]-onsets[p]:peaks[p]+offsets[p]]
        y_during = y_eye[peaks[p]-onsets[p]:peaks[p]+offsets[p]]
        dists = np.sqrt(((x_during - x_during.T)**2) + ((y_during - y_during.T)**2))
        amplitudes.append(np.max(dists))
    #separate eye mvmts into temporal and nasal
    netx = x_eye[peaks-onsets] - x_eye[peaks+offsets]
    nasal = np.where(netx<0)[0]
    temporal = np.where(netx>0)[0]
    direction_indicator = []
    for i in np.arange(len(peaks)):
        if i in nasal:
            direction_indicator.append('n')
        else:
            direction_indicator.append('t')
    #at this point I'll collect all these quantities into a dataframe,
    #where each row is a single saccade
    saccades_df = pd.DataFrame(dict(onset=onset_time,
                                    offset=offset_time,
                                    duration=duration,
                                    amplitude=amplitudes,
                                    direction=direction_indicator))#,velocities=sacc_velos,time_range=eta_times))
    saccades_df.index = [f"sacc_{i}" for i in range(len(onsets))]

    # LP: I feel there might be some redundancy here. Why eg nasal, temporal if you already have direction column?
    # In general, it is better not to return too many separate variables.
    return saccades_df, peaks, peak_time, onsets, offsets, velocity, nasal, temporal
#run the function
[saccades_df,peaks, peak_time, onsets, offsets, velocity, nasal, temporal] = find_saccades(x_eye,y_eye,15)
#find etas for eye movements
def find_etas(events,time,samples_before,samples_after):
    """
Using known events of interest, identify the consecutive indices and times before and after.
    Parameters
    ----------
    events : array of int
        Indices of events of interest.
    time : array of float
        Timestamps of each sample for the data type.
    samples_before : int
        Number of samples to include in ETA, prior to the event.
    samples_after : int
        Number of samples to include in ETA, after the event.

    Returns
    -------
    eta_indices: array of int, size events x samples_before+samples_after
        The indices before and after each event.
    eta_times: array of float, size events x samples_before+samples_after x 1
        The times before and after each event.

    """
    
    eta_indices = events[:, None] + np.arange(samples_before, samples_after)
    eta_times = time[eta_indices]
    
    return eta_indices, eta_times
   
[sacc_indices, sacc_times] = find_etas(peaks,eye_time,-5,10)
eta_x = np.squeeze(x_eye[sacc_indices])
#sampling rates for neural and eye recordings are different. This approximates the closest indices to equate time
n_peaks = np.zeros(np.size(peaks))
for index, time in enumerate(peak_time):
    n_peaks[index] = np.abs(neural_times - time).argmin()
n_peaks = n_peaks.astype(int)
#find etas for neural activity
[neuro_indices,neuro_times] = find_etas(n_peaks,neural_times,-3,5)
eta_neuro = neural_activity[neuro_indices]

#now that everything is set up, let's do some sample plots and analyses!
#overall behavior (helpful to know how much data we're missing)
fig, axs = plt.subplots(3, 1)
axs[0].plot(eye_time,pupil_size)
axs[0].set_title('Pupil size')
axs[1].plot(run_time, run_speed)
axs[1].set_title('Locomotion')
axs[2].plot(eye_time, x_eye)
axs[2].set_title('Eye movement in x')
plt.tight_layout()
#saccade identification
fig, axs = plt.subplots(2,1)
axs[0].plot(velocity)
axs[0].plot(peaks,velocity[peaks], 'x', color = 'red')
axs[0].set_title('Velocity with saccades')
axs[1].plot(velocity[sacc_indices].T, color = 'gray')
axs[1].plot(velocity[sacc_indices].mean(axis = 0), color = 'red')
axs[1].set_title('Velocity ETA')
plt.tight_layout()
#saccade etas
fig, axs = plt.subplots(3, 1)
#LP: why not a nice loop? :)
axs[0].imshow(eta_x.T)
axs[0].set_title('All saccades')
axs[1].plot(eta_x[nasal].T,  color = 'gray')
axs[1].set_title('Nasal saccades')
axs[1].plot(eta_x[nasal].mean(axis = 0), color = 'green')
axs[2].plot(eta_x[temporal].T, color = 'gray')
axs[2].set_title('Temporal saccades')
axs[2].plot(eta_x[temporal].mean(axis = 0), color = 'magenta')
plt.tight_layout()
#let's see what the overall neural eta looks like
plt.plot(neural_activity[neuro_indices].mean(axis = 0), color = 'gray')
#cool, some neurons seem to care about the event!
#split into excitatory and inhibitory neurons
ec_etas = neural_activity.T[neurons_df['type'].str.contains('EC')]
plt.plot(ec_etas.T[neuro_indices].mean(axis = 0), color = 'orange')
in_etas = neural_activity.T[neurons_df['type'].str.contains('IN')]
plt.plot(in_etas.T[neuro_indices].mean(axis = 0), color = 'blue')

#LP: absolutely necessary loop here! :D
#do this for every subtype of inhibitory neuron
pvalb_etas = neural_activity.T[neurons_df['type'].str.contains('Pvalb')]
plt.plot(pvalb_etas.T[neuro_indices].mean(axis = 0), color = 'red')
vip_etas = neural_activity.T[neurons_df['type'].str.contains('Vip')]
plt.plot(vip_etas.T[neuro_indices].mean(axis = 0), color = 'yellow')
lamp5_etas = neural_activity.T[neurons_df['type'].str.contains('Lamp5')]
plt.plot(lamp5_etas.T[neuro_indices].mean(axis = 0), color = 'green')
sncg_etas = neural_activity.T[neurons_df['type'].str.contains('Sncg')]
plt.plot(sncg_etas.T[neuro_indices].mean(axis = 0), color = 'blue')
sst_etas = neural_activity.T[neurons_df['type'].str.contains('Sst')]
plt.plot(sst_etas.T[neuro_indices].mean(axis = 0), color = 'violet')
#nice, we have a subtype of interest: sst are most activated by saccades!
