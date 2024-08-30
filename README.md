# python-cimec-Heather-Strelevitz
This is a preliminary analysis of a dataset containing behavioral and neural data. Specifically, this code is written to look for saccades within eye tracking data and use those saccades to time-lock two-photon recordings of neurons with known cell types. However, it can be easily adapted to matching any behavioral event of interest to any neural recording. 
* Your data should be np.arrays of values for measures taken over a session, given in samples, each with a corresponding np.array of the time stamp for every sample.
* Some common libraries are imported at the beginning of the code. Otherwise this script is self-contained and should easily run without further installations.
