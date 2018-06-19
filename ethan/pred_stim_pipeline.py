import numpy as np
import sys
# sys.path.insert(0, '/home1/esolo/notebooks/Network Synchrony/')
from ethan.MNE_pipeline_refactored import exclude_bad

def get_bp_tal_struct(sub, montage=0):

    from ptsa.data.readers import TalReader

    #Get electrode information -- bipolar
    tal_path = '/protocols/r1/subjects/'+sub+'/localizations/0/montages/'+str(montage)+'/neuroradiology/current_processed/pairs.json'
    tal_reader = TalReader(filename=tal_path)
    tal_struct = tal_reader.read()
    monopolar_channels = tal_reader.get_monopolar_channels()
    bipolar_pairs = tal_reader.get_bipolar_pairs()

    return tal_struct, bipolar_pairs, monopolar_channels

def get_stim_events(sub, task, montage=0):

    from ptsa.data.readers import BaseEventReader
    from ptsa.data.events import Events
    from ptsa.data.readers.IndexReader import JsonIndexReader
    reader = JsonIndexReader('/protocols/r1.json')

    #Get events
    evfiles = list(reader.aggregate_values('task_events', subject=sub, experiment=task, montage=montage)) #This is supposed to work but often does not
    if len(evfiles)<1:
        from glob import glob
        sessions = [s.split('/')[-1] for s in glob('/protocols/r1/subjects/'+sub+'/experiments/'+task+'/sessions/*')]
        evfiles = []
        for sess in sessions:
            evfiles.append('/protocols/r1/subjects/'+sub+'/experiments/'+task+'/sessions/'+sess+'/behavioral/current_processed/task_events.json')

    evs_on = np.array([]);
    for ef in evfiles:
        try:
            base_e_reader = BaseEventReader(filename=ef, eliminate_events_with_no_eeg=True)
            base_events = base_e_reader.read()
            if len(evs_on) == 0:
                evs_on = base_events[base_events.type=='STIM_ON']
            else:
                evs_on = np.concatenate((evs_on, base_events[base_events.type=='STIM_ON']), axis=0)
        except:
            continue
    evs_on = Events(evs_on);

    #Reorganize sessions around stim sites
    sessions = reader.sessions(subject=sub, experiment=task, montage=montage)
    sess_tags = []
    for i in range(len(evs_on)):
        stimtag = str(evs_on[i]['stim_params']['anode_number'])+'-'+str(evs_on[i]['stim_params']['cathode_number'])
        sess_tags.append(stimtag)
    sess_tags = np.array(sess_tags)
    if len(np.unique(sess_tags))<=len(sessions):
        session_array = evs_on['session']
    else:
        session_array = np.empty(len(evs_on))
        for idx, s in enumerate(np.unique(sess_tags)):
            session_array[sess_tags==s] = int(idx)
    sessions = np.unique(session_array)

    return evs_on, session_array

def get_word_events(sub, task, montage=0):
    from ptsa.data.readers import BaseEventReader
    from ptsa.data.events import Events
    from ptsa.data.readers.IndexReader import JsonIndexReader
    reader = JsonIndexReader('/protocols/r1.json')

    #Get events
    evfiles = list(reader.aggregate_values('task_events', subject=sub, experiment=task, montage=montage))

    evs_on = np.array([]);
    for ef in evfiles:
        try:
            base_e_reader = BaseEventReader(filename=ef, eliminate_events_with_no_eeg=True)
            base_events = base_e_reader.read()
            if len(evs_on) == 0:
                evs_on = base_events[base_events.type=='WORD']
            else:
                evs_on = np.concatenate((evs_on, base_events[base_events.type=='WORD']), axis=0)
        except:
            continue
    evs = Events(evs_on);

    return evs

def get_ps4_events(fn):

    from ptsa.data.readers import BaseEventReader
    from ptsa.data.events import Events

    #Get events
    evfiles = [fn]

    evs_on = np.array([]);
    for ef in evfiles:
        try:
            base_e_reader = BaseEventReader(filename=ef, eliminate_events_with_no_eeg=True)
            base_events = base_e_reader.read()
            if len(evs_on) == 0:
                evs_on = base_events[base_events.type=='STIM_ON']
            else:
                evs_on = np.concatenate((evs_on, base_events[base_events.type=='STIM_ON']), axis=0)
        except:
            continue
    evs_on = Events(evs_on);

    #Reorganize sessions around stim sites
    sessions = np.unique(evs_on['session'])
    sess_tags = []
    for i in range(len(evs_on)):
        stimtag = str(evs_on[i]['anode_num'])+'-'+str(evs_on[i]['cathode_num'])
        sess_tags.append(stimtag)
    sess_tags = np.array(sess_tags)
    if len(np.unique(sess_tags))<=len(sessions):
        session_array = evs_on['session']
    else:
        session_array = np.empty(len(evs_on))
        for idx, s in enumerate(np.unique(sess_tags)):
            session_array[sess_tags==s] = int(idx)
    sessions = np.unique(session_array)

    return evs_on, session_array

def get_stim_eeg(evs_on, tal_struct, bipolar_pairs, monopolar_channels,
                 chans=np.array([]), start_time=-1.0, end_time=1.5):

    from ptsa.data.readers import EEGReader
    from ptsa.data.filters import MonopolarToBipolarMapper

    #Get EEG clips
    eeg_reader = EEGReader(events=evs_on, channels=chans,
                          start_time=start_time, end_time=end_time, buffer_time=0.0)
    eegs = eeg_reader.read()
    eegs = eegs.baseline_corrected((-1.0, 2.5))

    if tal_struct is np.nan:
        return eegs, np.nan

    if ('bipolar_pairs' in list(eegs.coords)):  #Gotta do this for bipolar ENS subjects
        bps_eeg = [list(i) for i in np.array(eegs['bipolar_pairs'])]
        bps_tal = [list(i) for i in np.array(tal_struct['channel'])]
        bps_eeg_good = []
        bps_tal_good = []
        for iidx, i in enumerate(bps_eeg):
            for jidx, j in enumerate(bps_tal):
                if i == j:
                    bps_eeg_good.append(iidx)
                    bps_tal_good.append(jidx)
        bps_eeg_good = np.array(bps_eeg_good)
        bps_tal_good = np.array(bps_tal_good)

        eegs_raw = eegs[bps_eeg_good]
        tal_struct = tal_struct[bps_tal_good]
    else:
        try:
            m2b = MonopolarToBipolarMapper(time_series=eegs, bipolar_pairs=bipolar_pairs)
            eegs_raw = m2b.filter()
        except KeyError:  #we've got to do this subject with specified channels
            eeg_reader = EEGReader(events=evs_on, channels=monopolar_channels,
                          start_time=start_time, end_time=end_time, buffer_time=0.0)
            eegs = eeg_reader.read()
            eegs = eegs.baseline_corrected((-1.0, 2.5))

            #Bipolar rereference
            m2b = MonopolarToBipolarMapper(time_series=eegs, bipolar_pairs=bipolar_pairs)
            eegs_raw = m2b.filter()

    return eegs_raw, tal_struct

def get_tal_distmat(tal_struct):

    #Get distance matrix
    pos = []
    for ts in tal_struct:
        x = ts['atlases']['ind']['x']
        y = ts['atlases']['ind']['y']
        z = ts['atlases']['ind']['z']
        pos.append((x, y, z))
    pos = np.array(pos)

    dist_mat = np.empty((len(pos), len(pos))) # iterate over electrode pairs and build the adjacency matrix
    dist_mat.fill(np.nan)
    for i, e1 in enumerate(pos):
        for j, e2 in enumerate(pos):
            if (i <= j):
                dist_mat[i,j] = np.linalg.norm(e1 - e2, axis=0)
                dist_mat[j,i] = np.linalg.norm(e1 - e2, axis=0)
    distmat = 1./np.exp(dist_mat/120.)

    return distmat

def get_multitaper_power(eegs, tal_struct, pre=[0.05, 0.95], post=[1.55, 2.45],
                        freqs = np.array([5., 6., 7., 8.])):

    from ptsa.data.timeseries import TimeSeriesX
    import mne

    sr = int(eegs.samplerate) #get samplerate

    #Get multitaper power for all channels first
    eegs = eegs[:, :, :].transpose('events', 'bipolar_pairs', 'time')

    #Break into pre/post stimulation periods
    pre_clips = np.array(eegs[:, :, int(sr*pre[0]):int(sr*pre[1])])
    post_clips = np.array(eegs[:, :, int(sr*post[0]):int(sr*post[1])])

    #Get pre/post powers
    mne_evs = np.empty([pre_clips.shape[0], 3]).astype(int)
    mne_evs[:, 0] = np.arange(pre_clips.shape[0])
    mne_evs[:, 1] = pre_clips.shape[2]
    mne_evs[:, 2] = list(np.zeros(pre_clips.shape[0]))
    event_id = dict(resting=0)
    tmin=0.0
    info = mne.create_info([str(i) for i in range(eegs.shape[1])], sr, ch_types='eeg')

    pre_arr = mne.EpochsArray(np.array(pre_clips), info, mne_evs, tmin, event_id)
    post_arr = mne.EpochsArray(np.array(post_clips), info, mne_evs, tmin, event_id)

    #Use MNE for multitaper power
    pre_pows, fdone = mne.time_frequency.psd_multitaper(pre_arr, fmin=freqs[0], fmax=freqs[-1], tmin=0.0,
                                                       verbose=False);
    post_pows, fdone = mne.time_frequency.psd_multitaper(post_arr, fmin=freqs[0], fmax=freqs[-1], tmin=0.0,
                                                        verbose=False);

    pre_pows = np.mean(np.log10(pre_pows), 2) #will be shaped n_epochs, n_channels
    post_pows = np.mean(np.log10(post_pows), 2)

    return pre_pows, post_pows

def find_sat_events(eegs):
    #Return array of chans x events with 1s where saturation is found

    def zero_runs(a):
        a = np.array(a)
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    sat_events = np.zeros([eegs.shape[0], eegs.shape[1]])

    for i in range(eegs.shape[0]):
        for j in range(eegs.shape[1]):
            ts = eegs[i, j]
            zr = zero_runs(np.diff(np.array(ts)))
            numzeros = zr[:, 1]-zr[:, 0]
            if (numzeros>9).any():
                sat_events[i, j] = 1
                continue

    return sat_events.astype(bool)

def channel_exclusion(eegs, sr):
    #Return p-values indicating channels with significant artifact: t-test and levene variance test

    from scipy.stats import ttest_rel
    from ptsa.data.timeseries import TimeSeriesX
    from scipy.stats import levene

    def justfinites(arr):
        return arr[np.isfinite(arr)]


    pvals = []; lev_pvals = [];
    for i in range(eegs.shape[0]):
        ts = eegs[i, :, :]
        pre_eeg = np.mean(ts[:, int(sr*0.6):int(sr*0.95)], 1)
        post_eeg = np.mean(ts[:, int(sr*1.55):int(sr*1.9)], 1)
        eeg_t_chan, eeg_p_chan = ttest_rel(post_eeg, pre_eeg, nan_policy='omit')
        pvals.append(eeg_p_chan)
        try:
            lev_t, lev_p = levene(justfinites(post_eeg), justfinites(pre_eeg))
            lev_pvals.append(lev_p)
        except:
            lev_pvals.append(0.0)

    return np.array(pvals), np.array(lev_pvals)

def get_wm_dist(s, tal_struct, stimbp):
    import nibabel

    #Get distance to nearest white matter
    coordsR, faces = nibabel.freesurfer.io.read_geometry('/data/eeg/freesurfer/subjects/'+s+'/surf/rh.white')
    coordsL, faces = nibabel.freesurfer.io.read_geometry('/data/eeg/freesurfer/subjects/'+s+'/surf/lh.white')
    coords = np.vstack([coordsR, coordsL])

    xyz = [tal_struct[stimbp]['atlases']['ind']['x'], tal_struct[stimbp]['atlases']['ind']['y'], tal_struct[stimbp]['atlases']['ind']['z']]

    coords_diff = np.array([coords[:, 0]-xyz[0], coords[:, 1]-xyz[1], coords[:, 2]-xyz[2]]).T
    coords_dist = np.sqrt(np.sum(coords_diff**2, 1))
    dist_nearest_wm = np.min(coords_dist)

    return dist_nearest_wm

def get_elec_regions(tal_struct):

    regs = []

    for e in tal_struct['atlases']:
        try:
            if 'stein' in list(e.keys()):
                if (e['stein']['region'] is not None) and (len(e['stein']['region'])>1):
                    regs.append(e['stein']['region'].lower())
                    continue
                else:
                    pass
            if 'das' in list(e.keys()):
                if (e['das']['region'] is not None) and (len(e['das']['region'])>1):
                    regs.append(e['das']['region'].lower())
                    continue
                else:
                    pass
            if 'ind' in list(e.keys()):
                if (e['ind']['region'] is not None) and (len(e['ind']['region'])>1):
                    regs.append(e['ind']['region'].lower())
                    continue
                else:
                    pass
            if 'wb' in list(e.keys()):
                if (e['wb']['region'] is not None) and (len(e['wb']['region'])>1):
                    regs.append(e['wb']['region'].lower())
                    continue
                else:
                    pass
            else:
                regs.append('')
                #print 'no atlas found'
        except AttributeError:
            regs.append('')

    return np.array(regs)
