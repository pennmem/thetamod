def get_tmi(sub, task, nonstimtask, pthresh=0.0001):

    from pred_stim_pipeline import get_bp_tal_struct, get_stim_eeg, get_stim_events, find_sat_events, get_multitaper_power, channel_exclusion, get_tal_distmat
    from scipy.stats import ttest_rel, pearsonr
    from scipy.special import logit
    import numpy as np
    evs, ev_sessions = get_stim_events(sub, task)

    if len(evs)<25:
        print 'Too few events!'
        return

    try:
        tal_struct, bipolar_pairs, mpchans = get_bp_tal_struct(sub, montage=0)
        no_tal_struct = False
    except:
        no_tal_struct = True
        tal_struct = np.nan
        bipolar_pairs = np.nan
        mpchans = np.nan

    if no_tal_struct:
        eegs, _ = get_stim_eeg(evs, tal_struct, bipolar_pairs, mpchans, chans=np.array([]), start_time=-1.0, end_time=1.5)
    else:
        eegs, _ = get_stim_eeg(evs, tal_struct, bipolar_pairs, mpchans, chans=mpchans, start_time=-1.0, end_time=1.5)

    #Identify bad trials: stim occurs shortly before/after another event
    msdiff = np.diff(evs.mstime)
    msdiff = np.append(msdiff, np.nan)
    for idx, i in enumerate(msdiff):
        if i < 1500: 
            msdiff[idx] = np.nan
            msdiff[idx+1] = np.nan

    sat_events = find_sat_events(eegs) #saturated events

    import copy
    eegs_filt = copy.deepcopy(eegs) #Keep a copy of the original for power computations

    #Remove bad events
    chan_sat = np.where(sat_events==True)[0]
    ev_sat = np.where(sat_events==True)[1]

    for idx in range(len(ev_sat)):
        i = chan_sat[idx]
        j = ev_sat[idx]
        eegs_filt[i, j, :] = np.nan

    for idx in np.where(np.isnan(msdiff))[0]:
        eegs_filt[:, idx, :] = np.nan

    pre_pows, post_pows = get_multitaper_power(eegs, tal_struct=tal_struct, pre=[0.05, 0.95], post=[1.55, 2.45],
                    freqs = np.array([5., 8.]))

    #Remove saturated or repeated stim events
    post_pows[np.array(np.isnan(eegs_filt[:, :, 0].T))] = np.nan
    pre_pows[np.array(np.isnan(eegs_filt[:, :, 0].T))] = np.nan

    conn = np.load('/scratch/esolo/tmi_analysis/'+sub+'/'+sub+'_baseline3trials_network_theta-alpha.npy')

    tmi_R = []
    tmi_Z = []
    stim_elec_label = []
    stim_elec_idx = []

    for sess in np.unique(ev_sessions):

        pvals, lev_pvals = channel_exclusion(eegs_filt[:, ev_sessions==sess, :])

        #Find stim electrode
        anode_num = str(evs[evs['session']==sess][0]['stim_params']['anode_number'])
        cathode_num = str(evs[evs['session']==sess][0]['stim_params']['cathode_number'])
        eeg_bps = [(int(i[0]), int(i[1])) for i in np.array(eegs['bipolar_pairs'])]
        stimbp = [str(i) for i in eeg_bps].index('('+anode_num+', '+cathode_num+')')

        anode_label = str(evs[evs['session']==sess][0]['stim_params']['anode_label'])
        cathode_label = str(evs[evs['session']==sess][0]['stim_params']['cathode_label'])
        print anode_label+'-'+cathode_label

        T, p = ttest_rel(post_pows[ev_sessions==sess, :], pre_pows[ev_sessions==sess, :], axis=0, nan_policy='omit')
        T = np.array(T); T[T==0] = np.nan
        T[pvals<pthresh] = np.nan #Remove artifactual channels
        T[np.sum(np.isfinite(post_pows), 0)<20] = np.nan #remove channels with very few trials
        logit_conn = logit(conn[stimbp])
        filt = np.logical_and(np.isfinite(T), np.isfinite(logit_conn)) #the GOOD electrodes

        if len(T[filt])<10:
            print 'Too few electrodes!'
            tmi_Z.append(np.nan)
            tmi_R.append(np.nan)
            continue

        if ~no_tal_struct:
            print 'Doing regression....'
            ### DO REGRESSION ###
            import statsmodels.formula.api as sm
            distmat = get_tal_distmat(tal_struct)

            #Set up feature matrices
            chan_T = copy.copy(T)
            chan_T[~filt] = np.nan
            X = np.empty([np.sum(~np.isnan(chan_T)), 3])
            X[:, 0] = distmat[stimbp][~np.isnan(chan_T)]
            X[:, 1] = logit_conn[~np.isnan(chan_T)]
            X[:, 2] = np.ones(np.sum(~np.isnan(chan_T)))
            y = chan_T[~np.isnan(chan_T)]

            #Fit the model
            result = sm.OLS(y, X).fit()
            tru_coefs = copy.copy(result.params)

            #Shuffle for null coefficients
            null_coefs = []
            shuf_idxs = np.arange(X.shape[0])
            for k in range(1000):
                np.random.shuffle(shuf_idxs)
                null_result = sm.OLS(y, X[shuf_idxs, :]).fit()
                null_coefs.append(null_result.params)
            null_coefs = np.array(null_coefs)

            #Z-score and p-value true vs. null coefs
            coef_stats = np.empty([len(tru_coefs), 2])
            for idx, i in enumerate(tru_coefs):
                coef_stats[idx, 0] = (i-np.nanmean(null_coefs[:, idx]))/np.nanstd(null_coefs[:, idx]) #z-score
                coef_stats[idx, 1] = np.sum(null_coefs[:, idx]>i)/float(null_coefs.shape[0])

            tmi_Z.append(coef_stats[1, 0])

        else:
            distmat = np.nan
            tmi_Z.append(np.nan)

        tmi_R.append(pearsonr(T[filt], logit_conn[filt])[0])
        stim_elec_label.append(anode_label+'-'+cathode_label)
        stim_elec_idx.append(stimbp)

    mydict = {}
    mydict['subject'] = sub
    mydict['task'] = task
    mydict['conn'] = conn
    mydict['distmat'] = distmat
    mydict['T'] = T
    mydict['pvals'] = pvals
    mydict['post_pows'] = post_pows
    mydict['pre_pows'] = pre_pows
    mydict['tmi_Z'] = tmi_Z
    mydict['tmi_R'] = tmi_R
    mydict['stim_elec_label'] = stim_elec_label
    mydict['stim_elec_idx'] = stim_elec_idx
    mydict['sessions'] = ev_sessions

    import cPickle as pk
    import os
    try:
        os.mkdir('/scratch/esolo/tmi_analysis/'+sub+'/')
    except:
        pass
    pk.dump(mydict, open('/scratch/esolo/tmi_analysis/'+sub+'/tmi_output_'+task+'.pk', 'wb'))