def get_resting_networks(s):
    import MNE_pipeline_refactored as mpipe
    import numpy as np

    try:
        sub = s[0]
        nonstimtask = s[3]

        p = mpipe.mne_pipeline(sub, task=nonstimtask, win_st=0.0, win_fin=1.0, 
                                          conn_type='coh', mode='multitaper', samplerate=256.0, montage=0,
                                          root='/scratch/esolo/tmi_analysis/', buf_time=0.0)
        p.set_freq_range(band='theta-alpha')
        p.tal_struct = np.nan; p.ref_scheme = np.nan; p.monopolar_channels = np.nan
        p.ref_scheme = 'bipolar'; p.good_elecs_only = False; p.MTL_only = False
        p.set_events_resting()
        p.set_mne_structure_encoding()
        p.get_resting_networks_coh()
    except:
        return