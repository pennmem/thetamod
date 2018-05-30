from functools import partial

from mne import create_info, EpochsArray
from mne.time_frequency import psd_multitaper
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import ttest_rel, pearsonr


def compute_distances(channel_info):
    """Compute channel distances.

    Parameters
    ----------
    channel_info : dict
        Channel locations and other info.

    Returns
    -------
    distmat : np.ndarray

    """
    pos = np.array([
        (ch['atlases']['ind']['x'],
         ch['atlases']['ind']['y'],
         ch['atlases']['ind']['z'])
        for ch in channel_info
    ])

    # iterate over electrode pairs and build the adjacency matrix
    dist_mat = np.empty((len(pos), len(pos)))
    dist_mat.fill(np.nan)
    for i, e1 in enumerate(pos):
        for j, e2 in enumerate(pos):
            if i <= j:
                dist_mat[i, j] = np.linalg.norm(e1 - e2, axis=0)
                dist_mat[j, i] = np.linalg.norm(e1 - e2, axis=0)
    distmat = 1./np.exp(dist_mat/120.)

    return distmat


def compute_psd(eegs, pre=(0.05, 0.95), post=(1.55, 2.45), fmin=5., fmax=8.):
    """Compute power spectral density using multitapers.

    Parameters
    ----------
    eegs : TimeSeries
        EEG data
    pre : list-like
        ???
    post : list-like
        ???
    fmin : float
        Minimum frequency of interest (default: 5)
    fmax : float
        Maximum frequency of interest (default: 8)

    Returns
    -------
    power_dict : Dict[str, np.ndarray]
        A dict of arrays with keys pre and post. Each entry is a
        n_epochs x n_channels array.

    """
    sr = int(eegs.samplerate)

    # Get multitaper power for all channels first
    eegs = eegs[:, :, :].transpose('events', 'bipolar_pairs', 'time')

    # Break into pre/post stimulation periods
    pre_clips = np.array(eegs[:, :, int(sr*pre[0]):int(sr * pre[1])])
    post_clips = np.array(eegs[:, :, int(sr*post[0]):int(sr * post[1])])

    # Get pre/post powers
    mne_evs = np.empty([pre_clips.shape[0], 3]).astype(int)
    mne_evs[:, 0] = np.arange(pre_clips.shape[0])
    mne_evs[:, 1] = pre_clips.shape[2]
    mne_evs[:, 2] = list(np.zeros(pre_clips.shape[0]))
    event_id = dict(resting=0)
    tmin = 0.0
    info = create_info([str(i) for i in range(eegs.shape[1])], sr, ch_types='eeg')

    pre_arr = EpochsArray(np.array(pre_clips), info, mne_evs, tmin, event_id)
    post_arr = EpochsArray(np.array(post_clips), info, mne_evs, tmin, event_id)

    # Use MNE for multitaper power
    get_psd = partial(psd_multitaper,
                      fmin=fmin, fmax=fmax, tmin=tmin, verbose=False)
    pre_pows, fdone = get_psd(pre_arr)
    post_pows, fdone = get_psd(post_arr)

    # shape: n_epochs, n_channels
    pre_pows = np.mean(np.log10(pre_pows), 2)
    post_pows = np.mean(np.log10(post_pows), 2)

    power_dict = {
        'pre': pre_pows,
        'post': post_pows,
    }
    return power_dict


def compute_tmi(events, channel_info, powers):
    """Compute TMI scores.

    Parameters
    ----------
    events : np.recarray
        Events recarray.
    channel_info : dict
        Channel information.
    powers : Dict[str, np.ndarray]
        Pre- and post-stim powers.

    Returns
    -------
    tmi : dict
        Dictionary containing 'zscore' and 'rvalue' keys.

    """
    events = pd.DataFrame.from_records(events)

    tmi = {
        'zscore': [],
        'rvalue': [],
    }

    for i, session in enumerate(events.session.unique()):
        pre_powers = powers['pre'][i]
        post_powers = powers['post'][i]
        t, p = ttest_rel(post_powers, pre_powers, axis=0, nan_policy='omit')

        T = np.array(t)
        T[T == 0] = np.nan

        # FIXME: remove artifactual channels

        # ignore channels with too few trials
        T[np.sum(np.isfinite(post_powers), axis=0) < 20] = np.nan

        # TODO

    return tmi
