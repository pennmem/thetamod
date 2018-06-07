from functools import partial

from mne import create_info, EpochsArray
from mne.time_frequency import psd_multitaper
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import ttest_rel, pearsonr

from cmlreaders import CMLReader


def get_eeg(which, subject, experiment, session, buffer=50, window=900,
            stim_duration=500, reref=True, rootdir=None):
    """Get EEG data for pre- or post-stim periods.

    Parameters
    ----------
    which : str
        "pre" or "post"
    subject : str
        Subject ID.
    experiment : str
        Experiment name.
    session : int
        Session number.
    buffer : int
        Time in ms pre-stim to avoid (default: 50).
    window : int
        Evaluation window length in ms (default: 900).
    stim_duration : int
        Stimulation duration in ms (default: 500).
    reref : bool
        Try to rereference EEG data (will fail if recorded in hardware bipolar
        mode). Default: True.
    rootdir : str
        Path to rhino's root directory.

    Returns
    -------
    eeg : TimeSeries

    Notes
    -----
    This assumes all stim durations are the same.

    """
    if which not in ["pre", "post"]:
        raise ValueError("Specify 'pre' or 'post'")

    reader = CMLReader(subject, experiment, session, rootdir=rootdir)
    events = reader.load("events")
    stim_events = events[events.type == "STIM_ON"]

    if which == "pre":
        rel_start = -(buffer + window)
        rel_stop = -buffer
    else:
        rel_start = buffer + stim_duration
        rel_stop = buffer + stim_duration + window

    if reref:
        scheme = reader.load("pairs")
    else:
        scheme = None

    eeg = reader.load_eeg(events=stim_events,
                          rel_start=rel_start,
                          rel_stop=rel_stop,
                          scheme=scheme)

    return eeg


def compute_psd(eegs, fmin=5., fmax=8.):
    """Compute power spectral density using multitapers.

    Parameters
    ----------
    eegs : TimeSeries
        EEG data
    fmin : float
        Minimum frequency of interest (default: 5)
    fmax : float
        Maximum frequency of interest (default: 8)

    Returns
    -------
    powers : np.ndarray
        Power spectral densities

    Notes
    -----
    Ethan's method involves removing saturated events by looking at consecutive
    numbers of zeros. This will not work in general because hardware bipolar
    referencing can show saturation at other values. Instead, we assume here
    that channels showing a lot of saturation are removed prior to computing
    PSD.

    """
    ea = eegs.to_mne()
    pows, fdone = psd_multitaper(ea, fmin=fmin, fmax=fmax, tmin=0.0,
                                 verbose=False)
    powers = np.mean(np.log10(pows), 2)
    return powers


def get_distances(pairs):
    """Get distances as an adjacency matrix.

    Parameters
    ----------
    pairs : pd.DataFrame
        A DataFrame as returned by cmlreaders.

    Returns
    -------
    distmat : np.ndarray
        Adjacency matrix using exp(-distance / 120).

    """
    # positions matrix shaped as N_channels x 3
    pos = np.array([
        [row["ind.{}".format(c)] for c in ("x", "y", "z")]
        for _, row in pairs.iterrows()
    ])

    distmat = np.empty((len(pos), len(pos)))

    for i, d1 in enumerate(pos):
        for j, d2 in enumerate(pos):
            if i <= j:
                distmat[i, j] = np.linalg.norm(d1 - d2, axis=0)
                distmat[j, i] = np.linalg.norm(d1 - d2, axis=0)

    distmat = 1 / np.exp(distmat / 120.)
    return distmat


def regress_distance():
    """Do regression on channel distances.

    Parameters
    ----------

    Returns
    -------

    """


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

        # TODO: distance regression

    return tmi
