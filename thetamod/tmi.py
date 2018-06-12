import copy

from mne.time_frequency import psd_multitaper
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import ttest_rel, pearsonr
import statsmodels.formula.api as sm


def get_stim_events(reader):
    """Get all stim events.

    Parameters
    ----------
    reader : CMLReader

    Returns
    -------
    stim_events : pd.DataFrame

    """
    events = reader.load("events")
    stim_events = events[events.type == "STIM_ON"]
    return stim_events


def get_stim_channels(pairs, stim_events):
    """Extract unique stim channels from stim events.

    Parameters
    ----------
    pairs : pd.DataFrame
        Channel info.
    stim_events : pd.DataFrame
        Stimulation events.

    Returns
    -------
    indices : List[int]
        List of channel indices for each stim channel.

    """
    stim_params = pd.DataFrame([
        row for row in stim_events[stim_events.type == "STIM_ON"].stim_params
    ])

    labels = np.unique([
        "{}-{}".format(row.anode_label, row.cathode_label)
        for _, row in stim_params.iterrows()
    ])

    indices = [pairs[pairs.label == label].index[0] for label in labels]

    return indices


def get_eeg(which, reader, stim_events, buffer=50, window=900,
            stim_duration=500, reref=True):
    """Get EEG data for pre- or post-stim periods.

    Parameters
    ----------
    which : str
        "pre" or "post"
    reader : CMLReader
        Reader for loading EEG data.
    stim_events : pd.DataFrame
        Stimulation events as a DataFrame.
    buffer : int
        Time in ms pre-stim to avoid (default: 50).
    window : int
        Evaluation window length in ms (default: 900).
    stim_duration : int
        Stimulation duration in ms (default: 500).
    reref : bool
        Try to rereference EEG data (will fail if recorded in hardware bipolar
        mode). Default: True.

    Returns
    -------
    eeg : TimeSeries

    Notes
    -----
    This assumes all stim durations are the same.

    """
    if which not in ["pre", "post"]:
        raise ValueError("Specify 'pre' or 'post'")

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


def regress_distance(pre_psd, post_psd, conn, distmat, stim_channel_idxs,
                     nperms=1000):
    """Do regression on channel distances.

    Parameters
    ----------
    pre_psd : np.ndarray
        Pre-stim power spectral density
    post_psd : np.ndarray
        Post-stim power spectral density
    conn : np.ndarray
        Connectivity matrix.
    distmat : np.ndarray
        Distance adjacency matrix as computed from :func:`get_distance`.
    stim_channel_idx : int
        Index of the stim channel being analyzed

    Returns
    -------
    results : List[dict]
        A list of dictionaries of regression coefficients, one per stim channel.
        Keys are "coefs" and "null_coefs" for the true and null coefs,
        respectively.

    Notes
    -----
    The model used here is ..math::

        \vec{y} = \beta_0 \vec{x}_0 + \beta_1 \vec{x}_1 + \beta_2 \vec{x}_2

    where :math:`\vec{x}_1` are the distance adjacency values for all stim
    channels, :math:`\vec{x}_2` is the logistic transform of the connectivity
    matrix, and :math:`\vec{x}_2` is the intercept.

    """

    t, p = ttest_rel(post_psd, pre_psd, axis=0)
    t[t == 0] = np.nan  # FIXME: why?

    results = []
    for stim_channel_idx in stim_channel_idxs:
        logit_conn = logit(conn[stim_channel_idx])

        size = np.sum(np.isfinite(t))
        X = np.empty((size, 3))
        y = t

        X[:, 0] = distmat[stim_channel_idx][np.isfinite(t)]
        X[:, 1] = logit_conn[np.isfinite(t)]
        X[:, 2] = np.ones(size)  # intercept

        # print(conn[stim_channel_idx])
        # print(X)
        assert np.isfinite(X).all()

        rval, _ = pearsonr(t, logit_conn)

        result = sm.OLS(y, X).fit()
        coefs = copy.copy(result.params)

        def shuffle_index(N, size):
            idx = np.arange(size)
            for _ in range(N):
                np.random.shuffle(idx)
                yield idx

        # Get null coefficients by shuffling 1000 times
        null_coefs = [
            sm.OLS(y, X[idx, :]).fit().params
            for idx in shuffle_index(nperms, X.shape[0])
        ]

        results.append({
            "coefs": np.array(coefs),
            "null_coefs": np.array(null_coefs),
            "rvalue": rval
        })

    return results


def compute_tmi(regression_results_list):
    """Compute TMI scores.

    Parameters
    ----------
    regression_results : List[dict]
        Results from :func:`regress_distance`.

    Returns
    -------
    tmi : List[dict]
        List of dictionaries containing 'zscore' and 'rvalue' keys.

    """
    tmi = []
    for regression_results in regression_results_list:
        coefs = regression_results["coefs"]
        null_coefs = regression_results["null_coefs"]
        rvalue = regression_results["rvalue"]

        zscores = (coefs - np.nanmean(null_coefs,0)) / np.nanstd(null_coefs,0)

        tmi.append({
            "zscore": zscores[1],
            "rvalue": rvalue,
        })

    return tmi
