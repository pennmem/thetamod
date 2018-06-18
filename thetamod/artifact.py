import numpy as np
from cmlreaders import CMLReader
from scipy.stats import ttest_rel
from scipy.stats import levene
import itertools

"""
Three-stage filter for rejecting artifactual signals:

1. Reject any channel labelled as bad in `electrode_categories.txt`
2. Reject any trial that shows amplifier saturation
3. Reject any channel that shows too large a difference between the power spectra
pre- and post-stim, as measured using a paired t-test

"""

__all__ = ['get_saturated_events_mask', 'get_bad_channel_names',
           'get_channel_exclusion_pvals', 'invalidate_eeg']


def get_bad_channel_names(subject, montage, just_bad=None,rhino_root='/'):

    reader = CMLReader(subject, montage=montage,rootdir=rhino_root)
    electrode_categories = reader.load('electrode_categories')
    if just_bad is True:
        return (electrode_categories.get('bad_electrodes',[])
                + electrode_categories.get('broken_leads',[])
                )
    else:
        return list(itertools.chain(*electrode_categories.values()))


def get_saturated_events_mask(eegs):
    # Return array of chans x events with 1s where saturation is found

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
            numzeros = zr[:, 1] - zr[:, 0]
            if (numzeros > 9).any():
                sat_events[i, j] = 1
                continue

    return sat_events.astype(bool)


def get_channel_exclusion_pvals(pre_eeg, post_eeg):
    """
    Estimate which channels show broadband DC shift post-stimulation
    using T-test and Levene variance test.

    Parameters
    ----------
    pre_eeg: np.ndarray
       Pre-stimulus EEG signals
    post_eeg
       Post-stimulus EEG signals

    Returns
    -------

    pvals: np.ndarray
        P-values from paired t-test between pre-stim EEG and post-stim EEG

    lev_pvals: np.ndarray
        P-values from Levene variance test between pre-stim EEG and post-stim
        EEG.
    """

    def justfinites(arr):
        return arr[np.isfinite(arr)]

    pvals = []
    lev_pvals = []
    assert pre_eeg.shape == post_eeg.shape #FIXME: RAISE A PROPER EXCEPTION
    for i in range(pre_eeg.shape[0]):
        eeg_t_chan, eeg_p_chan = ttest_rel(post_eeg, pre_eeg, nan_policy='omit')
        pvals.append(eeg_p_chan)
        try:
            lev_t, lev_p = levene(justfinites(post_eeg), justfinites(pre_eeg))
            lev_pvals.append(lev_p)
        except Exception:
            lev_pvals.append(0.0)

    return np.array(pvals), np.array(lev_pvals)


def invalidate_eeg(reader,channels,pre_eeg,post_eeg,rhino_root,thresh=1e-5):
    saturation_mask = get_saturated_events_mask(post_eeg)
    bad_channel_list = get_bad_channel_names(reader.subject, reader.montage,
                                             rhino_root=rhino_root)
    pvals, _ = get_channel_exclusion_pvals(pre_eeg, post_eeg)
    excluded_dc_shift = pvals <= thresh
    bad_channel_mask = np.in1d(channels, bad_channel_list) | saturation_mask

    pre_eeg[:, bad_channel_mask, :] = np.nan
    post_eeg[:, bad_channel_mask, :] = np.nan
    pre_eeg[excluded_dc_shift, :] = np.nan
    post_eeg[excluded_dc_shift, :] = np.nan

    return pre_eeg, post_eeg
