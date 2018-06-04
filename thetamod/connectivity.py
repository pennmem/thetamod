import mne
import numpy as np

from cml_pipelines import task
from cmlreaders import CMLReader

__all__ = [
    'get_resting_state_connectivity',
    'read_eeg_data',
]

FREQUENCY_BANDS = {
    'theta': [4., 8.],
    'theta_cwt': np.arange(4., 8.+1, 1),
    'alpha': [9., 13.],
    'alpha_cwt': np.arange(9., 13.+1, 1),
    'theta-alpha': np.arange(4, 14, 1),
    'theta-alpha_cwt': [4., 13.],
    'beta': [16., 28.],
    'beta_cwt': np.arange(16., 28.+1, 2),
    'lowgamma': [30., 60.],
    'lowgamma_cwt': np.arange(30., 60.+1, 5),
    'highgamma': [60., 120.],
    'highgamma_cwt': np.arange(70., 90.+1, 10),
    'hfa_cwt': np.array([75., 80., 85., 90., 95., 100.]),
    'hfa': [30., 120.],
    'all_freqs': np.array([4., 5., 6., 7., 8.,
                           9., 10., 11., 12., 13.,
                           16., 18., 20., 22., 24., 26., 28.,
                           30., 32., 34., 36., 38., 40., 42., 44., 46., 48.,
                           50.]),
    'all_freqs_ext': np.array([4., 5., 6., 7., 8.,
                               9., 10., 11., 12., 13.,
                               16., 18., 20., 22., 24., 26., 28.,
                               30., 32., 34., 36., 38., 40., 42., 44., 46., 48.,
                               50., 55., 60., 65., 70., 75., 80.])
}


# @task()
def read_eeg_data(reader):
    """Read EEG data from "resting" events in a single session. This selects 3
    EEG epochs of 1 s each starting at offsets of 1, 3, and 7 seconds from the
    beginning of the countdown phase.

    Parameters
    ----------
    reader : CMLReader
        The reader object.

    Returns
    -------
    eeg
        EEG timeseries data.

    Notes
    -----
    This assumes a countdown phase of at least 10 seconds in length.

    """
    events = reader.load('events')
    countdowns = events[events.type == 'COUNTDOWN_START']

    # get times in ms of countdowns relative to session start
    start_times = countdowns.mstime.values - events.iloc[0].mstime

    # construct epochs
    epochs = []
    for start in start_times:
        list_epochs = []
        for offset in [1000, 3000, 7000]:
            list_epochs.append((start + offset, start + offset + 1000))
        epochs += list_epochs

    eeg = reader.load_eeg(epochs=epochs)
    return eeg


def get_resting_state_connectivity(array):
    """Compute resting state connectivity coherence matrix.

    Parameters
    ----------
    array : mne.EpochsArray

    Returns
    -------
    ???

    """
    freqs = FREQUENCY_BANDS['theta-alpha']
    fmin, fmax = freqs[0], freqs[-1]
    sample_rate = 1000.
    out = mne.connectivity.spectral_connectivity(array,
                                                 method='coh',
                                                 mode='multitaper',
                                                 sfreq=sample_rate,
                                                 fmin=fmin, fmax=fmax,
                                                 faverage=True,
                                                 tmin=0.0,
                                                 mt_adaptive=False,
                                                 n_jobs=1,
                                                 verbose=False)
    con, freqs, times, n_epochs, n_tapers = out

    # FIXME: what is this?
    cons_rec = con[:, :, 0]

    # Symmetrize average network
    mu = cons_rec
    mu_full = np.nansum(np.array([mu, mu.T]), 0)
    mu_full[np.diag_indices_from(mu_full)] = 0.0
    return mu_full
