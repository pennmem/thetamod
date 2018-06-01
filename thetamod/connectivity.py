import mne
import numpy as np

from cml_pipelines import task
from cmlreaders import CMLReader

from ptsa.data.TimeSeriesX import TimeSeriesX as TimeSeries

__all__ = [
    'ptsa_to_mne',
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


# @task()
def ptsa_to_mne(eegs):
    """Convert PTSA :class:`TimeSeries` data to MNE :class:`EpochsArray` data.

    Parameters
    ----------
    eegs : List[TimeSeries] or TimeSeries
        EEG data to convert to MNE format. This should already be organized in
        events x channels x time ordering.

    Returns
    -------
    epochs : EpochsArray

    """
    if isinstance(eegs, TimeSeries):
        eegs = [eegs]

    names = eegs[0]['channels'].data.tolist()
    info = mne.create_info(names, eegs[0]['samplerate'], ch_types='eeg')
    data = np.concatenate(eegs, axis=0)

    events = np.empty([data.shape[0], 3], dtype=int)
    events[:, 0] = list(range(data.shape[0]))
    # FIXME: are these ok?
    events[:, 1] = 0
    events[:, 2] = 0
    event_id = {'resting': 0}

    epochs = mne.EpochsArray(data, info, events, event_id=event_id,
                             verbose=False)
    return epochs


def resting_state_connectivity(subject, experiment, session, localization=0,
                               montage=0):
    """Compute resting state connectivity coherence matrix.

    Parameters
    ----------
    subject : str
        Subject ID
    experiment : str
        Experiment type
    session : int
        Session number
    localization : int
        Localization number (default: 0)
    montage : int
        Montage number (default: 0)

    Returns
    -------
    ???

    """
    freqs = FREQUENCY_BANDS['theta-alpha']
