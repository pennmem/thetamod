import functools

import mne
import numpy as np

from cml_pipelines import task
from cmlreaders import CMLReader

from ptsa.data import TimeSeriesX as TimeSeries
from ptsa.data.filters import ButterworthFilter, MonopolarToBipolarMapper
from ptsa.data.readers import (
    BaseEventReader, EEGReader, JsonIndexReader, ParamsReader
)

__all__ = [
    'get_resting_events',
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


@task()
def get_resting_events(reader, file_path=None):
    """Gets "resting" events from the COUNTDOWN phase.

    Parameters
    ----------
    reader : CMLReader
        The reader object.
    file_path : str
        Absolute path to the file to read (for testing purposes only)

    Returns
    -------
    events : pd.DataFrame

    """
    events = reader.load('events', file_path=file_path)
    events = events[events.type == 'COUNTDOWN_START']
    return events

    evsize = events.size
    events = np.concatenate((events, events, events))

    # Get samplerate to do eegoffsets
    filename = events[0].eegfile.split('noreref')[0] + 'sources.json'
    samplerate = ParamsReader(dataroot=events[0].eegfile,
                              filename=filename).read()['samplerate']

    # Look at first quarter of countdown
    events[:evsize].eegoffset = (
        events[:evsize].eegoffset + int(samplerate * 1.0)
    )

    # Look at last quarter of countdown
    events[evsize:evsize * 2].eegoffset = (
        events[evsize:evsize * 2].eegoffset + int(samplerate * 4.0)
    )

    events[evsize * 2:].eegoffset = events[evsize * 2:].eegoffset + int(samplerate * 7.0)

    return events


@task()
def read_eeg_data(subject, events):
    """Read EEG data."""


@task()
def ptsa_to_mne(eegs, task_phase='resting'):
    """Convert PTSA :class:`TimeSeries` data to MNE :class:`EpochsArray` data.

    Parameters
    ----------
    eegs : TimeSeries
    task_phase : str
        Task phase to consider (default: 'resting')

    Returns
    -------
    epochs : EpochsArray

    """
    names = []  # FIXME
    info = mne.create_info(names, eegs['samplerate'], ch_types='eeg')
    data = eegs.transpose('events', 'channels', 'time')

    events = np.empty([len(data['events']), 3], dtype=int)
    events[:, 0] = list(range(len(data['events'])))
    events[:, 1] = len(data['time'])
    events[:, 2] = list(eegs['events'].recalled)

    event_id = (
        {'resting': 0}
        if task_phase == 'resting' else
        {'recalled': 1, 'not_recalled': 0}
    )

    epochs = mne.EpochsArray(data, info, events, event_id=event_id)
    return epochs


def set_mne_structure_encoding(subject, events, channels=np.array([]),
                               notch_filter=True, ref_scheme='bipolar',
                               mtl_only=False, win_st=0.0, win_fin=1.0,
                               buffer_time=0.0):
    """Setup for MNE...

    Parameters
    ----------
    subject : str
    events : np.recarray
    channels : np.ndarray
    notch_filter : bool
        Filter out line noise (default: True)
    ref_scheme : str
        One of: 'bipolar', 'average'
    mtl_only : bool
        Only consider MTL electrodes (default: False)
    win_st : float
        Window start (default: 0.0)
    win_fin : float
        Window stop (default: 1.0)
    buffer_time : float
        Buffer time (default: 0)

    Returns
    -------
    TODO

    Raises
    ------
    ValueError
        When an invalid reference scheme is given.

    """
    if ref_scheme not in ['bipolar', 'average']:
        raise ValueError("ref_scheme must be 'bipolar' or 'average'")

    make_reader = functools.partial(EEGReader,
                                    events=events, start_time=win_st,
                                    end_time=win_fin,
                                    buffer_time=buffer_time)
    eeg_reader = make_reader(channels=channels)

    try:
        eegs = eeg_reader.read()
        if eegs.shape[0] == 0:
            # this subject needs monopolar channels to load data, even if we
            # asked for all of them
            if ref_scheme == 'average':
                # FIXME
                self.set_elec_info_avgref(good_elecs_only=self.good_elecs_only,
                                          MTL_only=mtl_only)
            if ref_scheme == 'bipolar':
                # FIXME
                self.set_elec_info_bipolar(good_elecs_only=self.good_elecs_only,
                                           MTL_only=mtl_only)

            eeg_reader = make_reader(channels=self.monopolar_channels)
            eegs = eeg_reader.read()
    except TypeError:
        for evidx in range(len(events)):
            if evidx == 0:
                eegs = EEGReader(events=events[evidx:evidx+1],
                                 channels=self.monopolar_channels,
                                 start_time=win_st, end_time=win_fin,
                                 buffer_time=buffer_time).read()
                sr_to_use = eegs['samplerate']
            else:
                new_eeg = EEGReader(events=events[evidx:evidx+1],
                                    channels=self.monopolar_channels,
                                    start_time=win_st, end_time=win_fin,
                                    buffer_time=buffer_time).read()
                new_eeg['samplerate'] = sr_to_use
                eegs = eegs.append(new_eeg, dim='events')

    if 'bipolar_pairs' in list(eegs.coords):
        bps_eeg = [list(i) for i in np.array(eegs['bipolar_pairs'])]
        bps_tal = [list(i) for i in np.array(self.tal_struct['channel'])]
        bps_eeg_good = []
        bps_tal_good = []
        for iidx, i in enumerate(bps_eeg):
            for jidx, j in enumerate(bps_tal):
                if i == j:
                    bps_eeg_good.append(iidx)
                    bps_tal_good.append(jidx)
        bps_eeg_good = np.array(bps_eeg_good)
        bps_tal_good = np.array(bps_tal_good)

        eegs = eegs[bps_eeg_good]
        self.tal_struct = self.tal_struct[bps_tal_good]

        #Resave data
        np.save(self.root+''+subject+'/elec_info_bipol.npy', self.tal_struct)

    if ref_scheme == 'bipolar' and 'bipolar_pairs' not in list(eegs.coords):
        # A non-BP ENS subject
        m2b = MonopolarToBipolarMapper(time_series=eegs, bipolar_pairs=self.bipolar_pairs)
        eegs = m2b.filter()

    if notch_filter:
        # Filter out line noise
        if subject[0:2] == 'FR':
            freq_range = [48., 52.]
        else:
            freq_range = [58., 62.]
        b_filter = ButterworthFilter(time_series=eegs, freq_range=freq_range,
                                     filt_type='stop', order=4)
        eegs_filtered = b_filter.filter()

        if samplerate is not None:
            eegs_filtered = eegs_filtered.resampled(self.samplerate)
    else:

        if samplerate is not None:
            eegs_filtered = eegs.resampled(self.samplerate)

    # Create MNE dataset
    n_channels = eegs_filtered.shape[0]
    if self.tal_struct is not np.nan:
        ch_names = list(self.tal_struct['tagName'])
    else:
        ch_names = [str(i) for i in range(n_channels)]
    info = mne.create_info(ch_names, self.samplerate, ch_types='eeg')

    # Reorganize data for MNE format
    if self.ref_scheme == 'bipolar':
        data = eegs_filtered.transpose('events', 'bipolar_pairs', 'time')
    else:
        try:
            data = eegs_filtered.transpose('events', 'channels', 'time')
        except:
            data = eegs_filtered.transpose('events', 'bipolar_pairs', 'time')

    # Create events array for MNE
    mne_evs = np.empty([data['events'].shape[0], 3]).astype(int)
    mne_evs[:, 0] = np.arange(data['events'].shape[0])
    mne_evs[:, 1] = data['time'].shape[0]
    mne_evs[:, 2] = list(self.evs.recalled)

    if self.task_phase == 'resting':
        event_id = dict(resting=0)
    else:
        event_id = dict(recalled=1, not_recalled=0)
    tmin=0.0

    arr = mne.EpochsArray(np.array(data), info, mne_evs, tmin, event_id)

    if self.ref_scheme == 'average':
        arr.set_eeg_reference(ref_channels=None)  #Set to average reference
        arr.apply_proj()

    self.arr = arr
    self.session = self.evs.session


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
    events = get_resting_events(subject, experiment, montage=montage)
