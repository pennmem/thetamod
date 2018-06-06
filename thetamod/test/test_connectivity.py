import os
from pathlib import Path
from pkg_resources import resource_filename
import pytest

import numpy as np
from numpy.testing import assert_equal
import pandas as pd

from cmlreaders import CMLReader, get_data_index
from cmlreaders.timeseries import TimeSeries

from thetamod.connectivity import *


@pytest.fixture
def rhino_root():
    return os.path.expanduser(os.environ.get('RHINO_ROOT', '/'))


@pytest.fixture
def subject():
    return 'R1387E'


@pytest.fixture
def events_file_path():
    return resource_filename('thetamod.test.data', 'R1387E_FR1_0_events.json')


@pytest.mark.rhino
def test_read_eeg(subject, rhino_root):
    reader = CMLReader(subject, 'FR1', session=0, rootdir=rhino_root)
    samplerate = reader.load('sources')['sample_rate']
    events = get_countdown_events(reader)
    resting = countdown_to_resting(events, samplerate)

    eeg = read_eeg_data(reader, resting, reref=False)

    # R1387E FR1 session 0 had 13 countdown start events and we get 3 epochs per
    # countdown
    expected_events = 13 * 3
    assert eeg.shape == (expected_events, 121, 1000)


@pytest.mark.rhino
def test_resting_state_connectivity(rhino_root):
    subject = "R1354E"

    index = get_data_index("r1", rhino_root)
    sessions = index[(index.subject == subject) &
                     (index.experiment == 'FR1')].session.values

    all_events = []
    all_resting = []
    epochs = []
    data = []

    for session in sessions:
        reader = CMLReader(subject, 'FR1', session, rootdir=rhino_root)
        events = get_countdown_events(reader)
        resting = countdown_to_resting(events, reader.load('sources')['sample_rate'])

        all_events.append(events)
        all_resting.append(resting)

        eeg = read_eeg_data(reader, resting, reref=False)
        epochs += eeg.epochs
        data.append(eeg)

    eegs = TimeSeries.concatenate(data)
    conn = get_resting_state_connectivity(eegs.to_mne())

    basename = ('{subject}_baseline3trials_network_theta-alpha.npy'
                .format(subject=subject))
    filename = Path(rhino_root).joinpath('scratch', 'esolo', 'tmi_analysis',
                                         subject, basename)

    data = np.load(filename)

    np.savez("test_output.npz",
             eeg=eegs.data,
             my_conn=conn,
             ethans_conn=data,
             events=pd.concat(all_events).to_records(),
             resting=pd.concat(all_resting).to_records(),
             epochs=epochs)

    assert_equal(conn, data)
