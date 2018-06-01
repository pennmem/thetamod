import os
from pkg_resources import resource_filename
import pytest

import mne

from cmlreaders import CMLReader
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
    eeg = read_eeg_data(reader)

    # R1387E FR1 session 0 had 13 countdown start events and we get 3 epochs per
    # countdown
    expected_events = 13 * 3
    assert eeg.shape == (expected_events, 121, 1000)


@pytest.mark.rhino
def test_ptsa_to_mne(subject, rhino_root):
    eegs = []
    for session in [0, 1]:
        reader = CMLReader(subject, 'FR1', session, rootdir=rhino_root)
        eegs.append(read_eeg_data(reader))

    converted = ptsa_to_mne(eegs)
    assert isinstance(converted, mne.EpochsArray)
    assert converted.info['nchan'] == 121
    assert converted.get_data().shape == (78, 121, 1000)
