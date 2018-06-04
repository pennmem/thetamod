import os
from pathlib import Path
from pkg_resources import resource_filename
import pytest

import numpy as np
from numpy.testing import assert_equal

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
def test_resting_state_connectivity(rhino_root):
    subject = "R1286J"
    reader = CMLReader(subject, 'FR1', 0, rootdir=rhino_root)
    eegs = read_eeg_data(reader)
    conn = get_resting_state_connectivity(eegs.to_mne())

    basename = ('{subject}_baseline3trials_network_theta-alpha.npy'
                .format(subject=subject))
    filename = Path(rhino_root).joinpath('scratch', 'esolo', 'tmi_analysis',
                                         subject, basename)

    data = np.load(filename)
    pytest.set_trace()
    assert_equal(conn, data)
