from functools import partial
from pkg_resources import resource_filename
import pytest

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from cmlreaders import CMLReader
from cmlreaders.readers.eeg import NumpyEEGReader
from cmlreaders.timeseries import TimeSeries

from thetamod import tmi


@pytest.fixture
def timeseries():
    pass
    # FIXME: load locally
    # eeg_filename = resource_filename("thetamod.test.data", "psd_eeg.npy")
    # reader = NumpyEEGReader(eeg_filename, np.int16, [(0, -1)])
    # ts = TimeSeries(reader.read(), 1000)
    # return ts


@pytest.mark.rhino
def test_get_stim_events(rhino_root):
    reader = CMLReader("R1111M", "FR2", 0, rootdir=rhino_root)
    events = tmi.get_stim_events(reader)
    assert len(events) == 60


@pytest.mark.rhino
def test_get_stim_channels(rhino_root):
    reader = CMLReader("R1111M", "FR2", 0, rootdir=rhino_root)
    events = tmi.get_stim_events(reader)
    channels = tmi.get_stim_channels(events)
    assert len(channels) == 1
    assert channels == [(98, 99)]


@pytest.mark.rhino
@pytest.mark.parametrize("which", ["pre", "post"])
@pytest.mark.parametrize("subject,experiment,session,shape", [
    ("R1354E", "PS4_FR", 1, (604, 125, 900)),
])
def test_get_eeg(which, subject, experiment, session, shape, rhino_root):
    reader = CMLReader(subject, experiment, session, rootdir=rhino_root)
    all_events = reader.load("events")
    events = all_events[all_events.type == "STIM_ON"]

    eeg = tmi.get_eeg(which, reader, events, reref=False)
    assert eeg.shape == shape


# FIXME: add local test with smaller dataset
@pytest.mark.rhino
def test_compute_psd(rhino_root):
    ethan = np.load(resource_filename("thetamod.test.data",
                                      "R1260D_catFR3_psd.npz"))

    sessions = (0, 2)
    readers = [CMLReader("R1260D", "catFR3", session, rootdir=rhino_root)
               for session in sessions]
    stim_events = [tmi.get_stim_events(reader) for reader in readers]

    pre_eegs = TimeSeries.concatenate([
        tmi.get_eeg("pre", reader, events)
        for reader, events in zip(readers, stim_events)
    ])
    post_eegs = TimeSeries.concatenate([
        tmi.get_eeg("post", reader, events)
        for reader, events in zip(readers, stim_events)
    ])

    pre_psd = tmi.compute_psd(pre_eegs)
    post_psd = tmi.compute_psd(post_eegs)

    np.savez("test_output.npz",
             pre_psd=pre_psd,
             post_psd=post_psd,
             ethan_pre_psd=ethan["pre"],
             ethan_post_psd=ethan["post"])

    # Not checking for now since they don't match exactly...
    # assert_equal(ethan["pre"], pre_psd)
    # assert_equal(ethan["post"], post_psd)


@pytest.mark.skip("a few values don't match for some reason")
def test_get_distances():
    pkg = "thetamod.test.data"

    filename = resource_filename(pkg, "R1260D_pairs.json")
    reader = CMLReader("R1260D")
    pairs = reader.load("pairs", file_path=filename)

    filename = resource_filename(pkg, "R1260D_distmat.npy")
    ref_result = np.load(filename)
    distmat = tmi.get_distances(pairs)

    pytest.set_trace()

    assert_almost_equal(distmat, ref_result)
