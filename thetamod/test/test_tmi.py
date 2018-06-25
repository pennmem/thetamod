from functools import partial
from pkg_resources import resource_filename
import pytest
import pickle

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

from cmlreaders import CMLReader
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
    pairs = reader.load("pairs")
    events = tmi.get_stim_events(reader)
    channels = tmi.get_stim_channels(pairs, events)
    assert len(channels) == 1
    assert channels == [140]


@pytest.mark.rhino
@pytest.mark.parametrize("which", ["pre", "post"])
@pytest.mark.parametrize("subject,experiment,session,shape", [
    ("R1354E", "PS4_FR", 1, (604, 125, 900)),
])
def test_get_eeg(which, subject, experiment, session, shape, rhino_root):
    reader = CMLReader(subject, experiment, session, rootdir=rhino_root)
    all_events = reader.load("events")
    events = all_events[all_events.type == "STIM_ON"]

    eeg = tmi.get_eeg(which, reader, events)
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
    is_not_nan_pre = ~np.isnan(ethan["pre"])
    is_not_nan_post = ~np.isnan(ethan["post"])
    assert_allclose(ethan["pre"][is_not_nan_pre], pre_psd[is_not_nan_pre],
                    )
    assert_allclose(ethan["post"][is_not_nan_post], post_psd[is_not_nan_post])


def test_get_distances():
    pkg = "thetamod.test.data"

    filename = resource_filename(pkg, "R1260D_pairs.json")
    reader = CMLReader("R1260D")
    pairs = reader.load("pairs", file_path=filename).sort_values(
        by=['contact_1', 'contact_2'])

    filename = resource_filename(pkg, "R1260D_distmat.npy")
    ref_result = np.load(filename)
    distmat = tmi.get_distances(pairs)

    assert_almost_equal(distmat, ref_result)


def test_compute_tstat():
    pkg = 'thetamod.test.data'
    results_fname = resource_filename(pkg,'tmi_output_R1260D_catFR3.pk')
    with open(results_fname,'rb') as result_file:
        results = pickle.load(result_file, encoding='bytes')

    last_session = np.unique(results[b'sessions'])[-1]
    session_mask = results[b'sessions'] == last_session

    conn = results[b'conn']
    conn += np.finfo(conn.dtype).eps

    event_mask = np.isnan(results[b'pre_pows'][session_mask])
    channel_mask = results[b'pvals'] < 0.0001

    regressions, tstats = tmi.regress_distance(
        results[b'pre_pows'][session_mask],
        results[b'post_pows'][session_mask],
        conn,
        results[b'distmat'],
        results[b'stim_elec_idx'][:1],
        event_mask=event_mask,
        artifact_channels=channel_mask)

    old_tstat = results[b'T']
    assert_almost_equal(tstats, old_tstat)


def test_compute_tmi():
    pkg = 'thetamod.test.data'
    results_fname = resource_filename(pkg,'tmi_output_R1260D_catFR3.pk')
    with open(results_fname,'rb') as result_file:
        results = pickle.load(result_file, encoding='bytes')

    last_session = np.unique(results[b'sessions'])[-1]
    session_mask = results[b'sessions'] == last_session

    conn = np.load(resource_filename(
        pkg,'R1260D_baseline3trials_network_theta-alpha.npy'))

    conn += np.finfo(conn.dtype).eps

    event_mask = np.isnan(results[b'pre_pows'][session_mask])

    regressions, tstats = tmi.regress_distance(
        results[b'pre_pows'][session_mask],
        results[b'post_pows'][session_mask],
        conn,
        results[b'distmat'],
        results[b'stim_elec_idx'][:1],
        event_mask=event_mask)

    new_tmi = tmi.compute_tmi(regressions)[0]['zscore']

    old_tmi = results[b'tmi_Z'][-1]
    assert_almost_equal(new_tmi, old_tmi, decimal=1)


if __name__ == "__main__":
    test_compute_tstat()
