from functools import partial
from pkg_resources import resource_filename
import pytest

import numpy as np
from numpy.testing import assert_equal

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
@pytest.mark.parametrize("which", ["pre", "post"])
@pytest.mark.parametrize("subject,experiment,session,shape", [
    ("R1354E", "PS4_FR", 1, (604, 125, 900)),
])
def test_get_eeg(which, subject, experiment, session, shape, rhino_root):
    eeg = tmi.get_eeg(which, subject, experiment, session, reref=False,
                      rootdir=rhino_root)
    assert eeg.shape == shape


# FIXME: add local test with smaller dataset
@pytest.mark.rhino
def test_compute_psd(rhino_root):
    ethan = np.load(resource_filename("thetamod.test.data",
                                      "R1260D_catFR3_psd.npz"))

    sessions = (0, 2)

    get_eeg = partial(tmi.get_eeg, subject="R1260D", experiment="catFR3",
                      rootdir=rhino_root)
    pre_eegs = TimeSeries.concatenate([get_eeg("pre", session=session)
                                       for session in sessions])
    post_eegs = TimeSeries.concatenate([get_eeg("post", session=session)
                                        for session in sessions])

    pre_psd = tmi.compute_psd(pre_eegs)
    post_psd = tmi.compute_psd(post_eegs)

    np.savez("test_output.npz",
             pre_psd=pre_psd,
             post_psd=post_psd,
             ethan_pre_psd=ethan["pre"],
             ethan_post_psd=ethan["post"])

    assert_equal(ethan["pre"], pre_psd)
    assert_equal(ethan["post"], post_psd)
