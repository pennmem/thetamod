import pytest

from thetamod import tmi
from thetamod.test.fixtures import rhino_root  # noqa


@pytest.mark.rhino
@pytest.mark.parametrize("which", ["pre", "post"])
@pytest.mark.parametrize("subject,experiment,session,shape", [
    ("R1354E", "PS4_FR", 1, (604, 125, 900)),
])
def test_get_eeg(which, subject, experiment, session, shape, rhino_root):
    eeg = tmi.get_eeg(which, subject, experiment, session, rootdir=rhino_root)
    assert eeg.shape == shape
