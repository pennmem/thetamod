from pkg_resources import resource_filename
import pytest

from cmlreaders import CMLReader
from thetamod.connectivity import *


@pytest.fixture
def subject():
    return 'R1387E'


@pytest.fixture
def events_file_path():
    return resource_filename('thetamod.test.data', 'R1387E_FR1_0_events.json')


def test_get_resting_events(subject, events_file_path):
    reader = CMLReader(subject=subject, experiment='FR1', session=0)
    events = get_resting_events(reader, events_file_path).compute()
    assert len(events) == 13
    types = events.type.unique()
    assert len(types) == 1
    assert types[0] == 'COUNTDOWN_START'
