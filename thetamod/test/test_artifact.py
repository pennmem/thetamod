import pytest
import ethan.MNE_pipeline_refactored
import ethan.pred_stim_pipeline
import thetamod.artifact
from cmlreaders import CMLReader
from thetamod.tmi import get_stim_events, get_eeg


@pytest.mark.parametrize("just_bad",[True, False])
@pytest.mark.parametrize("subject, montage", [
    ('R1111M', 0),
])
def test_electrode_categories(subject, montage, just_bad, rhino_root):
    ethan_electrode_list = ethan.MNE_pipeline_refactored.exclude_bad(
        subject, montage, just_bad, rhino_root)
    ethan_electrode_list = [name for name in ethan_electrode_list if
                            name.upper()==name and name.isalnum()
                            and 'R1' not in name]
    new_electrode_list = thetamod.artifact.get_bad_channel_names(
        subject, montage, just_bad, rhino_root
    )
    assert len(ethan_electrode_list) == len(new_electrode_list)
    assert set(ethan_electrode_list) == set(new_electrode_list)


@pytest.mark.parametrize("subject, experiment", [
    ('R1286J', 'catFR3',),
    ('R1332M', 'PS4_catFR'),
])
def test_saturated_events(subject, experiment, rhino_root):
    reader = CMLReader(subject, experiment, session=0, rootdir=rhino_root)
    events = get_stim_events(reader)
    eeg = get_eeg('pre', reader, events).data
    ethan_artifact_mask = ethan.pred_stim_pipeline.find_sat_events(eeg)
    new_artifact_mask = thetamod.artifact.get_saturated_events_mask(eeg)
    assert new_artifact_mask.any()
    assert (ethan_artifact_mask == new_artifact_mask).all()

@pytest.mark.skip
def test_channel_exclusion_mask():
    full_eeg = get_eeg()
    ethan_p, ethan_lev_p = ethan.pred_stim_pipeline.channel_exclusion(full_eeg)
    pre_eeg, post_eeg = (get_eeg(which,...) for which in ('pre', 'post'))
    new_p, new_levp = thetamod.artifact.get_channel_exclusion_pvals(
        pre_eeg,post_eeg
    )
    assert (new_p == ethan_p).all()
    assert (new_levp == ethan_lev_p).all()


def test_invalidate_eeg(rhino_root):
    reader = CMLReader(subject='R1286J', experiment='catFR3', session=0,
                       rootdir=rhino_root)
    pairs = reader.load("pairs")

    stim_events = get_stim_events(reader)

    pre_eeg, post_eeg = (
        get_eeg(which, reader, stim_events)
        for which in ("pre", "post")
    )

    thetamod.artifact.invalidate_eeg(reader, pre_eeg, post_eeg, rhino_root)


if __name__ == "__main__":
    test_invalidate_eeg('/Volumes/rhino_root')
