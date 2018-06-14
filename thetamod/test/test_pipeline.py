import pickle as pkl
import pytest
import os.path

import numpy as np
from numpy.testing import assert_almost_equal
import thetamod.pipeline


def load_existing_results(subject, experiment, rhino_root):
    data_fname = os.path.join(rhino_root, 'scratch', 'esolo', 'tmi_analysis',
                              subject, 'tmi_output_{}.pk'.format(experiment)
                              )
    with open(data_fname, 'rb') as datafile:
        results = pkl.load(datafile, encoding='bytes')
    return {k.decode(): v for (k, v) in results.items()}


def get_tmi_by_session(existing_result, session):
    existing_sessions = np.unique(existing_result['sessions']).tolist()
    existing_tmi = existing_result['tmi_Z'][
        existing_sessions.index(session)]
    return existing_tmi


class TestTMIPipline(object):

    @pytest.mark.parametrize(['subject', 'experiment', 'session'],
                             [('R1260D', 'catFR3', 0), ]
                             )
    @pytest.mark.rhino
    @pytest.mark.xfail(raises=AssertionError)
    def test_pipeline_nodask(self, subject, experiment, session, rhino_root):
        existing_result = load_existing_results(subject,
                                                     experiment,
                                                     rhino_root)
        existing_tmi = get_tmi_by_session(existing_result, session)
        pipeline = thetamod.pipeline.TMIPipeline(subject,experiment,session,
                                                 rootdir=rhino_root)
        new_result = pipeline.run_nodask()[0]['zscore']
        assert_almost_equal(new_result, existing_tmi)

    @pytest.mark.rhino
    @pytest.mark.parametrize(['subject', 'experiment', 'session'],
                             [('R1260D', 'catFR3', 0),
                              ('R1111M', 'FR2', 1), ]
                             )
    def test_pipeline_rhino(self, subject, experiment, session, rhino_root):
        thetamod.pipeline.TMIPipeline(subject, experiment, session,
                                      rhino_root).run()

    @pytest.mark.parametrize(['subject', 'experiment', 'session'],
                             [
                                 ('R1260D', 'catFR3', 0),
                             ]
                             )
    @pytest.mark.rhino
    @pytest.mark.xfail(raises=AssertionError)
    def test_pipeline_reproducible(self, subject, experiment, session, rhino_root):
        pipeline = thetamod.pipeline.TMIPipeline(subject,experiment,session,
                                                 rootdir=rhino_root)
        result_a = pipeline.run_nodask()[0]['zscore']
        pipeline = thetamod.pipeline.TMIPipeline(subject, experiment, session,
                                                 rootdir=rhino_root)
        result_b = pipeline.run_nodask()[0]['zscore']

        assert_almost_equal(result_a,result_b,decimal=2)

    @pytest.mark.rhino
    def test_multiple_stim_sites(self, rhino_root):
        subject = 'R1332M'
        experiment = 'PS4_catFR'
        session = 0
        pipeline = thetamod.pipeline.TMIPipeline(subject, experiment, session,
                                                 rootdir=rhino_root)
        results = pipeline.run_nodask()
        assert len(results) == 2

if __name__ == '__main__':
    rhino_root = os.environ['RHINO_ROOT']
    # TestTMIPipline().test_pipeline_rhino('R1260D', 'catFR3', 0, rhino_root)
    # TestTMIPipline().test_pipeline_nodask('R1260D', 'catFR3', 0, rhino_root)
    TestTMIPipline().test_multiple_stim_sites(rhino_root)

