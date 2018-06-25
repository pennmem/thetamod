import pickle as pkl
import pytest
import os.path

import numpy as np
from numpy.testing import assert_almost_equal
import thetamod.pipeline
import thetamod.artifact


def load_existing_results(subject, experiment, rhino_root):
    data_fname = os.path.join(rhino_root, 'scratch', 'esolo', 'tmi_analysis',
                              subject, 'tmi_output_{}.pk'.format(experiment)
                              )
    with open(data_fname, 'rb') as datafile:
        results = pkl.load(datafile, encoding='bytes')
    return {k.decode(): v for (k, v) in results.items()}


def load_existing_connectivity(subject,rhino_root):
    data_fname = os.path.join(rhino_root, 'scratch', 'esolo', 'tmi_analysis',
                              subject,
                              '{}_baseline3trials_network_theta-alpha.npy'.format(subject)
                              )
    return np.load(data_fname)


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
    def test_pipeline_equal(self, subject, experiment, session, rhino_root):
        existing_result = load_existing_results(subject,
                                                     experiment,
                                                     rhino_root)
        existing_tmi = get_tmi_by_session(existing_result, session)
        pipeline = thetamod.pipeline.TMIPipeline(subject,experiment,session,
                                                 rootdir=rhino_root)
        new_result = pipeline.run()[0]['zscore']
        assert_almost_equal(new_result, existing_tmi)

    @pytest.mark.parametrize(['subject', 'experiment'],
                             [('R1260D', 'catFR3'), ]
                             )
    @pytest.mark.rhino
    def test_pipeline_full(self, subject, experiment, rhino_root):
        from cml_pipelines import memory
        memory.clear(warn=False)
        existing_result = load_existing_results(subject,
                                                experiment,
                                                rhino_root)

        session = np.unique(existing_result['sessions'])[-1]
        session_mask = existing_result['sessions'] == session

        pipeline = thetamod.pipeline.TMIPipeline(subject, experiment, session,
                                                 rootdir=rhino_root)
        conn = pipeline.conn.compute()
        existing_conn = load_existing_connectivity(subject, rhino_root)
        # FIXME: Figure out what's wrong here
        assert_almost_equal(existing_conn, conn, 3)

        assert_almost_equal(existing_result['pvals'] < 0.0001,
                            pipeline.channel_exclusion_mask.compute())
        assert_almost_equal(existing_result['distmat'],
                            pipeline.distmat.compute())

        ethan_post_pows = existing_result['post_pows'][session_mask]
        ethan_pre_pows = existing_result['pre_pows'][session_mask]

        assert_almost_equal(np.isnan(ethan_post_pows),
                            pipeline.bad_events_mask.compute())

        assert_almost_equal(ethan_post_pows[~pipeline.bad_events_mask.compute()],
                            pipeline.post_psd.compute()[
                                ~pipeline.bad_events_mask.compute()
                            ])

        assert_almost_equal(ethan_pre_pows[~pipeline.bad_events_mask.compute()],
                            pipeline.pre_psd.compute()[
                                ~pipeline.bad_events_mask.compute()])

        existing_tstat = existing_result['T']
        tstat = pipeline.tstats.compute()
        assert_almost_equal(existing_tstat,
                            tstat)

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
        result_a = pipeline.run()[0]['zscore']
        pipeline = thetamod.pipeline.TMIPipeline(subject, experiment, session,
                                                 rootdir=rhino_root)
        result_b = pipeline.run()[0]['zscore']

        assert_almost_equal(result_a,result_b,decimal=2)

    @pytest.mark.rhino
    def test_multiple_stim_sites(self, rhino_root):
        subject = 'R1332M'
        experiment = 'PS4_catFR'
        session = 0
        pipeline = thetamod.pipeline.TMIPipeline(subject, experiment, session,
                                                 rootdir=rhino_root)
        results = pipeline.run()
        assert len(results) == 2

if __name__ == '__main__':
    from cml_pipelines import memory
    memory.clear(warn=False)
    rhino_root = os.environ['RHINO_ROOT']
    params = [
        ('R1195E', 'FR3', '0'),
        ('R1200T', 'FR3', '0'),
        ('R1204T', 'FR3', '0'),
        ('R1154D', 'FR3', '0'),
        ('R1161E', 'FR3', '0'),
        ('R1163T', 'FR3', '0'),
        ('R1166D', 'FR3', '0'),
        ('R1260D', 'catFR3', '0'),
        ('R1264P', 'catFR3', '0'),
        ('R1274T', 'catFR3', '0'),
    ]
    with open('pipeline_comparison.csv', 'w') as csv:
        print('subject', 'experiment','new_result', 'old_result', file=csv, sep='\t')
        for (subject, experiment, session) in params:
            try:
                new_result = thetamod.pipeline.TMIPipeline(
                    subject, experiment, session, rhino_root).run()[0]['zscore']
                old_result = get_tmi_by_session(
                    load_existing_results(subject, experiment, rhino_root),
                    int(session)
                )
                print(subject, experiment, new_result, old_result, file=csv, sep='\t')
            except Exception as e:
                import traceback as tb
                tb.print_exc()
                print(subject, experiment, file=csv, sep='\t')

