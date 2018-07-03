import pickle as pkl
import pytest
import os.path

from scipy.special import logit
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import thetamod.pipeline
import thetamod.artifact
import thetamod.tmi
from cml_pipelines import memory
memory.clear(warn=False)

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


def params_list():
    for x in  [
        ('R1154D', 'FR3', 4),
        # ('R1195E', 'FR3', 2),
        # ('R1200T', 'FR3', 0),
        ('R1204T', 'FR3', 0),
        ('R1161E', 'FR3', 1),
        # ('R1163T', 'FR3', 1),
        # ('R1166D', 'FR3', 1),
        # ('R1260D', 'catFR3', 2),
        ('R1274T', 'catFR3', 0),
    ]:
        yield x


@pytest.fixture(params=params_list(), ids=lambda x: x[0])
def pipeline(request, rhino_root):
    subj,exp,sess = request.param
    return thetamod.pipeline.TMIPipeline(
        subj, exp, sess, rootdir=rhino_root
    )


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

    @pytest.mark.parametrize("test_name",
                             ["coefficients",
                              "pre_pows",
                              "post_pows",
                              "events_mask",
                              "distmat",
                              "tstat",
                              "connectivity",
                              ])
    @pytest.mark.rhino
    def test_pipeline_full(self, test_name, pipeline, rhino_root):
        memory.clear(warn=False)

        subj = pipeline.subject
        exp = pipeline.experiment
        existing_result = load_existing_results(subj,
                                                exp,
                                                rhino_root)

        self.__getattribute__(test_name)(existing_result, pipeline)

    def tstat(self, existing_result, pipe):
        existing_tstat = existing_result['T']
        tstat = pipe.tstats.compute()
        assert_almost_equal(existing_tstat,
                            tstat)

    def pre_pows(self, existing_result, pipe):
        session_mask = existing_result['sessions'] == pipe.session

        ethan_pre_pows = existing_result['pre_pows'][session_mask]
        bad_events_mask = pipe.bad_events_mask.compute()
        ethan_bad_events_mask = np.isnan(ethan_pre_pows)
        assert_almost_equal(ethan_pre_pows[~(bad_events_mask | ethan_bad_events_mask)],
                       pipe.pre_psd.compute()[
                           ~(bad_events_mask | ethan_bad_events_mask)],)

    def post_pows(self, existing_result, pipe):
        session_mask = existing_result['sessions'] == pipe.session
        assert session_mask.any()

        ethan_post_pows = existing_result['post_pows'][session_mask]
        bad_events_mask =  pipe.bad_events_mask.compute()
        ethan_bad_events_mask  = np.isnan(ethan_post_pows)
        assert_almost_equal(ethan_post_pows[~(bad_events_mask | ethan_bad_events_mask)],
                            pipe.post_psd.compute()[
                                ~(bad_events_mask | ethan_bad_events_mask)
                            ])

    def events_mask(self,existing_result,pipe):
        session_mask = existing_result['sessions'] == pipe.session

        ethan_post_pows = existing_result['post_pows'][session_mask]
        bad_events_mask =  pipe.bad_events_mask.compute()
        ethan_bad_events_mask  = np.isnan(ethan_post_pows)
        assert_almost_equal(bad_events_mask,ethan_bad_events_mask)

    def distmat(self, existing_result, pipe):
        assert_almost_equal(existing_result['distmat'],
                       pipe.distmat.compute(),)

    def connectivity(self, existing_result, pipe):
        existing_conn = load_existing_connectivity(
            pipe.subject, pipe.rootdir)
        conn = pipe.conn.compute()
        assert_almost_equal(existing_conn,conn, 3)
        assert_almost_equal(logit(existing_conn), logit(conn), 3)

    def channel_exclusion_mask(self, existing_result,pipe):

        assert_almost_equal((existing_result['pvals'] < 0.0001).sum(),
                            pipe.channel_exclusion_mask.compute().sum())

    def coefficients(self,existing_result, pipe):
        new_coefs = pipe.regressions[0]['coefs'].compute()
        t = existing_result['T']
        tmask = ~np.isnan(t)
        _, old_coefs, _, _  = thetamod.tmi.do_regression(
            existing_result['conn'],
            existing_result['distmat'],
            existing_result['stim_elec_idx'][0],
            t,
            tmask
        )
        assert_allclose(old_coefs,new_coefs)

    def coefficient_mismatch(self,existing_result,pipe):
        memory.clear(warn=False)
        new_coefs = pipe.regressions[0]['coefs'].compute()
        t = existing_result['T']
        tmask = ~np.isnan(t)
        _, old_coefs, _, _  = thetamod.tmi.do_regression(
            existing_result['conn'],
            existing_result['distmat'],
            existing_result['stim_elec_idx'][0],
            t,
            tmask
        )
        differences = old_coefs-new_coefs
        assert_almost_equal(np.sign(differences).mean(),0,4)



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
    params = params_list()
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

