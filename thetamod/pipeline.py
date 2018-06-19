import functools
from typing import Optional

from dask.delayed import Delayed
import numpy as np

from cml_pipelines import make_task, memory
from cmlreaders import CMLReader, get_data_index
from cmlreaders.timeseries import TimeSeries

from . import connectivity, tmi, artifact


def clear_cache_on_completion(func):
    """Clears the joblib cache on completion of func."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        memory.clear(warn=False)
        return result
    return wrapper


class TMIPipeline(object):
    def __init__(self, subject: str, experiment: str, session: int,
                 rootdir: Optional[str] = None, nperms=1000):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.rootdir = rootdir
        self.nperms = nperms

        self._pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Delayed:
        """Build the pipeline."""
        reader = self.get_reader()
        pairs = reader.load("pairs")

        stim_events = make_task(tmi.get_stim_events, reader)
        stim_channels = make_task(tmi.get_stim_channels, pairs, stim_events)

        pre_eeg, post_eeg = [
            make_task(tmi.get_eeg, which, reader, stim_events, cache=False)
            for which in ("pre", "post")
        ]

        pre_eeg, post_eeg = make_task(
            artifact.invalidate_eeg, reader, pre_eeg, post_eeg,
            rhino_root=self.rootdir, nout=2)

        distmat = make_task(tmi.get_distances, pairs)

        pre_psd = make_task(tmi.compute_psd, pre_eeg)
        post_psd = make_task(tmi.compute_psd, post_eeg)

        conn = make_task(self.get_resting_connectivity)
        regressions = make_task(tmi.regress_distance,
                                pre_psd, post_psd,
                                conn, distmat, stim_channels)

        results = make_task(tmi.compute_tmi, regressions)

        return results

    def run_nodask(self):
        reader = self.get_reader()
        pairs = reader.load("pairs")

        stim_events = tmi.get_stim_events(reader)
        stim_channels = tmi.get_stim_channels(pairs, stim_events)
        conn = self.get_resting_connectivity()

        pre_eeg, post_eeg = (
            tmi.get_eeg(which, reader, stim_events)
            for which in ("pre", "post")
        )
        distmat = tmi.get_distances(pairs)
        pre_psd, post_psd = (tmi.compute_psd(eeg) for eeg in (pre_eeg, post_eeg))

        regressions = tmi.regress_distance(pre_psd, post_psd, conn,
                                           distmat, stim_channels, self.nperms)

        return tmi.compute_tmi(regressions)

    @clear_cache_on_completion
    def run(self, get=None):
        """Run the pipeline and return the results."""
        return self._pipeline.compute(get=get)

    def get_reader(self, subject: Optional[str] = None,
                   experiment: Optional[str] = None,
                   session: Optional[int] = None) -> CMLReader:
        """Return a reader for loading data. Defaults to the instance's subject,
        experiment, and session.

        """
        subject = subject if subject is not None else self.subject
        experiment = experiment if experiment is not None else self.experiment
        session = session if session is not None else self.session

        return CMLReader(subject, experiment, session, rootdir=self.rootdir)

    def get_resting_connectivity(self) -> np.ndarray:
        """Compute resting state connectivity."""
        df = get_data_index(rootdir=self.rootdir)
        sessions = df[(df.subject == self.subject) &
                      (df.experiment == "FR1")].session

        # Read EEG data for "resting" events
        eeg_data = []
        for session in sessions:
            reader = self.get_reader(experiment="FR1", session=session)
            rate = reader.load('sources')['sample_rate']
            reref = not reader.load('sources')['name'].endswith('.h5')
            events = connectivity.get_countdown_events(reader)
            resting = connectivity.countdown_to_resting(events, rate)
            eeg = connectivity.read_eeg_data(reader, resting, reref=reref)
            eeg_data.append(eeg)

        eegs = TimeSeries.concatenate(eeg_data)
        conn = connectivity.get_resting_state_connectivity(eegs.to_mne(),
                                                           eegs.samplerate)
        return conn
