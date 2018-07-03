import functools
from typing import Optional

from dask.delayed import Delayed
import numpy as np
import dask

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
        self.pairs = reader.load("pairs").sort_values(by=["contact_1",
                                                          "contact_2"])

        self.stim_events = make_task(tmi.get_stim_events, reader)
        self.stim_channels = make_task(tmi.get_stim_channels,
                                       self.pairs, self.stim_events)

        pre_eeg, post_eeg = [
            make_task(tmi.get_eeg, which, reader, self.stim_events, cache=False)
            for which in ("pre", "post")
        ]

        self.bad_events_mask = make_task(artifact.get_bad_events_mask, post_eeg.data,
                                         self.stim_events)

        self.pre_psd = make_task(tmi.compute_psd, pre_eeg)
        self.post_psd = make_task(tmi.compute_psd, post_eeg)
        self.distmat = make_task(tmi.get_distances, self.pairs)
        self.conn = make_task(self.get_resting_connectivity)

        self.channel_exclusion_mask = make_task(
            artifact.get_channel_exclusion_mask, pre_eeg.data,
            post_eeg.data, pre_eeg.samplerate)

        self.regressions, self.tstats = make_task(
            tmi.regress_distance,
            self.pre_psd, self.post_psd,
            self.conn, self.distmat, self.stim_channels,
            event_mask=self.bad_events_mask,
            artifact_channels=self.channel_exclusion_mask,
            nout=2)

        results = make_task(tmi.compute_tmi, self.regressions)

        return results

    @clear_cache_on_completion
    def run(self, get=dask.get):
        """Run the pipeline and return the results."""
        return self._pipeline.compute(get=get)

    def get_reader(self, subject: Optional[str] = None,
                   experiment: Optional[str] = None,
                   session: Optional[int] = None) -> CMLReader:
        """Return a reader for loading data. Defaults to the instance's subject,
        experiment, and session.

        """
        idx = get_data_index('r1', self.rootdir)

        subject = subject if subject is not None else self.subject
        experiment = experiment if experiment is not None else self.experiment
        session = int(session if session is not None else self.session)

        montage = idx.loc[(idx.subject == subject)
                          & (idx.experiment == experiment)
                          & (idx.session == session)].montage.unique()[0]

        return CMLReader(subject, experiment, session,
                         montage=montage, rootdir=self.rootdir)

    def get_resting_connectivity(self) -> np.ndarray:
        """Compute resting state connectivity."""
        df = get_data_index(rootdir=self.rootdir)
        sessions = df[(df.subject == self.subject) &
                      (df.experiment == "FR1")].session.unique()

        if len(sessions) == 0:
            raise RuntimeError("No FR1 sessions exist for %s"%self.subject)
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
