from functools import partial, wraps
from typing import Optional

from dask.delayed import Delayed
import numpy as np

from cml_pipelines import make_task
from cml_pipelines.wrapper import memory
from cmlreaders import CMLReader, get_data_index
from cmlreaders.timeseries import TimeSeries

from . import connectivity, tmi


def clear_cache_on_completion(func):
    """Clears the joblib cache on completion of func."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        memory.clear(warn=False)
        return result
    return wrapper


class TMIPipeline(object):
    def __init__(self, subject: str, experiment: str, session: int,
                 rootdir: Optional[str] = None):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.rootdir = rootdir

        self._pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Delayed:
        """Build the pipeline."""
        reader = self.get_reader()
        pairs = reader.load("pairs")

        stim_events = make_task(tmi.get_stim_events, reader)
        stim_channels = make_task(tmi.get_stim_channels, stim_events)

        get_eeg = partial(tmi.get_eeg,
                          reader=reader,
                          stim_events=stim_events,
                          reref=True)

        pre_eeg = make_task(get_eeg, "pre")
        post_eeg = make_task(get_eeg, "post")

        distmat = make_task(tmi.get_distances, pairs)

        pre_psd = make_task(tmi.compute_psd, pre_eeg)
        post_psd = make_task(tmi.compute_psd, post_eeg)

        conn = make_task(self.get_resting_connectivity,
                         self.subject, self.rootdir)
        regression = make_task(tmi.regress_distance,
                               pre_psd, post_psd, conn, distmat, stim_channels)

        result = make_task(tmi.compute_tmi, regression)

        return result

    @clear_cache_on_completion
    def run(self):
        """Run the pipeline and return the results."""
        return self._pipeline.compute()

    def get_reader(self, subject: Optional[str] = None,
                   experiment: Optional[str] = None,
                   session: Optional[int] = None) -> CMLReader:
        """Return a reader for loading data. Defaults to the instance's subject,
        experiment, and session.

        """
        subject = subject or self.subject
        experiment = experiment or self.experiment
        session = session or self.session

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
            events = connectivity.get_countdown_events(reader)
            resting = connectivity.countdown_to_resting(events, rate)
            eeg = connectivity.read_eeg_data(reader, resting, reref=True)
            eeg_data.append(eeg)

        eegs = TimeSeries.concatenate(eeg_data)
        conn = connectivity.get_resting_state_connectivity(eegs.to_mne())
        return conn
