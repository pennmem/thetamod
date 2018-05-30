#Import tools
import mne
import numpy as np
import pickle as pk
from os import mkdir
from glob import glob
from ptsa.data.readers import BaseEventReader
from ptsa.data.readers import TalReader
from ptsa.data.readers import EEGReader
from ptsa.data.filters import MonopolarToBipolarMapper
from ptsa.data.filters import ButterworthFilter
from ptsa.data.events import Events
from ptsa.data.readers import JsonIndexReader

import warnings
warnings.filterwarnings("ignore")

#Get pyFR subjects
ev_files = glob('/data/events/pyFR/*_events.mat')
pyFRsubs = [d.split('/')[-1].split('_ev')[0] for d in ev_files]
reader = JsonIndexReader('/protocols/r1.json')

MTL_labels = ['left ca1', 'left ca2', 'left ca3', 'left dg', 'left sub', 'left prc', 'left ec', 'left phc',
 'right ca1', 'right ca2', 'right ca3', 'right dg', 'right sub', 'right prc', 'right ec', 'right phc']

class mne_pipeline():
#Computing powers/connectivity using common average reference

    def __init__(self, s, win_st=0.0, win_fin=1.6, task='RAM_FR1', montage=0, notch_filter=True,
             samplerate=None, conn_type='coh', mode='multitaper', buf_time=1.0, n_cycles=6.0,
            root='/scratch/esolo/mne_analysis/',):
        self.s = s
        self.win_st = win_st
        self.win_fin = win_fin
        self.task = task
        self.samplerate = samplerate
        self.conn_type = conn_type
        self.mode = mode
        self.n_cycles = n_cycles
        self.root = root
        self.montage = montage
        self.buffer_time = buf_time
        self.notch_filter = notch_filter

        try:
            mkdir(root+s+'/')
        except:
            pass

    def set_freq_range(self, band='all_freqs'):

        #Define frequency bands
        freq_info = {
        'theta':[4., 8.],
        'theta_cwt': np.arange(4., 8.+1, 1),
        'alpha': [9., 13.],
        'alpha_cwt': np.arange(9., 13.+1, 1),
        'theta-alpha': np.arange(4, 14, 1),
        'theta-alpha_cwt': [4., 13.],
        'beta': [16., 28.],
        'beta_cwt': np.arange(16., 28.+1, 2),
        'lowgamma': [30., 60.],
        'lowgamma_cwt': np.arange(30., 60.+1, 5),
        'highgamma': [60., 120.],
        'highgamma_cwt': np.arange(70., 90.+1, 10),
        'hfa_cwt': np.array([75., 80., 85., 90., 95., 100.]), #np.logspace(np.log10(30.), np.log10(120.), num=25),
        'hfa': [30., 120.],
        'all_freqs': np.array([4., 5., 6., 7., 8.,
                                9., 10., 11., 12., 13.,
                                16., 18., 20., 22., 24., 26., 28.,
                                30., 32., 34., 36., 38., 40., 42., 44., 46., 48.,
                                50.]),
        'all_freqs_ext': np.array([4., 5., 6., 7., 8.,
                                 9., 10., 11., 12., 13.,
                                 16., 18., 20., 22., 24., 26., 28.,
                                 30., 32., 34., 36., 38., 40., 42., 44., 46., 48.,
                                 50., 55., 60., 65., 70., 75., 80.])
            }

        if self.mode=='cwt_morlet':
            if type(band) is str:
                self.cwtfreq = freq_info[band+'_cwt']
            else:
                cwtfreqs = []
                for b in band:
                    cwtfreqs.append(freq_info[b+'_cwt'])
                cwtfreqs = tuple(cwtfreqs)
                self.cwtfreq = np.hstack(cwtfreqs)

        if self.mode=='multitaper':
            self.freq = freq_info[band]

        self.band = band

        return

    def get_elec_regions(self, tal_struct):
        regs = []
        for e in tal_struct['atlases']:
            try:
                if 'stein' in list(e.keys()):
                    if (e['stein']['region'] is not None) and (len(e['stein']['region'])>1):
                        regs.append(e['stein']['region'].lower())
                        continue
                    else:
                        pass
                if 'das' in list(e.keys()):
                    if (e['das']['region'] is not None) and (len(e['das']['region'])>1):
                        regs.append(e['das']['region'].lower())
                        continue
                    else:
                        pass
                if 'ind' in list(e.keys()):
                    if (e['ind']['region'] is not None) and (len(e['ind']['region'])>1):
                        regs.append(e['ind']['region'].lower())
                        continue
                    else:
                        pass
                if 'wb' in list(e.keys()):
                    if (e['wb']['region'] is not None) and (len(e['wb']['region'])>1):
                        regs.append(e['wb']['region'].lower())
                        continue
                    else:
                        pass
                else:
                    regs.append('')
                    #print 'no atlas found'
            except AttributeError:
                regs.append('')
        return np.array(regs)

    def set_elec_info_avgref(self, good_elecs_only=True, MTL_only=False):

        self.MTL_only = MTL_only
        sub = self.s
        self.ref_scheme = 'AvgRef'
        self.good_elecs_only = good_elecs_only
        self.MTL_only = MTL_only

        #Get tal struct
#        try:
#            if self.montage!=0:
#                tal_path = '/data/eeg/'+sub+'_'+str(self.montage)+'/tal/'+sub+'_'+str(self.montage)+'_talLocs_database_monopol.mat'
#            else:
#                tal_path = '/data/eeg/'+sub+'/tal/'+sub+'_talLocs_database_monopol.mat'
#            tal_struct = TalReader(filename=tal_path, struct_name='talStruct').read()
#            tal_path_bipol = '/data/eeg/'+sub+'/tal/'+sub+'_talLocs_database_bipol.mat'
#            tal_reader = TalReader(filename=tal_path_bipol)
#            monopolar_channels = tal_reader.get_monopolar_channels()
#            bipolar_pairs = tal_reader.get_bipolar_pairs()
#            self.monopolar_channels = monopolar_channels
#        except IOError:
        try:
            tal_path = '/protocols/r1/subjects/'+sub+'/localizations/'+str(self.montage)+'/montages/'+str(self.montage)+'/neuroradiology/current_processed/contacts.json'
            tal_reader = TalReader(filename=tal_path, struct_type='mono')
            tal_struct = tal_reader.read()
            monopolar_channels = tal_reader.get_monopolar_channels()
            self.monopolar_channels = monopolar_channels
        except:
            print('Tal Struct could not be loaded!')
            tal_struct = np.nan
            self.monopolar_channels = np.nan
            bipolar_pairs = np.nan

        if good_elecs_only:
            bad_elecs = exclude_bad(sub, self.montage, just_bad=False)
            good_chan_idxs = []
            for idx, tagName in enumerate(tal_struct['tagName']):
                if str(tagName) in bad_elecs:
                    continue
                else:
                    good_chan_idxs.append(idx)
            self.tal_struct = tal_struct[good_chan_idxs]
            self.monopolar_channels = monopolar_channels[good_chan_idxs]
        else:
            self.tal_struct = tal_struct
            self.monopolar_channels = monopolar_channels

        if self.MTL_only:
            #Only use MTL electrodes
            regs = self.get_elec_regions(self.tal_struct)
            MTLelecs = [elecidx for elecidx, l in enumerate(regs) if l in MTL_labels]

            if len(MTLelecs) < 2:
                print('Less than 2 MTL electrodes!')
                raise ValueError

            MTL_info = self.tal_struct[MTLelecs]
            MTL_dict = {}
            MTL_dict['locTag'] = regs[MTLelecs]
            MTL_dict['tagName'] = list(self.tal_struct[MTLelecs]['tagName'])

            import pickle as pk
            try:
                pk.dump(MTL_dict, open(self.root+''+sub+'/MTL_info_AvgRef.pk', 'wb'))
                #np.save(self.root+''+sub+'/MTL_info_AvgRef.npy', MTL_info)
            except:
                os.mkdir(self.root+'/'+sub+'/')
                pk.dump(MTL_dict, open(self.root+''+sub+'/MTL_info_AvgRef.pk', 'wb'))
                #np.save(self.root+''+sub+'/MTL_info_AvgRef.npy', MTL_info)

            self.MTLelecs = MTLelecs
            self.MTL_info = MTL_info
            self.tal_struct = tal_struct[MTLelecs]
            self.monopolar_channels = monopolar_channels[MTLelecs]
        else:
            elec_info = self.tal_struct
            np.save(self.root+''+sub+'/elec_info_AvgRef.npy', elec_info)

        return

    def set_elec_info_bipolar(self, good_elecs_only=True, MTL_only=False):

        self.MTL_only = MTL_only
        sub = self.s
        self.ref_scheme = 'bipolar'
        self.good_elecs_only = good_elecs_only
        self.MTL_only = MTL_only

        #Get tal struct
        try:
            tal_path = '/protocols/r1/subjects/'+sub+'/localizations/'+str(self.montage)+'/montages/'+str(self.montage)+'/neuroradiology/current_processed/pairs.json'
            tal_reader = TalReader(filename=tal_path)
            tal_struct = tal_reader.read()
            monopolar_channels = tal_reader.get_monopolar_channels()
            bipolar_pairs = tal_reader.get_bipolar_pairs()
            self.monopolar_channels = monopolar_channels
        except:
            print('Tal Struct could not be loaded!')
            tal_struct = np.nan
            self.monopolar_channels = np.nan
            bipolar_pairs = np.nan

        if good_elecs_only:
            bad_elecs = exclude_bad(sub, self.montage, just_bad=True)
            good_chan_idxs = []
            for idx, tagName in enumerate(tal_struct['tagName']):
                if (str(tagName.split('-')[0]) in bad_elecs) or (str(tagName.split('-')[1]) in bad_elecs):
                    continue
                else:
                    good_chan_idxs.append(idx)
            self.tal_struct = tal_struct[good_chan_idxs]
            self.bipolar_pairs = bipolar_pairs[good_chan_idxs]
        else:
            self.tal_struct = tal_struct
            self.monopolar_channels = monopolar_channels

        if self.MTL_only:
            #Only use MTL electrodes
            regs = np.array(self.get_elec_regions(self.tal_struct))
            MTLelecs = [elecidx for elecidx, l in enumerate(regs) if l in MTL_labels]

            if len(MTLelecs) < 2:
                print('Less than 2 MTL electrodes!')
                raise ValueError

            MTL_info = self.tal_struct[MTLelecs]
            MTL_dict = {}
            MTL_dict['locTag'] = regs[MTLelecs]
            MTL_dict['tagName'] = self.tal_struct[MTLelecs]['tagName']

            import pickle as pk
            try:
                pk.dump(MTL_dict, open(self.root+''+sub+'/MTL_info_bipol.pk', 'wb'))
                #np.save(self.root+''+sub+'/MTL_info_bipol.npy', MTL_info)
            except:
                os.mkdir(self.root+'/'+sub+'/')
                pk.dump(MTL_dict, open(self.root+''+sub+'/MTL_info_bipol.pk', 'wb'))
                #np.save(self.root+''+sub+'/MTL_info_bipol.npy', MTL_info)

            self.MTLelecs = MTLelecs
            self.MTL_info = MTL_info
            self.tal_struct = tal_struct[MTLelecs]
            self.bipolar_pairs = bipolar_pairs[MTLelecs]
        else:
            self.tal_struct = tal_struct
            self.bipolar_pairs = bipolar_pairs
            elec_info = self.tal_struct
            np.save(self.root+''+sub+'/elec_info_bipol.npy', elec_info)

        return

    def set_events_encoding(self):

        sub = self.s
        self.task_phase = 'encoding'

        #Get available montages
        montages = reader.montages(subject=sub, experiment=self.task)

        #Set montage
        montage=str(self.montage)
        sessions = reader.sessions(subject=sub, experiment=self.task, montage=montage)

        #Get events
        evfiles = list(reader.aggregate_values('task_events', subject=sub, experiment=self.task, montage=montage))

        evs = np.array([])
        for ef in evfiles:
            base_e_reader = BaseEventReader(filename=ef, eliminate_events_with_no_eeg=True)
            base_events = base_e_reader.read()
            if len(evs) == 0:
                evs = base_events[base_events.type=='WORD']
            else:
                evs = np.concatenate((evs, base_events[base_events.type=='WORD']), axis=0)
        evs = Events(evs)
        self.evs = evs

        np.save(self.root+'/'+sub+'/Events_encoding_'+self.task+'.npy', evs['recalled'])

        return

    def set_events_resting(self):
        sub = self.s
        self.task_phase = 'resting'

        #Get available montages
        montages = reader.montages(subject=sub, experiment=self.task)

        #Set montage
        montage=str(self.montage)
        sessions = reader.sessions(subject=sub, experiment=self.task, montage=montage)

        #Get events
        evfiles = list(reader.aggregate_values('task_events', subject=sub, experiment=self.task, montage=montage))

        incl = ['COUNTDOWN_START']
        evs = np.array([])
        for ef in evfiles:
            base_e_reader = BaseEventReader(filename=ef, eliminate_events_with_no_eeg=True)
            base_events = base_e_reader.read()
            if len(evs) == 0:
                evs_mask = np.zeros(base_events.size)
                for idx, ev in enumerate(base_events):
                    if ev.type in incl:
                        evs_mask[idx] = 1
                evs = base_events[evs_mask.astype(bool)]
            else:
                evs_mask = np.zeros(base_events.size)
                for idx, ev in enumerate(base_events):
                    if ev.type in incl:
                        evs_mask[idx] = 1
                tmp = base_events[evs_mask.astype(bool)]
                evs = np.concatenate((evs, tmp), axis=0)
        evsize = evs.size
        evs = Events(np.concatenate((evs, evs, evs)))
        #evs = Events(evs)

        #Get samplerate to do eegoffsets
        from ptsa.data.readers import ParamsReader
        init_sr = ParamsReader(dataroot=evs[0].eegfile,
                          filename=evs[0].eegfile.split('noreref')[0]+'sources.json').read()['samplerate']

        evs[:evsize].eegoffset = evs[:evsize].eegoffset+int(init_sr*1.0) #Look at first quarter of countdown
        evs[evsize:evsize*2].eegoffset = evs[evsize:evsize*2].eegoffset+int(init_sr*4.0) #Look at last quarter of countdown
        evs[evsize*2:].eegoffset = evs[evsize*2:].eegoffset+int(init_sr*7.0)

        ##Old, buggy code##
        #evs[:evsize].mstime = evs[:evsize].mstime+int(init_sr*2.5) #Look at first quarter of countdown
        #evs[evsize:].mstime = evs[evsize:].mstime+int(init_sr*7.5) #Look at last quarter of countdown
        #####
        self.evs = evs

    def set_events_stimulation(self):

        sub = self.s
        self.task_phase = 'stim'

        #Get available montages
        montages = reader.montages(subject=sub, experiment=self.task)

        #Set montage
        montage=str(self.montage)
        sessions = reader.sessions(subject=sub, experiment=self.task, montage=montage)

        #Get events
        evfiles = list(reader.aggregate_values('task_events', subject=sub, experiment=self.task, montage=montage))

        evs_on = np.array([]); evs_off = np.array([])
        all_evs = np.array([])
        for ef in evfiles:
            base_e_reader = BaseEventReader(filename=ef, eliminate_events_with_no_eeg=True)
            base_events = base_e_reader.read()
            if len(evs_on) == 0:
                evs_on = base_events[base_events.type=='STIM_ON']
                evs_off = base_events[base_events.type=='STIM_OFF']
                all_evs = base_events[np.logical_or(base_events.type=='STIM_ON',
                                                   base_events.type=='STIM_OFF')]
            else:
                evs_on = np.concatenate((evs_on, base_events[base_events.type=='STIM_ON']), axis=0)
                evs_off = np.concatenate((evs_off, base_events[base_events.type=='STIM_OFF']), axis=0)
                all_evs = np.concatenate((all_evs, base_events[np.logical_or(base_events.type=='STIM_ON',
                                                   base_events.type=='STIM_OFF')]), axis=0)

        evs_on = Events(evs_on); evs_off = Events(evs_off)
        self.evs_on = evs_on; self.evs_off = evs_off

        #save out all events
        np.save(self.root+'/'+sub+'/Events_stimulation_'+self.task+'.npy', Events(all_evs))

    def set_mne_structure_stim(self):

        sub = self.s

        ##### Now post-stim #####
        eeg_reader = EEGReader(events=self.evs_on, channels=np.array([]),
                              start_time=-1.0, end_time=1.5, buffer_time=0.0)
        eegs = eeg_reader.read()
        eegs = eegs.baseline_corrected((-1.0, 2.5))
        self.eegs_raw = eegs

        if ('bipolar_pairs' in list(eegs.coords)):  #Gotta do this for bipolar ENS subjects
            bps_eeg = [list(i) for i in np.array(eegs['bipolar_pairs'])]
            bps_tal = [list(i) for i in np.array(self.tal_struct['channel'])]
            bps_eeg_good = []
            bps_tal_good = []
            for iidx, i in enumerate(bps_eeg):
                for jidx, j in enumerate(bps_tal):
                    if i == j:
                        bps_eeg_good.append(iidx)
                        bps_tal_good.append(jidx)
            bps_eeg_good = np.array(bps_eeg_good)
            bps_tal_good = np.array(bps_tal_good)

            eegs = eegs[bps_eeg_good]
            self.tal_struct = self.tal_struct[bps_tal_good]

            #Resave data
            np.save(self.root+''+sub+'/elec_info_bipol.npy', self.tal_struct)

        if (self.ref_scheme=='bipolar') and ('bipolar_pairs' not in list(eegs.coords)): #A non-BP ENS subject
            #Get bipolar timeseries
            m2b = MonopolarToBipolarMapper(time_series=eegs, bipolar_pairs=self.bipolar_pairs)
            eegs = m2b.filter()

        eegs_poststim = eegs

        self.eegs_poststim = eegs_poststim
        self.session_poststim = self.evs_off.session

    def set_events_retrieval(self):

        import sys
        sys.path.append('/home1/esolo/Logan_CML/CML_lib/')
        from RetrievalCreationHelper import create_matched_events, DeliberationEventCreator

        sub = self.s
        self.task_phase = 'retrieval'

        #Get available montages
        montages = reader.montages(subject=sub, experiment=self.task)

        #Just use the first montage for the task to avoid confusion
        montage=str(self.montage)
        sessions = reader.sessions(subject=sub, experiment=self.task, montage=montage)

        evs = np.array([])
        for sess in sessions:
            events = create_matched_events(subject=sub, experiment=self.task, session=int(sess),
                                       rec_inclusion_before = 2000, rec_inclusion_after = 1500,
                                       remove_before_recall = 2000, remove_after_recall = 2000,
                                       recall_eeg_start = -1000, recall_eeg_end = 0,
                                       match_tolerance = 2000, verbose=False, goodness_fit_check=False)
            if len(evs)==0:
                evs = events
            else:
                evs = np.concatenate((evs, events), axis=0)
        evs = Events(evs)
        self.evs = evs
        #np.save(self.root+'/'+sub+'/Events_retrieval_'+self.task+'.npy', evs.type=='REC_WORD')

        return

    def set_mne_structure_baseline(self):

        sub = self.s

        self.set_events_encoding()

        #Get EEGs
        eeg_reader_rec = EEGReader(events=self.evs[np.logical_and(self.evs.type=='WORD', self.evs.recalled==1)], channels=self.monopolar_channels,
                              start_time=0.5, end_time=1.5, buffer_time=self.buffer_time)
        base_eegs_rec = eeg_reader_rec.read()

        if self.ref_scheme=='bipolar':
            #Get bipolar timeseries
            m2b = MonopolarToBipolarMapper(time_series=base_eegs_rec, bipolar_pairs=self.bipolar_pairs)
            base_eegs_rec = m2b.filter()

        if self.notch_filter:
            #Filter out line noise
            if sub[0:2] == 'FR':
                freq_range = [48., 52.]
            else:
                freq_range = [58., 62.]
            b_filter_rec = ButterworthFilter(time_series=base_eegs_rec, freq_range=[58., 62.], filt_type='stop', order=4)
            eegs_filtered_rec = b_filter_rec.filter()

            if self.samplerate is not None:
                eegs_filtered_rec = eegs_filtered_rec.resampled(self.samplerate) #resample if needed
            else:
                eegs_filtered_rec = eegs_filtered_rec
        else:

            if self.samplerate is not None:
                eegs_filtered_rec = base_eegs_rec.resampled(self.samplerate)
            else:
                eegs_filtered_rec = base_eegs_rec

        self.set_events_resting()

        eeg_reader_base = EEGReader(events=self.evs, channels=self.monopolar_channels,
                              start_time=0.0, end_time=1.0, buffer_time=self.buffer_time)
        base_eegs_base = eeg_reader_base.read()

        if self.ref_scheme=='bipolar':
            #Get bipolar timeseries
            m2b = MonopolarToBipolarMapper(time_series=base_eegs_base, bipolar_pairs=self.bipolar_pairs)
            base_eegs_base = m2b.filter()

        if self.notch_filter:
            #Filter out line noise
            if sub[0:2] == 'FR':
                freq_range = [48., 52.]
            else:
                freq_range = [58., 62.]
            b_filter_base = ButterworthFilter(time_series=base_eegs_base, freq_range=[58., 62.], filt_type='stop', order=4)
            eegs_filtered_base = b_filter_base.filter()

            if self.samplerate is not None:
                eegs_filtered_base = eegs_filtered_base.resampled(self.samplerate) #resample if needed
            else:
                eegs_filtered_base = eegs_filtered_base
        else:

            if self.samplerate is not None:
                eegs_filtered_base = base_eegs_base.resampled(self.samplerate)
            else:
                eegs_filtered_base = base_eegs_base

        if self.samplerate is None:
            self.samplerate = int(base_eegs_rec['samplerate'])

        #Reorganize data for MNE format
        if self.ref_scheme=='bipolar':
            eegs_filtered_rec = eegs_filtered_rec.transpose('events', 'bipolar_pairs', 'time')
            eegs_filtered_base = eegs_filtered_base.transpose('events', 'bipolar_pairs', 'time')
        else:
            eegs_filtered_rec = eegs_filtered_rec.transpose('events', 'channels', 'time')
            eegs_filtered_base = eegs_filtered_base.transpose('events', 'channels', 'time')

        eegs_filtered = np.concatenate((eegs_filtered_rec, eegs_filtered_base), axis=0)
        rec_evs = np.hstack((np.ones(base_eegs_rec.shape[1]), np.zeros(base_eegs_base.shape[1])))

        #Create MNE dataset
        n_channels = eegs_filtered.shape[0]
        ch_names = list(self.tal_struct['tagName'])
        info = mne.create_info(ch_names, self.samplerate, ch_types='eeg')

        data = eegs_filtered

        #Create events array for MNE
        mne_evs = np.empty([rec_evs.size, 3]).astype(int)
        mne_evs[:, 0] = np.arange(rec_evs.size)
        mne_evs[:, 1] = data.shape[2]
        mne_evs[:, 2] = list(rec_evs)

        event_id = dict(recalled=1, not_recalled=0)
        tmin=0.0

        arr = mne.EpochsArray(np.array(data), info, mne_evs, tmin, event_id)

        if self.ref_scheme=='AvgRef':
            arr.set_eeg_reference(ref_channels=None) #Set to average reference
            arr.apply_proj()

        self.arr = arr
        self.session = sess_evs
        self.task_phase='baseline'

    def set_mne_structure_retrieval(self):

        sub = self.s

        #Get EEGs
        eeg_reader_rec = EEGReader(events=self.evs[self.evs.type=='REC_WORD'], channels=self.monopolar_channels,
                              start_time=self.win_st, end_time=self.win_fin, buffer_time=self.buffer_time)
        base_eegs_rec = eeg_reader_rec.read()

        if self.ref_scheme=='bipolar':
            #Get bipolar timeseries
            m2b = MonopolarToBipolarMapper(time_series=base_eegs_rec, bipolar_pairs=self.bipolar_pairs)
            base_eegs_rec = m2b.filter()

        if self.notch_filter:
            #Filter out line noise
            if sub[0:2] == 'FR':
                freq_range = [48., 52.]
            else:
                freq_range = [58., 62.]
            b_filter_rec = ButterworthFilter(time_series=base_eegs_rec, freq_range=[58., 62.], filt_type='stop', order=4)
            eegs_filtered_rec = b_filter_rec.filter()

            if self.samplerate is not None:
                eegs_filtered_rec = eegs_filtered_rec.resampled(self.samplerate) #resample if needed
            else:
                eegs_filtered_rec = eegs_filtered_rec
        else:

            if self.samplerate is not None:
                eegs_filtered_rec = base_eegs_rec.resampled(self.samplerate)
            else:
                eegs_filtered_rec = base_eegs_rec

        eeg_reader_base = EEGReader(events=self.evs[self.evs.type=='REC_BASE'], channels=self.monopolar_channels,
                              start_time=self.win_st, end_time=self.win_fin, buffer_time=self.buffer_time)
        base_eegs_base = eeg_reader_base.read()

        if self.ref_scheme=='bipolar':
            #Get bipolar timeseries
            m2b = MonopolarToBipolarMapper(time_series=base_eegs_base, bipolar_pairs=self.bipolar_pairs)
            base_eegs_base = m2b.filter()

        if self.notch_filter:
            #Filter out line noise
            if sub[0:2] == 'FR':
                freq_range = [48., 52.]
            else:
                freq_range = [58., 62.]
            b_filter_base = ButterworthFilter(time_series=base_eegs_base, freq_range=[58., 62.], filt_type='stop', order=4)
            eegs_filtered_base = b_filter_base.filter()

            if self.samplerate is not None:
                eegs_filtered_base = eegs_filtered_base.resampled(self.samplerate) #resample if needed
            else:
                eegs_filtered_base = eegs_filtered_base
        else:

            if self.samplerate is not None:
                eegs_filtered_base = base_eegs_base.resampled(self.samplerate)
            else:
                eegs_filtered_base = base_eegs_base

        if self.samplerate is None:
            self.samplerate = int(base_eegs_rec['samplerate'])

        #Reorganize data for MNE format
        if self.ref_scheme=='bipolar':
            eegs_filtered_rec = eegs_filtered_rec.transpose('events', 'bipolar_pairs', 'time')
            eegs_filtered_base = eegs_filtered_base.transpose('events', 'bipolar_pairs', 'time')
        else:
            eegs_filtered_rec = eegs_filtered_rec.transpose('events', 'channels', 'time')
            eegs_filtered_base = eegs_filtered_base.transpose('events', 'channels', 'time')

        eegs_filtered = np.concatenate((eegs_filtered_rec, eegs_filtered_base), axis=0)
        rec_evs = np.hstack((np.ones(base_eegs_rec.shape[1]), np.zeros(base_eegs_base.shape[1])))
        sess_evs = np.hstack((self.evs[self.evs.type=='REC_WORD']['session'], self.evs[self.evs.type=='REC_BASE']['session']))
        np.save(self.root+'/'+sub+'/Events_retrieval_'+self.task+'.npy', rec_evs)

        #Create MNE dataset
        n_channels = eegs_filtered.shape[0]
        ch_names = list(self.tal_struct['tagName'])
        info = mne.create_info(ch_names, self.samplerate, ch_types='eeg')

        data = eegs_filtered

        #Create events array for MNE
        mne_evs = np.empty([rec_evs.size, 3]).astype(int)
        mne_evs[:, 0] = np.arange(rec_evs.size)
        mne_evs[:, 1] = data.shape[2]
        mne_evs[:, 2] = list(rec_evs)

        event_id = dict(recalled=1, not_recalled=0)
        tmin=0.0

        arr = mne.EpochsArray(np.array(data), info, mne_evs, tmin, event_id)

        if self.ref_scheme=='AvgRef':
            arr.set_eeg_reference(ref_channels=None) #Set to average reference
            arr.apply_proj()

        self.arr = arr
        self.session = sess_evs

    def set_mne_structure_encoding(self, channels=np.array([])):

        sub = self.s

        #Get EEGs
        eeg_reader = EEGReader(events=self.evs, channels=channels,
                              start_time=self.win_st, end_time=self.win_fin, buffer_time=self.buffer_time)

        if self.tal_struct is np.nan:  #we refused to load a tal_struct for this subject, skip everything!
            eegs = eeg_reader.read()
            if eegs.shape[0]==0: #this subject needs monopolar channels to load data
                if self.ref_scheme=='AvgRef':
                    self.set_elec_info_avgref(good_elecs_only=self.good_elecs_only, MTL_only=self.MTL_only)
                if self.ref_scheme=='bipolar':
                    self.set_elec_info_bipolar(good_elecs_only=self.good_elecs_only, MTL_only=self.MTL_only)
                eeg_reader = EEGReader(events=self.evs, channels=self.monopolar_channels,
                              start_time=self.win_st, end_time=self.win_fin, buffer_time=self.buffer_time)
                eegs = eeg_reader.read()
            else:
                pass
        else:
            try:
                eegs = eeg_reader.read()
                if eegs.shape[0]==0:  #this subject needs monopolar channels to load data, even if we asked for all of them
                    if self.ref_scheme=='AvgRef':
                        self.set_elec_info_avgref(good_elecs_only=self.good_elecs_only, MTL_only=self.MTL_only)
                    if self.ref_scheme=='bipolar':
                        self.set_elec_info_bipolar(good_elecs_only=self.good_elecs_only, MTL_only=self.MTL_only)
                    eeg_reader = EEGReader(events=self.evs, channels=self.monopolar_channels,
                              start_time=self.win_st, end_time=self.win_fin, buffer_time=self.buffer_time)
                    eegs = eeg_reader.read()
                else:
                    pass
            except TypeError:
                for evidx in range(len(self.evs)):
                    if evidx==0:
                        eegs = EEGReader(events=self.evs[evidx:evidx+1], channels=self.monopolar_channels,
                                      start_time=self.win_st, end_time=self.win_fin, buffer_time=self.buffer_time).read()
                        sr_to_use = eegs['samplerate']
                    else:
                        new_eeg = EEGReader(events=self.evs[evidx:evidx+1], channels=self.monopolar_channels,
                                      start_time=self.win_st, end_time=self.win_fin, buffer_time=self.buffer_time).read()
                        new_eeg['samplerate'] = sr_to_use
                        eegs = eegs.append(new_eeg, dim='events')

            if ('bipolar_pairs' in list(eegs.coords)):  #Gotta do this for bipolar ENS subjects
                bps_eeg = [list(i) for i in np.array(eegs['bipolar_pairs'])]
                bps_tal = [list(i) for i in np.array(self.tal_struct['channel'])]
                bps_eeg_good = []
                bps_tal_good = []
                for iidx, i in enumerate(bps_eeg):
                    for jidx, j in enumerate(bps_tal):
                        if i == j:
                            bps_eeg_good.append(iidx)
                            bps_tal_good.append(jidx)
                bps_eeg_good = np.array(bps_eeg_good)
                bps_tal_good = np.array(bps_tal_good)

                eegs = eegs[bps_eeg_good]
                self.tal_struct = self.tal_struct[bps_tal_good]

                #Resave data
                np.save(self.root+''+sub+'/elec_info_bipol.npy', self.tal_struct)

        if (self.ref_scheme=='bipolar') and ('bipolar_pairs' not in list(eegs.coords)): #A non-BP ENS subject
            #Get bipolar timeseries
            m2b = MonopolarToBipolarMapper(time_series=eegs, bipolar_pairs=self.bipolar_pairs)
            eegs = m2b.filter()

        if self.notch_filter:
            #Filter out line noise
            if sub[0:2] == 'FR':
                freq_range = [48., 52.]
            else:
                freq_range = [58., 62.]
            b_filter = ButterworthFilter(time_series=eegs, freq_range=freq_range, filt_type='stop', order=4)
            eegs_filtered = b_filter.filter()

            if self.samplerate is not None:
                eegs_filtered = eegs_filtered.resampled(self.samplerate) #resample if needed
            else:
                eegs_filtered = eegs_filtered
        else:

            if self.samplerate is not None:
                eegs_filtered = eegs.resampled(self.samplerate)
            else:
                eegs_filtered = eegs

        if self.samplerate is None:
            self.samplerate = int(eegs['samplerate'])

        #Create MNE dataset
        n_channels = eegs_filtered.shape[0]
        if self.tal_struct is not np.nan:
            ch_names = list(self.tal_struct['tagName'])
        else:
            ch_names = [str(i) for i in range(n_channels)]
        info = mne.create_info(ch_names, self.samplerate, ch_types='eeg')

        #Reorganize data for MNE format
        if self.ref_scheme=='bipolar':
            data = eegs_filtered.transpose('events', 'bipolar_pairs', 'time')
        else:
            try:
                data = eegs_filtered.transpose('events', 'channels', 'time')
            except:
                data = eegs_filtered.transpose('events', 'bipolar_pairs', 'time')

        #Create events array for MNE
        mne_evs = np.empty([data['events'].shape[0], 3]).astype(int)
        mne_evs[:, 0] = np.arange(data['events'].shape[0])
        mne_evs[:, 1] = data['time'].shape[0]
        mne_evs[:, 2] = list(self.evs.recalled)

        if self.task_phase=='resting':
            event_id = dict(resting=0)
        else:
            event_id = dict(recalled=1, not_recalled=0)
        tmin=0.0

        #if np.sum(self.evs.recalled)<20:
        #    print 'Fewer than 20 recalled events!'
        #    return

        arr = mne.EpochsArray(np.array(data), info, mne_evs, tmin, event_id)

        if self.ref_scheme=='AvgRef':
            arr.set_eeg_reference(ref_channels=None) #Set to average reference
            arr.apply_proj()

        self.arr = arr
        self.session = self.evs.session

        return

    def mne_pipeline_powers(self, match_trial_n=False, threshold_n=False):
        #Can only be done with self.mode=='cwt_morlet'

        arr = self.arr
        cwtfreq = self.cwtfreq

        if self.mode!='cwt_morlet':
            raise ValueError('Mode must be cwt_morlet for powers!')

        from mne.time_frequency import tfr_array_morlet
        sfreq = arr.info['sfreq']  # the sampling frequency
        powers = tfr_array_morlet(arr, sfreq=sfreq, frequencies=cwtfreq, n_cycles=self.n_cycles,
                                  output='power', verbose=True)

        powers = np.log10(powers) #take log for low-power decay at high freqs

        #z-score by sessions
        from scipy.stats import zscore
        for s in np.unique(self.session):
            #z-score powers by session
            vals = powers[self.session==s, :, :, :]
            valsz = zscore(vals, axis=0, ddof=1)
            powers[self.session==s, :, :, :] = valsz

        if self.MTL_only:
            np.save(self.root+'/'+self.s+'/'+self.s+'_powers_MNE_'+self.ref_scheme+'_MTL_'+self.task_phase+'_'+str(self.win_st)+'-'+str(self.win_fin)+'_'+str(self.band)+'.npy',
                powers)
        else:
            np.save(self.root+'/'+self.s+'/'+self.s+'_powers_MNE_'+self.ref_scheme+'_'+self.task_phase+'_'+str(self.win_st)+'-'+str(self.win_fin)+'_'+str(self.band)+'.npy',
                powers)

    def mne_pipeline_powers_stim(self):

        from ptsa.data.filters.MorletWaveletFilter import MorletWaveletFilter

        sessions = self.session_prestim #same for pre- and post-stim

        for i in range(2):
            if i == 0:
                eegs = self.eegs_prestim
            else:
                eegs = self.eegs_poststim

            if self.mode!='cwt_morlet':
                raise ValueError('Mode must be cwt_morlet for powers!')

#             from mne.time_frequency import tfr_array_morlet
#             sfreq = arr.info['sfreq']  # the sampling frequency
#             powers = tfr_array_morlet(arr, sfreq=sfreq, frequencies=self.cwtfreq, n_cycles=self.n_cycles,
#                                       output='power', verbose=True)

            wf = MorletWaveletFilter(time_series=eegs,
                                     freqs=self.cwtfreq, width=5, output='power')
            pow_wavelet, _ = wf.filter()
            np.log10(pow_wavelet.data, out=pow_wavelet.data) #take log for low-power decay at high freqs
            pow_wavelet = pow_wavelet.transpose("events","bipolar_pairs","frequency","time") #transpose for easier computations
            powers = pow_wavelet

            if self.MTL_only:
                powers = powers[:, self.MTLelecs, :, :]
            else:
                powers = powers

            #Remove mirrored buffer
            if i == 0:
                #powers = powers[:, :, :, int(self.samplerate*0.5):-int(self.samplerate*0.5)]
                #powers = powers.remove_buffer(0.5)
                pass
            else:
                #powers = powers[:, :, :, int(self.samplerate*0.1):-int(self.samplerate*0.1)]
                #powers = powers.remove_buffer(1.0)
                pass

            powers = np.array(np.nanmean(powers.resampled(self.samplerate), 2)) #Average over frequencies because these files are big

            if i == 0:
                np.save(self.root+'/'+self.s+'/'+self.s+'_powers_MNE_'+self.ref_scheme+'_prestim_'+str(self.band)+'.npy',
                        powers)
            else:
                np.save(self.root+'/'+self.s+'/'+self.s+'_powers_MNE_'+self.ref_scheme+'_poststim_'+str(self.band)+'.npy',
                        powers)

    def mne_pipeline_phases(self, match_trial_n=False, threshold_n=False):
        #Can only be done with self.mode=='cwt_morlet'

        arr = self.arr
        cwtfreq = self.cwtfreq

        if self.mode!='cwt_morlet':
            raise ValueError('Mode must be cwt_morlet for powers!')

        from mne.time_frequency import tfr_array_morlet
        sfreq = arr.info['sfreq']  # the sampling frequency
        phases = tfr_array_morlet(arr, sfreq=sfreq, frequencies=cwtfreq, n_cycles=self.n_cycles,
                                  output='phase', verbose=True)

        if self.MTL_only:
            np.save(self.root+'/'+self.s+'/'+self.s+'_phases_'+self.ref_scheme+'_MTL_'+self.task_phase+'_'+str(self.win_st)+'-'+str(self.win_fin)+'_'+str(self.band)+'.npy',
                phases)
        else:
            np.save(self.root+'/'+self.s+'/'+self.s+'_phases_'+self.ref_scheme+'_'+self.task_phase+'_'+str(self.win_st)+'-'+str(self.win_fin)+'_'+str(self.band)+'.npy',
                phases)

        return

    def get_task_networks_coh(self,):
        arr = self.arr
        evs = self.evs
        sub = self.s

        #Prep file name
        import os
        fn = self.root+''+self.s+'/'+self.s+'_mne_task_'+self.ref_scheme+'_'+str(self.conn_type)+'_'+self.mode+'_'+str(self.win_st)+'-'+str(self.win_fin)+'_'+str(self.band)+'.npy'

        #Assess connectivity
        #fmin, fmax = self.freq[0], self.freq[-1]
        fmin = (4., 9., 15., 30., 70.); fmax = (8., 13., 25., 50., 90.)
        #fmin = (5., 15., 30., 70.); fmax = (15., 25., 40., 80.)
        sfreq = arr.info['sfreq']  # the sampling frequency
        tmin = 0.0

        #rec_idxs = np.arange(len(arr['recalled']))
        #nrec_idxs = np.arange(len(arr['not_recalled']))

        cons_rec = []
        for idx in range(len(arr)):
            con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
                arr[idx], method=self.conn_type, mode=self.mode, sfreq=sfreq, fmin=fmin, fmax=fmax,
                faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1, verbose=False,)

        #con = con[:, :, 0] #no need for last dimension
            cons_rec.append(con)
            #print idx

        cons_rec = np.array(cons_rec)
        #cons_rec = con

        try:
            np.save(fn, cons_rec)
        except:
            try:
                os.mkdir(self.root+'/'+sub+'/')
                np.save(fn, cons_rec)
            except:
                 np.save(fn, cons_rec)

        #Symmetrize and save average network
        #mu = np.mean(cons_rec, 0)
        #mu = cons_rec
        #mu_full = np.nansum(np.array([mu, mu.T]), 0)
        #mu_full[np.diag_indices_from(mu_full)] = 0.0
        #np.save(self.root+''+self.s+'/'+self.s+'_baseline_network_'+str(self.band)+'.npy', mu_full)

        return

    def get_resting_networks_coh(self, ):
        arr = self.arr
        evs = self.evs
        sub = self.s

        #Assess connectivity
        fmin, fmax = self.freq[0], self.freq[-1]
        sfreq = arr.info['sfreq']  # the sampling frequency
        tmin = 0.0

        cons_rec, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
            arr, method=self.conn_type, mode=self.mode, sfreq=sfreq, fmin=fmin, fmax=fmax,
            faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1, verbose=False,)

        cons_rec = cons_rec[:, :, 0]
        cons_rec = np.array(cons_rec)

        #Symmetrize and save average network
        mu = np.mean(cons_rec, 0)
        mu = cons_rec
        mu_full = np.nansum(np.array([mu, mu.T]), 0)
        mu_full[np.diag_indices_from(mu_full)] = 0.0
        np.save(self.root+''+self.s+'/'+self.s+'_baseline3trials_network_'+str(self.band)+'.npy', mu_full)

        return

    def get_coh(self,):

        arr = self.arr
        evs = self.evs
        sub = self.s

        #Prep file name
        import os
        fn = self.root+''+self.s+'/'+self.s+'_mne_'+self.ref_scheme+'_'+str(self.conn_type)+'_'+self.mode+'_'+str(self.win_st)+'-'+str(self.win_fin)+'_'+str(self.band)+'.npy'

        #Assess connectivity
        fmin, fmax = self.freq[0], self.freq[-1]
        sfreq = arr.info['sfreq']  # the sampling frequency
        tmin = 0.0

        rec_idxs = np.arange(len(arr['recalled']))
        nrec_idxs = np.arange(len(arr['not_recalled']))

        cons_rec = []
        for idx in range(len(arr['recalled'])):
            con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
                arr['recalled'][idx], method=self.conn_type, mode=self.mode, sfreq=sfreq, fmin=fmin, fmax=fmax,
                faverage=False, tmin=tmin, mt_adaptive=False, n_jobs=1, verbose=False, mt_bandwidth=None)

            #con = con[:, :, 0] #no need for last dimension
            cons_rec.append(con)
            #print idx

        cons_rec = np.array(cons_rec)

        cons_nrec = []
        for idx in range(len(arr['not_recalled'])):
            con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
                arr['not_recalled'][idx], method=self.conn_type, mode=self.mode, sfreq=sfreq, fmin=fmin, fmax=fmax,
                faverage=False, tmin=tmin, mt_adaptive=False, n_jobs=1, verbose=False, mt_bandwidth=None)

            #con = con[:, :, 0] #no need for last dimension
            cons_nrec.append(con)
            #print idx

        np.save(self.root+''+self.s+'/taper_freqs_'+str(self.win_st)+'-'+str(self.win_fin)+'.npy', freqs)
        cons_nrec = np.array(cons_nrec)

        try:
            np.save(fn, np.array([cons_rec, cons_nrec]))
        except:
            try:
                os.mkdir(self.root+'/'+sub+'/')
                np.save(fn, np.array([cons_rec, cons_nrec]))
            except:
                np.save(fn, np.array([cons_rec, cons_nrec]))


    def get_resting_networks_trials(self,):
        arr = self.arr
        evs = self.evs
        sub = self.s

        #Assess connectivity
        fmin, fmax = self.freq[0], self.freq[-1]
        sfreq = arr.info['sfreq']  # the sampling frequency
        tmin = 0.0

        cons = []
        for idx in range(len(arr)):
            cons_res, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
                arr[idx], method=self.conn_type, mode=self.mode, sfreq=sfreq, fmin=fmin, fmax=fmax,
                faverage=False, tmin=tmin, mt_adaptive=False, n_jobs=1, verbose=False,)

            cons.append(cons_res)
        cons = np.array(cons)

        #Prep file name
        import os
        fn = self.root+''+self.s+'/'+self.s+'_mne_resting_'+self.ref_scheme+'_'+str(self.conn_type)+'_'+self.mode+'_'+str(self.win_st)+'-'+str(self.win_fin)+'_'+str(self.band)+'.npy'

        try:
            np.save(fn, cons)
        except:
            try:
                os.mkdir(self.root+'/'+sub+'/')
                np.save(fn, cons)
            except:
                np.save(fn, cons)

    def mne_pipeline_PLI(self, nRandomizations=500, match_trial_n=False, threshold_n=False, avg_epochs=False):

        arr = self.arr
        evs = self.evs
        cwtfreq = self.cwtfreq
        sub = self.s

        #Prep file name
        import os
        if not avg_epochs:
            if self.MTL_only:
                fn = self.root+''+self.s+'/'+self.s+'_mne_MTL_dstat_'+self.ref_scheme+'_'+self.task_phase+'_'+str(self.conn_type)+'_'+self.mode+'_'+str(self.win_st)+'-'+str(self.win_fin)+'_'+str(self.band)+'.npy'
            else:
                fn = self.root+''+self.s+'/'+self.s+'_mne_dstat_'+self.ref_scheme+'_'+self.task_phase+'_'+str(self.conn_type)+'_'+self.mode+'_'+str(self.win_st)+'-'+str(self.win_fin)+'_'+str(self.band)+'.npy'
        else:
            if self.MTL_only:
                fn = self.root+''+self.s+'/'+self.s+'_mne_MTL_dstat_averaged_'+self.ref_scheme+'_'+self.task_phase+'_'+str(self.conn_type)+'_'+self.mode+'_'+str(self.win_st)+'-'+str(self.win_fin)+'_'+str(self.band)+'.npy'
            else:
                fn = self.root+''+self.s+'/'+self.s+'_mne_dstat_averaged_'+self.ref_scheme+'_'+self.task_phase+'_'+str(self.conn_type)+'_'+self.mode+'_'+str(self.win_st)+'-'+str(self.win_fin)+'_'+str(self.band)+'.npy'

        if os.path.isfile(fn):
            #print 'Subject run already!'
            #return
            pass
        else:
            pass

#         if self.MTL_only:
#             import itertools
#             tmp = list(itertools.combinations(self.MTLelecs, 2))
#             list1 = [i[0] for i in tmp]
#             list2 = [i[1] for i in tmp]
#             MTLinfo1 = [self.tal_struct[i[0]] for i in tmp]
#             MTLinfo2 = [self.tal_struct[i[1]] for i in tmp]
#             conn_inds = (list1, list2)
#             np.save(self.root+''+self.s+'/MTL_combs_'+self.ref_scheme+'.npy', [np.array([MTLinfo1[idx]['locTag'] for idx in range(0, len(MTLinfo1))]),
#                     np.array([MTLinfo2[idx]['locTag'] for idx in range(0, len(MTLinfo2))])])

        #Assess connectivity
        fmin, fmax = self.cwtfreq[0], self.cwtfreq[-1]
        sfreq = arr.info['sfreq']  # the sampling frequency
        tmin = 0.0

        rec_idxs = np.arange(len(arr['recalled']))
        nrec_idxs = np.arange(len(arr['not_recalled']))
        if match_trial_n is True:

            #If more remembered than not rememebred, shrink to not-remembered size with random shuffle
            if rec_idxs.size > nrec_idxs.size:
                np.random.shuffle(rec_idxs)
                rec_idxs = rec_idxs[:nrec_idxs.size]

            if nrec_idxs.size > rec_idxs.size:
                np.random.shuffle(nrec_idxs)
                nrec_idxs = nrec_idxs[:rec_idxs.size]

        if threshold_n is True:
            if len(rec_idxs) < 30 or len(nrec_idxs) < 30:
                raise ValueError('Insufficient Trial Count!')
                return

        print('Computing for '+str(rec_idxs.size)+' remembered and '+str(nrec_idxs.size)+' not-rememebred events.')

        cons_rec, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(arr['recalled'][rec_idxs],
            method=self.conn_type, mode=self.mode, sfreq=sfreq, fmin=fmin, fmax=fmax,
            faverage=False, n_jobs=1, verbose=True, cwt_frequencies=cwtfreq, cwt_n_cycles=self.n_cycles)

        cons_nrec, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
            arr['not_recalled'][nrec_idxs], method=self.conn_type, mode=self.mode, sfreq=sfreq, fmin=fmin, fmax=fmax,
            faverage=False, n_jobs=1, verbose=True, cwt_frequencies=cwtfreq, cwt_n_cycles=self.n_cycles)

        if nRandomizations==0:
            self.cons_rec = cons_rec
            self.cons_nrec = cons_nrec

            if self.MTL_only:
                fn = self.root+''+self.s+'/'+self.s+'_mne_MTL_'+self.ref_scheme+'_'+self.task_phase+'_'+str(self.conn_type)+'_'+self.mode+'_'+str(self.win_st)+'-'+str(self.win_fin)+'_'+str(self.band)+'.npy'
            else:
                fn = self.root+''+self.s+'/'+self.s+'_mne_'+self.ref_scheme+'_'+self.task_phase+'_'+str(self.conn_type)+'_'+self.mode+'_'+str(self.win_st)+'-'+str(self.win_fin)+'_'+str(self.band)+'.npy'

            import os
            try:
                os.mkdir(self.root+'/'+sub+'/')
                np.save(fn, [cons_rec, cons_nrec])
            except:
                np.save(fn, [cons_rec, cons_nrec])
        else:
            pass

        if nRandomizations==0:
            return

        #Clip off the buffer
        if type(self.conn_type)==str:
            cons_rec = cons_rec[:, :, :, int(self.buffer_time*self.samplerate):int(self.buffer_time*self.samplerate)+int((self.win_fin-self.win_st)*self.samplerate)]
            cons_nrec = cons_nrec[:, :, :, int(self.buffer_time*self.samplerate):int(self.buffer_time*self.samplerate)+int((self.win_fin-self.win_st)*self.samplerate)]
        else:
            for q in range(len(cons_rec)):
                cons_rec[q] = cons_rec[q][:, :, :, int(self.buffer_time*self.samplerate):int(self.buffer_time*self.samplerate)+int((self.win_fin-self.win_st)*self.samplerate)]
                cons_nrec[q] = cons_nrec[q][:, :, :, int(self.buffer_time*self.samplerate):int(self.buffer_time*self.samplerate)+int((self.win_fin-self.win_st)*self.samplerate)]


        fstats = []
        n1 = float(len(arr['recalled'])); n2 = float(len(arr['not_recalled']))
        binsize=int(self.samplerate/10.0) #100ms windows

        if type(self.conn_type)==str:
            total_wins = cons_rec.shape[3]/binsize

            diff = cons_rec-cons_nrec
            if avg_epochs:
                diff_avg = np.reshape(diff[:, :, :, :binsize*total_wins],
                       (cons_rec.shape[0], cons_rec.shape[1], cons_rec.shape[2], total_wins, binsize))
                diff_avg = np.mean(diff_avg, axis=4)
                fstats.append(diff_avg)
            else:
                fstats.append(diff)
        else:
            for i in range(len(cons_rec)):
                fstats.append([])
                total_wins = cons_rec[i].shape[3]/binsize

                diff = cons_rec[i]-cons_nrec[i]
                if avg_epochs:
                    diff_avg = np.reshape(diff[:, :, :, :binsize*total_wins],
                       (diff.shape[0], diff.shape[1], diff.shape[2], total_wins, binsize))
                    diff_avg = np.mean(diff_avg, axis=4)
                    fstats[i].append(diff_avg)
                else:
                    fstats[i].append(diff)

        #Load preshuffled vectors
        #shuffled_vecs = np.load('/scratch/esolo/mne_analysis/'+self.s+'/shuffled_vecs.npy')
        shuffled_vecs = np.arange(n1+n2)

        from copy import copy
        for k in range(nRandomizations):

            if match_trial_n:
                idxs = np.arange(len(rec_idxs)+len(nrec_idxs))
                np.random.shuffle(idxs)
                n1 = len(rec_idxs)
            else:
                np.random.shuffle(shuffled_vecs)
                idxs = copy(shuffled_vecs)

            null_rec, _, _, _, _ = mne.connectivity.spectral_connectivity(arr[idxs[:int(n1)].astype(int)],
            method=self.conn_type, mode=self.mode, sfreq=sfreq, fmin=fmin, fmax=fmax,
            faverage=False, n_jobs=1, verbose=False, cwt_frequencies=cwtfreq, cwt_n_cycles=self.n_cycles)

            null_nrec, _, _, _, _ = mne.connectivity.spectral_connectivity(arr[idxs[int(n1):].astype(int)],
            method=self.conn_type, mode=self.mode, sfreq=sfreq, fmin=fmin, fmax=fmax,
            faverage=False, n_jobs=1, verbose=False, cwt_frequencies=cwtfreq, cwt_n_cycles=self.n_cycles)

            #Clip off the buffer
            if type(self.conn_type)==str:
                null_rec = null_rec[:, :, :, int(self.buffer_time*self.samplerate):int(self.buffer_time*self.samplerate)+int((self.win_fin-self.win_st)*self.samplerate)]
                null_nrec = null_nrec[:, :, :, int(self.buffer_time*self.samplerate):int(self.buffer_time*self.samplerate)+int((self.win_fin-self.win_st)*self.samplerate)]
            else:
                for q in range(len(null_rec)):
                    null_rec[q] = null_rec[q][:, :, :, int(self.buffer_time*self.samplerate):int(self.buffer_time*self.samplerate)+int((self.win_fin-self.win_st)*self.samplerate)]
                    null_nrec[q] = null_nrec[q][:, :, :, int(self.buffer_time*self.samplerate):int(self.buffer_time*self.samplerate)+int((self.win_fin-self.win_st)*self.samplerate)]

            #Get average of the difference in 200ms windows
            if type(self.conn_type)==str:
                null_diff = null_rec-null_nrec
                if avg_epochs:
                    null_diff_avg = np.reshape(null_diff[:, :, :, :binsize*total_wins],
                                              (null_diff.shape[0], null_diff.shape[1], null_diff.shape[2], total_wins, binsize))
                    null_diff_avg = np.mean(null_diff_avg, axis=4)
                    fstats.append(null_diff_avg)
                else:
                    fstats.append(null_diff)
            else:
                for i in range(len(null_rec)):
                    null_diff = null_rec[i]-null_nrec[i]
                    if avg_epochs:
                        null_diff_avg = np.reshape(null_diff[:, :, :, :binsize*total_wins],
                              (null_diff.shape[0], null_diff.shape[1], null_diff.shape[2], total_wins, binsize))
                        null_diff_avg = np.mean(null_diff_avg, axis=4)
                        fstats[i].append(null_diff_avg)
                    else:
                        fstats[i].append(null_diff)
            print(k)

        fstats = np.array(fstats)

        try:
            np.save(fn, fstats)
        except:
            try:
                os.mkdir(self.root+'/'+sub+'/')
                np.save(fn, fstats)
            except:
                np.save(fn, fstats)

#Helper Functions

def exclude_bad(s, montage, just_bad=None):
    import numpy as np
    from glob import glob
    try:
        if montage!=0:
            fn = '/scratch/pwanda/electrode_categories/electrode_categories_'+s+'_'+str(montage)+'.txt'
        else:
            if len(glob('/data/eeg/'+s+'/docs/electrode_categories.txt'))>0:
                fn = '/data/eeg/'+s+'/docs/electrode_categories.txt'
            else:
                if len(glob('/scratch/pwanda/electrode_categories/electrode_categories_'+s+'.txt'))>0:
                    fn = '/scratch/pwanda/electrode_categories/electrode_categories_'+s+'.txt'
                else:
                    fn = '/scratch/pwanda/electrode_categories/'+s+'_electrode_categories.txt'

        with open(fn, 'r') as fh:
            lines = [mystr.replace('\n', '') for mystr in fh.readlines()]
    except:
        lines = []

    if just_bad is True:
        bidx=len(lines)
        try:
            bidx = [s.lower().replace(':', '').strip() for s in lines].index('bad electrodes')
        except:
            try:
                bidx = [s.lower().replace(':', '').strip() for s in lines].index('broken leads')
            except:
                lines = []
        lines = lines[bidx:]

    return lines
