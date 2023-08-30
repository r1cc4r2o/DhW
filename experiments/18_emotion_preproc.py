import glob
import numpy as np
import pandas as pd


import mne
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals, get_datatypes, print_dir_tree, make_report, find_matching_paths
from mne.decoding import UnsupervisedSpatialFilter
from tqdm import tqdm

import torch


import pickle


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA




# list files
files_name_eeglab = sorted(glob.glob('./emotion_recognition_data/*/eeg/sub-*_task-ImaginedEmotion_eeg.set'))
files_name_tsv = sorted(glob.glob('./emotion_recognition_data/*/eeg/sub-*_task-ImaginedEmotion_events.tsv'))

""" Experiment setup for the EEG emotion recognition study.

    source: https://openneuro.org/datasets/ds003004/versions/1.1.1

    PARADIGM: The study uses the method of guided imagery to induce resting, eyes-closed 
    participants using voice-guided imagination to enter distinct 15 emotion states during 
    acquisition of high-density EEG data.

    During the study, participants listen to 15 voice recordings that each suggest 
    imagining a scenario in which they have experienced -- or would experience the named 
    target emotion. Some target emotions have positive valence (e.g., joy, happiness), 
    others negative valence (e.g., sadness, anger). Before and between the 15 emotion 
    imagination periods, participants hear relaxation suggestions ('Now return to a neutral 
    state by ...').

    PROCEDURE: When the participant first begins to feel the target emotion, they are asked 
    to indicate this by pressing a handheld button. Participants are asked to continue 
    feeling the emotion as long as possible. To intensify and lengthen the periods of experienced 
    emotion, participants are asked to interoceptively perceive and attend relevant somatosensory 
    sensations. When the target feeling wanes (typically after 1 and 5 minutes), participants push 
    the button again to leave the emotion imagination period and cue the relaxation instructions.

"""


# pd.read_csv('./emotion_recognition_data/sub-01/eeg/sub-01_task-ImaginedEmotion_events.tsv', sep='\t')


#######################################################
######## LOAD THE ANNONTATIONS FROM THE TSV FILE ######
#######################################################

list_of_emotions = [ 
                    'awe', 
                    'frustration',
                    'joy',
                    'anger',
                    'happy',
                    'sad',
                    'love',
                    'fear',
                    'compassion',
                    'jealousy',
                    'content',
                    'grief',
                    'relief',
                    'excite',
                    'disgust'
                    ]

press_button = 'press1'


def get_emotion_onset(tsv_file, duration = 100.0):
    """ Get the onset of the emotion from the tsv file.

    Args:
        @tsv_file: the tsv file with the emotion data
        @duration: the duration of the emotion in seconds

    Returns:
        @df: a dataframe with the emotion, the onset and the duration
    
    """

    df = pd.read_csv(tsv_file, sep='\t')

    emotions = []
    onsents = []

    # we need to check if the next value is press1
    # when the person press1, means that he/she is 
    # feeling the emotion
    for idx, i in enumerate(zip(df['value'], df['onset'])):

        # check if the value is in the list of emotions
        if i[0] in list_of_emotions:  
            # check if the next value is press1
            if df['value'][idx+1] == press_button:

                # print(df['value'][idx], df['value'][idx+1], df['onset'][idx+1])

                # append the emotion
                emotions.append(df['value'][idx])

                # append the onset
                onsents.append(df['onset'][idx+1])

    # create a dataframe
    df = pd.DataFrame({'value': emotions, 'onset': onsents, 'duration': [duration]*len(emotions)})

    return df

#######################################################
######## LOAD EEG DATA FROM THE EEG FILE ##############
#######################################################

# load the data
def get_data_eeglab(files_name_eeglab, files_name_tsv, base_path):
    """ This function loads the data from the eeglab files and the tsv 
    files and saves them in a npz file.

    @param files_name_eeglab: list of strings
        list of the eeglab files
    @param files_name_tsv: list of strings
        list of the tsv files
    @param base_path: string
        path where to save the data

    @return: None
    """
    files_name = zip(files_name_eeglab, files_name_tsv)
    
    data = []
    for eeglab_file, tsv in files_name:
        try:
            #######################################################
            ######## LOAD EEG DATA FROM THE EEG FILE ##############
            #######################################################
            # load the data
            raw = mne.io.read_raw_eeglab(eeglab_file, preload=True)

            #######################################################
            ######## LOAD ANNOTATIONS FROM THE TSV FILE ###########
            #######################################################

            df = get_emotion_onset(tsv, duration = 100.0)

            # add the annotations to the raw data
            annot = mne.Annotations(
                onset=list(df['onset']),  # in seconds
                duration=list(np.array([60.0]*len(df['onset']))),  # in seconds, too
                description=list(df['value']),
            )

            # print the annotations
            # print(annot)

            raw2 = raw.copy().set_annotations(annot)

            # get events from the annotations
            events_from_annot, event_dict = mne.events_from_annotations(raw2)

            # epochs # 10 seconds before the onset and 90 seconds after the onset
            epochs = mne.Epochs(raw2, events_from_annot, event_id=event_dict, tmin=-10, tmax=90, preload=True)

            # apply baseline correction
            epochs.apply_baseline((0, 90.0))

            # filter the data
            epochs.filter(4, 80)

            # get the data
            epochs_data = epochs.get_data()

            ev_pca_data = []
            for sample in tqdm(range(epochs_data.shape[0])):
                pca = UnsupervisedSpatialFilter(PCA(15), average=False)
                pca_data = pca.fit_transform(epochs_data[sample][np.newaxis, :, :])
                ev = mne.EvokedArray(
                    np.mean(pca_data, axis=0),
                    mne.create_info(15, epochs.info["sfreq"]),
                    tmin=-10,
                )
                ev_pca_data.append(ev.get_data())

            
            # save the data
            print('SAVED: '+base_path + 'data_{}.npz'.format(eeglab_file.split('/')[2]))
            np.savez_compressed(base_path + 'data_{}.npz'.format(eeglab_file.split('/')[2]), data=np.stack(ev_pca_data,dtype='float32')*1e22)
            del ev_pca_data 
            # save dataframe
            df.to_csv(base_path + 'data_{}.csv'.format(eeglab_file.split('/')[2]), index=False)
            
        except:
            print('ERROR: '+base_path + 'data_{}.npz'.format(eeglab_file.split('/')[2]))
            pass




###############################################
###############################################
###############################################


base_path = './data_with_pca/emotion_recognition_preprocessed_data/'

get_data_eeglab(files_name_eeglab, files_name_tsv, base_path)


gender_map = {'M': 0, 'F': 1}
age_map = {'25-35': 0, '35-45': 1, '45-55': 2, '55+': 3}
part_id = {
    'sub-01': 0,
    'sub-02': 1,
    'sub-03': 2,
    'sub-04': 3,
    'sub-05': 4,
    'sub-06': 5,
    'sub-07': 6,
    'sub-08': 7,
    'sub-09': 8,
    'sub-10': 9,
    'sub-11': 10,
    'sub-12': 11,
    'sub-13': 12,
    'sub-14': 13,
    'sub-15': 14,
    'sub-16': 15,
    'sub-17': 16,
    'sub-18': 17,
    'sub-19': 18,
    'sub-20': 19,
    'sub-21': 20,
    'sub-22': 21,
    'sub-23': 22,
    'sub-24': 23,
    'sub-25': 24,
    'sub-26': 25,
    'sub-27': 26,
    'sub-28': 27,
    'sub-29': 28,
    'sub-30': 29,
    'sub-31': 30,
    'sub-32': 31,
    'sub-33': 32,
    'sub-34': 33,
    'sub-35': 34
}

emotions = {'awe': 0,
            'fear': 1,
            'joy': 2,
            'jealousy': 3,
            'excite': 4,
            'disgust': 5,
            'love': 6,
            'grief': 7,
            'content': 8,
            'frustration': 9,
            'happy': 10,
            'anger': 11,
            'relief': 12,
            'sad': 13,
            'compassion': 14}

b_path = './data_with_pca/emotion_recognition_preprocessed_data/'

info = torch.tensor([])
data_X = torch.tensor([])
data_y = torch.tensor([])

number_steps = 14
length_s = 25601
max_sub_win = max([M-m for m, M in zip(ind[:-1], ind[1:])])

ind = np.linspace(0, length_s, number_steps, dtype=int)
file_name = sorted(glob.glob(b_path+'*.npz'))
participants = pd.read_csv('./emotion_recognition_data/participants.tsv', sep='\t').sort_values(by=['participant_id']).reset_index(drop=True)


byte = 0
# for file in file_name:
for file in file_name:
    print(file)
    data = np.load(file)
    data = data['data'][:, :, :length_s]
    print(data.shape)

    data_cutted = torch.cat([torch.stack([
        torch.tensor(d[:,m:M], dtype=torch.float32) if d[:,m:M].shape[1] == max_sub_win else torch.cat([torch.tensor(d[:,m:M], dtype=torch.float32), torch.zeros((15, max_sub_win-d[:,m:M].shape[1]))], dim=1)
            for m, M in zip(ind[:-1], ind[1:])]) for d in data])

    participant_info = participants[participants['participant_id'] == file.split('/')[-1].split('_')[1].split('.')[0]].values[0]

    participant_info = torch.tensor(
                            np.array([np.array([part_id[participant_info[0]], gender_map[participant_info[1]], participant_info[2], participant_info[4]])]*data_cutted.shape[0]), 
                                dtype=torch.int16)

    subj_id = file.split('/')[-1].split('_')[1].split('.')[0]
    l = torch.tensor([emotions[emotion] for emotion in pd.read_csv(b_path+f'/data_{subj_id}.csv')['value'].values]*(number_steps-1))

    # save info in list
    info = torch.cat([info, participant_info])
    data_X = torch.cat([data_X, data_cutted])
    data_y = torch.cat([data_y, l])


# save tensor
torch.save(data_X, b_path+'preprocessed_cutted/data.pt')
torch.save(info,  b_path+'preprocessed_cutted/info.pt')
torch.save(data_y,  b_path+'preprocessed_cutted/labels.pt')

# # out
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-01.npz
# (14, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-02.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-03.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-04.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-05.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-06.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-07.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-08.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-09.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-10.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-11.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-12.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-13.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-14.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-15.npz
# (14, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-16.npz
# (14, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-17.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-18.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-19.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-20.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-21.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-23.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-25.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-26.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-28.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-29.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-30.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-31.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-32.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-34.npz
# (15, 15, 25601)
# ./data_with_pca/emotion_recognition_preprocessed_data/data_sub-35.npz
# (15, 15, 25601)


# print('Number of channel which preserve the 95% of the variance: mean {}, std {}, max {} '.format(
# np.array([sample.shape[0] for sample in ev_pca_data]).mean(),\
#       np.array([sample.shape[0] for sample in ev_pca_data]).std(),\
#         np.array([sample.shape[0] for sample in ev_pca_data]).max()))

# Number of channel which preserve the 90% of the variance: mean 10.5, std 3.556683848755748, max 18 



mapping_list = {}
mapping_id = []


def get_name_from_id(code, id_map):
    """ Get the name of the event from the event id 

    Args:
        @param code: array of event ids
        @param id_map: event id dictionary

    Returns:
        @return: array of event names
    
    """
    return [key for c in code for key, value in id_map.items() if value == c]




path = './openneuro-aws/ds004276-download/'

datatype = 'meg'
extensions = [".tsv",".json"]  
bids_paths = find_matching_paths(path, datatypes=datatype, extensions=extensions)
bids_path = BIDSPath(root=path, datatype=datatype)
print(bids_path.match())

for i in range(12,19,1):

    if i < 10:
        subject = f'00{i}'
    else:
        subject = f'0{i}'

    print(f'Processing subject {i}')

    try:
        # get the raw file of the first subject
        task = 'words'
        suffix = 'meg'


        bids_path_subject = bids_path.update(subject=subject, task=task, suffix=suffix)
        # print(bids_path_subject)
        # get the raw data
        raw = read_raw_bids(bids_path=bids_path_subject, verbose=False)

        # print(raw.get_data().shape)

        # pd.DataFrame()
        df = pd.read_csv(f'./openneuro-aws/ds004276-download/sub-{subject}/beh/sub-{subject}_task-words_beh.tsv', sep='\t')

        # print(df)

        # create annotations
        annot = mne.Annotations(
            onset=list(df['Trial']),  # in seconds
            duration=list(df['TTime']/10000),  # in seconds, too
            description=list(df['Code']),
        )
        # print(annot)

        raw2 = raw.copy().set_annotations(annot)
        del raw

        # pd.DataFrame(raw2.annotations)

        # get events from the annotations
        events_from_annot, event_dict = mne.events_from_annotations(raw2)

        tmin = -0.1
        tmax = 0.8

        # epochs # 10 seconds before the onset and 90 seconds after the onset
        epochs = mne.Epochs(
                            raw2, 
                            events_from_annot, 
                            event_id=event_dict, 
                            tmin=tmin, 
                            tmax=tmax, 
                            preload=True, 
                            event_repeated='merge'
                        )
        
        del raw2

        # apply Band-pass filter 
        # analyzing alpha oscillations (8-13 Hz)
        # analyzing beta oscillations (13-30 Hz) 
        # analyzing gamma oscillations (30-80 Hz)
        epochs.filter(4, 80)

        # apply baseline correction
        # subtracting a baseline period 
        # from each time point in the data
        epochs.apply_baseline((0, .8))

        id_list = get_name_from_id(epochs.events[:, 2], epochs.event_id)

        for data, id_k in zip(epochs.get_data().astype(np.float32), id_list):
            pca = UnsupervisedSpatialFilter(PCA(4), average=False)
            # print(data[None, :, :].shape)
            pca_data = pca.fit_transform(data[None, :, :])
            ev = mne.EvokedArray(
                np.mean(pca_data, axis=0),
                mne.create_info(4, epochs.info["sfreq"]),
                tmin=tmin,
            )
            # print(ev.get_data().shape)

        #     print(data.shape)
            if id_k in mapping_list:
                mapping_list[id_k] = np.concatenate((mapping_list[id_k], ev.get_data()[None, :]), axis=0)
            else:
                mapping_list[id_k] = ev.get_data()[None, :]

        mapping_id.append(epochs.event_id)

        with open(f'./data_with_pca/word_recognition_preprocessed_data/sub-{subject}_data.pkl', 'wb') as f:
            pickle.dump(mapping_list, f)

        with open(f'./data_with_pca/word_recognition_preprocessed_data/sub-{subject}_id.pkl', 'wb') as f:
            pickle.dump(mapping_id, f)

    except:
        print(f'Error with subject {subject}')