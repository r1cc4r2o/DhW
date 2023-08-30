import glob
import mne
import numpy as np
import pandas as pd
import torch
import pickle
from mne.decoding import UnsupervisedSpatialFilter
from tqdm import tqdm

from sklearn.decomposition import PCA, FastICA


# list files
files_name_eeglab = sorted(glob.glob('./face_processing_data/*/eeg/sub-*_task-FaceRecognition_eeg.set'))
files_name_tsv = sorted(glob.glob('./face_processing_data/sub-*/eeg/sub-*_task-FaceRecognition_events.tsv'))



###############################################################

def get_onset_duration(df):
    """ This function gets the onset and duration given the df """
    onset = []
    duration = []
    trial_type = []
    label = []

    for i in range(len(df)):
        if df['event_type'][i] == 'button_press':
            if df['event_type'][i-1] == 'faces':
                onset.append(df['onset'][i-1])
                duration.append(df['onset'][i] - df['onset'][i-1])
                trial_type.append(df['trial_type'][i-1])
                label.append(df['value'][i-1])

                # code to get the max duration
                # this allows me to get a size 
                # of the window without overlaps
                # of 1.4 seconds
                # try:
                #     duration_max.append(df['onset'][i+1] - df['onset'][i-1])
                # except:
                #     pass


    return np.array(onset), np.array(duration), np.array(trial_type), np.array(label) # , np.array(duration_max)

###############################################################

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

    # target labels
    target_labels = torch.tensor([])
    target_trial_type = np.array([])
    X = torch.tensor([])
    event_dict_all = []
    subject_id = torch.tensor([])

    data = []
    for eeglab_file, tsv in files_name:

        #######################################################
        ######## LOAD EEG DATA FROM THE EEG FILE ##############
        #######################################################
        # load the data
        raw = mne.io.read_raw_eeglab(eeglab_file, preload=True)

        #######################################################
        ######## LOAD ANNOTATIONS FROM THE TSV FILE ###########
        #######################################################

        df = pd.read_csv(tsv, sep='\t')

        # get the onset and duration
        onset, duration, trial_type, label = get_onset_duration(df)

        # duration_max = np.max(duration) # 1.4s

        # get the target labels
        target_labels = torch.cat([target_labels, torch.from_numpy(label)])
        target_trial_type = np.concatenate([target_trial_type, trial_type])

        # add the annotations to the raw data
        annot = mne.Annotations(
            onset=list(onset),  # in seconds
            duration=[1.4]*onset.shape[0],  # in seconds, too
            description=list(trial_type),
            # id_code=list(df['value']),
        )

        # print the annotations
        # print(annot)

        raw2 = raw.copy().set_annotations(annot)

        # raw2.plot()
        # print(pd.DataFrame(raw2.annotations))

        # get events from the annotations
        events_from_annot, event_dict = mne.events_from_annotations(raw2)

        tmin=-0.2
        tmax=0.55

        # epochs
        epochs = mne.Epochs(raw2, events_from_annot, event_id=event_dict, tmin=tmin, tmax=tmax, preload=True)

        # filter the data
        epochs.filter(4, 80)

        # apply baseline correction
        epochs.apply_baseline((0, 0.55))

        data_pca = epochs.get_data()

        ev_pca_full = []
        for i in range(data_pca.shape[0]):
            pca = UnsupervisedSpatialFilter(PCA(7), average=False)
            pca_data = pca.fit_transform(data_pca[i][None, :, :])
            ev = mne.EvokedArray(
                np.mean(pca_data, axis=0),
                mne.create_info(pca_data.shape[1], epochs.info["sfreq"]),
                tmin=tmin,
            )
            # print(ev.get_data().shape)
            ev_pca_full.append(ev.get_data()*1e22)
        
        ev_pca_full = np.stack(ev_pca_full, axis=0)
        # print(ev_pca_full.shape)

        # subject id
        subject_id = torch.cat([subject_id,torch.tensor([int(i) for i in [eeglab_file.split('/')[2].split('-')[1]]])])

        # concatenate the data
        X = torch.cat([X, torch.from_numpy(ev_pca_full).to(torch.float32)])

        # concatenate the event dict
        event_dict_all.append(event_dict)

        # Save the data
        data.append((eeglab_file.split('/')[2], ev_pca_full, epochs.info, epochs.events, epochs.event_id, event_dict, epochs.tmin, epochs.tmax))

        for d in data:
            print('SAVED: '+base_path + 'data_{}.npz'.format(d[0]))
            np.savez_compressed(base_path + 'data_{}.npz'.format(d[0]), data=np.array(d[1],dtype='float32'))

        # # save the data
        torch.save(X, base_path + '/dataset/X.pt')
        torch.save(target_labels, base_path + '/dataset/target_labels.pt')
        # torch.save(target_trial_type, base_path + '/dataset/target_trial_type.pt')
        np.save(base_path + '/dataset/target_trial_type.npy', target_trial_type)
        # torch.save(subject_id, base_path + '/dataset/subject_id.pt')
        with open(base_path + '/dataset/event_dict_all.pkl', 'wb') as f:
            pickle.dump(event_dict_all, f)
        

base_path = './data_with_pca/face_recognition_preprocessed_data/'
get_data_eeglab(files_name_eeglab, files_name_tsv, base_path)


# files_name = sorted(glob.glob('./data_with_pca/face_recognition_preprocessed_data/data_sub-*.npz'))
# print(torch.load('./data_with_pca/face_recognition_preprocessed_data/dataset/target_labels.pt').shape == np.array([np.load(file_name)['data'].shape[0] for file_name in files_name]).sum())
# np.array([np.load(file_name)['data'].shape[0] for file_name in files_name]),files_name

# [ True]
# (array([584, 860, 794, 805, 883, 884, 882, 889, 882, 884, 888, 880, 883,
#         872, 885, 883, 881, 873]),
#  ['./data_with_pca/face_recognition_preprocessed_data/data_sub-002.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-003.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-004.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-005.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-006.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-007.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-008.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-009.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-010.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-011.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-012.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-013.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-014.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-015.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-016.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-017.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-018.npz',
#   './data_with_pca/face_recognition_preprocessed_data/data_sub-019.npz'])

# print(X.shape, y.shape)
# torch.Size([15392, 7, 189]) torch.Size([15392])