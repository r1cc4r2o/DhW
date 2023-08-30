import numpy as np
from pandas import read_csv
from sklearn.metrics import roc_auc_score

import mne
from mne.io import read_raw_fif, concatenate_raws
from mne.datasets import visual_92_categories
import pickle
from mne.decoding import UnsupervisedSpatialFilter
from tqdm import tqdm

import torch

from sklearn.decomposition import PCA, FastICA


data_path = visual_92_categories.data_path()

# Define stimulus
fname = data_path / 'visual_stimuli.csv'

conds = read_csv(fname)

conditions = []
for c in conds.values:
    cond_tags = list(c[:2])
    cond_tags += [('not-' if i == 0 else '') + conds.columns[k]
                for k, i in enumerate(c[2:], 2)]
    conditions.append('/'.join(map(str, cond_tags)))


event_id = dict(zip(conditions, conds.trigger + 1))

n_runs = 4  
fnames = [data_path / f'sample_subject_{b}_tsss_mc.fif' for b in range(n_runs)]
raws = [read_raw_fif(fname, verbose='error', on_split_missing='ignore')
        for fname in fnames]  # ignore filename warnings

# concatenate raw data
raw = concatenate_raws(raws)

# find events
events = mne.find_events(raw, min_duration=.002)

print('number of events: ', len(events))
print('number of events: ', len(event_id))

picks = mne.pick_types(raw.info, meg=True)


epochs = mne.Epochs(raw, events=events, event_id=event_id, baseline=None, picks=picks, tmin=-.1, tmax=.500, preload=True)

# print('number of events: ', len([epochs[i].get_data().shape for i in epochs.event_id]))

pca_epochs = epochs.get_data()


# I run the procedure over $1000$ samples to decide how many 
# channels to keep. In my opinion, the best choice is the one 
# that keeps $90\%$ of the variance. Since I could not run the 
# pca over the entire dataset. I decided to run it over $1000$ 
# samples and then use the same number of channels for the 
# entire dataset. As number of channels I choose the maximum 
# number of channels that keeps $90\%$ of the variance. 
# In this case, the number of channels is $30$.

# pca_data = []
# for sample in tqdm(range(pca_epochs.shape[0])[:1000]):
#     pca = UnsupervisedSpatialFilter(PCA(0.90), average=False)
#     pca_data.append(pca.fit_transform(pca_epochs[sample][np.newaxis, :, :])[0])
    
# 100%|██████████| 1000/1000 [02:51<00:00,  5.83it/s]

# print('Number of channel which preserve the 90% of the variance: mean {}, std {}, max {} '.format(
# np.array([sample.shape[0] for sample in pca_data]).mean(),\
#       np.array([sample.shape[0] for sample in pca_data]).std(),\
#         np.array([sample.shape[0] for sample in pca_data]).max()))

# Number of channel which preserve the 90% of the variance: mean 22.866, std 3.1009746854819693, max 30 

ev_pca_data = []
for sample in tqdm(range(pca_epochs.shape[0])):
    pca = UnsupervisedSpatialFilter(PCA(30), average=False)
    pca_data = pca.fit_transform(pca_epochs[sample][np.newaxis, :, :])
    ev = mne.EvokedArray(
        np.mean(pca_data, axis=0),
        mne.create_info(30, epochs.info["sfreq"]),
        tmin=-.1,
    )
    ev_pca_data.append(ev.get_data())

# # error 
# torch.stack([torch.tensor(sample)*1e22 for sample in ev_pca_data]).type(torch.float32).mean()\
#     -torch.stack([torch.tensor(sample)*1e22 for sample in ev_pca_data]).mean(),\
# torch.stack([torch.tensor(sample) for sample in ev_pca_data]).type(torch.float32).mean()\
#     -torch.stack([torch.tensor(sample) for sample in ev_pca_data]).mean()
    
# (tensor(5.2928, dtype=torch.float64), tensor(-2.7015e-22, dtype=torch.float64))

torch.save(torch.stack([torch.tensor(sample) for sample in ev_pca_data]).type(torch.float32), 'data_with_pca/visual_stimuli/ev_pca_data_float32_shift.pt')
torch.save(torch.stack([torch.tensor(sample) for sample in ev_pca_data]),'data_with_pca/visual_stimuli/ev_pca_data_float64.pt')