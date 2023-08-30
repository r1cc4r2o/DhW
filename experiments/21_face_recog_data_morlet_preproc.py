from mne.time_frequency import tfr_array_morlet
import numpy as np
from mne.baseline import rescale
from mne.viz.utils import centers_to_edges
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.print_figure_kwargs = {'pad_inches':0}
plt.ioff()

from tqdm import tqdm
import torch


###########################################################


base_path = './data_with_pca/face_recognition_preprocessed_data/'
X = torch.load(base_path + '/dataset/X.pt')
y = torch.load(base_path + '/dataset/target_labels.pt')

print(X.shape, y.shape)

###########################################################

freqs = np.arange(4.0, 80.0, 3.0)
n_cycles = freqs / 3.0  # different number of cycle per frequency
sfreq = 250.0  # sampling in Hz
times = np.arange(0.0, 0.189, 0.001, dtype=np.float16) 

n_sample_per_class = 150

###########################################################

base_path_save_tfr = './data_with_pca/face_recognition_preprocessed_data/tfr_dataset/full_data/'

out = torch.tensor([])
interval = np.arange(0, X.shape[0], 513)

# split in batches
for m, M in tqdm(zip(interval[:-1], interval[1:])):

    power = tfr_array_morlet(
        X[m:M],
        sfreq=sfreq,
        freqs=freqs,
        n_jobs=16,
        n_cycles=n_cycles,
        output='complex', # get tfr per trial
        verbose=False
    )

    rescale(power, times, (0.0, 0.189), mode="mean", copy=False, verbose=False)

    out = torch.cat((out, torch.from_numpy(power*1e-18).type(torch.float32)), dim=0)

print(out.shape)

# torch.Size([15392, 7, 189]) torch.Size([15392])
# 30it [01:40:38,  1.27s/it]
# torch.Size([15390, 7, 26, 189])

torch.save(out, base_path_save_tfr+'X.pt')
torch.save(y, base_path_save_tfr+'y.pt')

