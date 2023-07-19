#################################################################
#                        IMPORT
#################################################################


import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.manifold import MDS

import mne
from mne.io import read_raw_fif, concatenate_raws
from mne.datasets import visual_92_categories
import pickle


#################################################################
#                       EXTRACTION OF DATA
#################################################################



data_path = visual_92_categories.data_path()

# Define stimulus - trigger mapping
fname = data_path / 'visual_stimuli.csv'


for min_trigger, max_trigger in zip(list(np.linspace(0, 92, 3, dtype=int)[:2]),list(np.linspace(0, 92, 3, dtype=int)[1:])):
    
    conds = read_csv(fname)

    conds = conds[min_trigger:max_trigger]

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

    events = events[min_trigger <= events[:, 2]]
    events = events[events[:, 2] <= max_trigger]

    print('number of events: ', len(events))
    print('number of events: ', len(event_id))

    picks = mne.pick_types(raw.info, meg=True)


    epochs = mne.Epochs(raw, events=events, event_id=event_id, baseline=None, picks=picks, tmin=-.1, tmax=.500, preload=True)


    # print('number of events: ', len([epochs[i].get_data().shape for i in epochs.event_id]))

    ica_epochs = epochs.copy().load_data().pick_types(meg=True, eeg=False)
    method = 'picard'
    ica = mne.preprocessing.ICA(n_components=0.9, method=method, random_state=0)

    ica.fit(ica_epochs)
    ica.apply(ica_epochs)

    explained_var_ratio = ica.get_explained_variance_ratio(ica_epochs)
    for channel_type, ratio in explained_var_ratio.items():
        print(
            f'Fraction of {channel_type} variance explained by all components: '
            f'{ratio}'
        )

    with open(f'./data/visual_stimuli/ica_epochs_min{min_trigger}_max_{max_trigger}.pickle', 'wb') as f:
        pickle.dump(ica_epochs.get_data(), f, protocol=pickle.HIGHEST_PROTOCOL)

    ica_epochs.save(f'./data/visual_stimuli/ica_epochs_min{min_trigger}_max_{max_trigger}.fif', overwrite=True)



#################################################################
# 4142 events found
# Event IDs: [  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
#   19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
#   37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54
#   55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72
#   73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90
#   91  92  93 200 222 244]
# number of events:  1380
# number of events:  46
# Not setting metadata
# 1380 matching events found
# No baseline correction applied
# 0 projection items activated
# Loading data for 1380 events and 601 original time points ...
# 1 bad epochs dropped
# Fitting ICA to data using 306 channels (please be patient, this may take a while)
# Selecting by explained variance: 30 components
# Fitting ICA took 109.5s.
# Applying ICA to Epochs instance
#     Transforming to ICA space (30 components)
#     Zeroing out 0 ICA components
#     Projecting back using 306 PCA components
# Fraction of mag variance explained by all components: 0.9693809573537527
# Fraction of grad variance explained by all components: 0.8669398567595916
# /home/rickbook/document/neuro/project-abns/project-abns/experiments/4.1_extract_clean_data_visualstimuli.py:93: RuntimeWarning: This filename (./data/visual_stimuli/ica_epochs_min0_max_46.fif) does not conform to MNE naming conventions. All epochs files should end with -epo.fif, -epo.fif.gz, _epo.fif or _epo.fif.gz
#   ica_epochs.save(f'./data/visual_stimuli/ica_epochs_min{min_trigger}_max_{max_trigger}.fif', overwrite=True)
# Overwriting existing file.
# 4142 events found
# Event IDs: [  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
#   19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
#   37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54
#   55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72
#   73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90
#   91  92  93 200 222 244]
# number of events:  1410
# number of events:  46
# Not setting metadata
# 1380 matching events found
# No baseline correction applied
# 0 projection items activated
# Loading data for 1380 events and 601 original time points ...
# 1 bad epochs dropped
# Fitting ICA to data using 306 channels (please be patient, this may take a while)
# Selecting by explained variance: 30 components
# Fitting ICA took 104.9s.
# Applying ICA to Epochs instance
#     Transforming to ICA space (30 components)
#     Zeroing out 0 ICA components
#     Projecting back using 306 PCA components
# Fraction of mag variance explained by all components: 0.9701552033008111
# Fraction of grad variance explained by all components: 0.8676646923821766
# /home/rickbook/document/neuro/project-abns/project-abns/experiments/4.1_extract_clean_data_visualstimuli.py:93: RuntimeWarning: This filename (./data/visual_stimuli/ica_epochs_min46_max_92.fif) does not conform to MNE naming conventions. All epochs files should end with -epo.fif, -epo.fif.gz, _epo.fif or _epo.fif.gz
#   ica_epochs.save(f'./data/visual_stimuli/ica_epochs_min{min_trigger}_max_{max_trigger}.fif', overwrite=True)
# Overwriting existing file.
