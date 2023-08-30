import os

import pickle
import numpy as np
import pandas as pd

import torch
import glob
import clip

import mne
from mne.decoding import UnsupervisedSpatialFilter
from tqdm import tqdm

from sklearn.decomposition import PCA, FastICA




X = torch.tensor([])
y = torch.tensor([])

with open('./data_without_pca/word_recognition_preprocessed_data/dataset/words.pkl', 'rb') as f:
    words = pickle.load(f)

words = {w: i for i, w in enumerate(words)}

file_name = glob.glob('./data_with_pca/word_recognition_preprocessed_data/*18*_data.pkl')

mapping_list = {}

for file in file_name:
    print(file)

    with open(file, 'rb') as f:
        data = pickle.load(f)

    # convert values to float32
    data = {key: value.astype(np.float32) for key, value in data.items()}

    for key, value in data.items():
        try:
            # check if the file exists
            if os.path.exists(f'./data_with_pca/word_recognition_preprocessed_data/{key}.pt'):
                data_temp = torch.load(f'./data_with_pca/word_recognition_preprocessed_data/{key}.pt')
                data_temp = torch.cat((data_temp, torch.from_numpy(value)), dim=0)
                torch.save(data_temp, f'./data_with_pca/word_recognition_preprocessed_data/{key}.pt')
            else:
                torch.save(torch.from_numpy(value), f'./data_with_pca/word_recognition_preprocessed_data/{key}.pt')

        except:
            pass
        
        

with open('./data_without_pca/word_recognition_preprocessed_data/dataset/words.pkl', 'rb') as f:
    words = pickle.load(f)

words = {w: i for i, w in enumerate(words)}


# load the brain data
import torch

data = torch.tensor([])
y = torch.tensor([])

for w in words:
    try:
        print(w)
        data_temp = torch.load(f'./data_with_pca/word_recognition_preprocessed_data/{w}.pt')
        data = torch.cat((data, data_temp), dim=0).to(torch.float32)
        print(data.shape)
        y = torch.cat((y, torch.tensor([words[w]]*data_temp.shape[0]))).to(torch.float32)
        del data_temp

        # save checkpoint
        torch.save(data, './data_with_pca/word_recognition_preprocessed_data/dataset/X.pt')
        torch.save(y, './data_with_pca/word_recognition_preprocessed_data/dataset/y.pt')
    except:
        pass

print(data.shape)

device = "cuda" if torch.cuda.is_available() else "cpu"

with open('./data_without_pca/word_recognition_preprocessed_data/dataset/words.pkl', 'rb') as f:
    words = pickle.load(f)

y = torch.load('./data_with_pca/word_recognition_preprocessed_data/dataset/y.pt')

idx_words = {i: w for i, w in enumerate(words)}

model, preprocess = clip.load("ViT-B/32", device=device)


with torch.no_grad():
    text_features = model.encode_text(torch.cat([clip.tokenize(idx_words[id_word]) for id_word in y.tolist()]).to(device))

text_features /= text_features.norm(dim=-1, keepdim=True)

# save the text features
torch.save(text_features, './data_with_pca/word_recognition_preprocessed_data/dataset/word_text_features.pt')

