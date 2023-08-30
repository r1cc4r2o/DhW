# Experiments folder

This folder contains some of the experiments I carried out. The main objective of the project is stated in the [report](). Here, I provide a brief description of what each notebook/python file contains.

---
### Preliminaries Experiments
- [0_overview_mne.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/0_overview_mne.ipynb): Follow the tutorial to learn how to use MNE
- [3_extract_sync_stimulus.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/3_extract_sync_stimulus.ipynb): Learn how to extract sync event
- [5_simulation.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/5_simulation.ipynb): Learn how to make the bem model and apply the inverse
- [5.1_visualize.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/5.1_visualize.ipynb): Visualize simple feature space
- [2.1_predict_signal_from_signal.py](https://github.com/r1cc4r2o/DhW/blob/main/experiments/2.1_predict_signal_from_signal.py): Predict which come next
- [2.2_predict_signal_from_attentionmap+stimulus.py](https://github.com/r1cc4r2o/DhW/blob/main/experiments/2.2_predict_signal_from_attentionmap+stimulus.py): Predict which come next
- [7_similarity_maps_crosspeople.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/7_similarity_maps_crosspeople.ipynb): Compute the euclidean distance maps reported in the seminar paper
---
### Preprocessing
- [4_visual_stimuli.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/4_visual_stimuli.ipynb): Load Visual stimuli dataset
- [4.1_extract_clean_data_visualstimuli.py](https://github.com/r1cc4r2o/DhW/blob/main/experiments/4.1_extract_clean_data_visualstimuli.py): Visual stimuli dataset preprocess
- [8_openneuro_word_recognition_read_data.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/8_openneuro_word_recognition_read_data.ipynb): Read and load the word recognition dataset
- [9_face_processing_read_preprocess_data.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/9_face_processing_read_preprocess_data.ipynb): Read and load the face recognition dataset
- [9.1_face_recognition_TimeFrequencyRepresentation.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/9.1_face_recognition_TimeFrequencyRepresentation.ipynb): Read, load and extract the Time-Frequencies Representations from the face recognition dataset
- [10_emotion_recognition_read_data.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/9_face_processing_read_preprocess_data.ipynb): Read and load the emotion recognition dataset
- [12_preprocess_emotion_recognition_data.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/9_face_processing_read_preprocess_data.ipynb): Preprocessing Emotion Recognition dataset
- [13_read_visual_stimuli_data.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/13_read_visual_stimuli_data.ipynb): Read visual stimuli dataset
- [15_mel_spectrogram.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/15_mel_spectrogram.ipynb): Extract Mel-Spectrogram
- [17_visual_stimuli_preproc.py](https://github.com/r1cc4r2o/DhW/blob/main/experiments/17_visual_stimuli_preproc.py): Pre-process visual stimuli dataset v2
- [18_emotion_preproc.py](https://github.com/r1cc4r2o/DhW/blob/main/experiments/18_emotion_preproc.py): Pre-process emotion recognition dataset v2
- [19_word_recog_data_preproc.py](https://github.com/r1cc4r2o/DhW/blob/main/experiments/19_word_recog_data_preproc.py): Pre-process word recognition dataset v2
- [20_face_recog_data_preproc.py](https://github.com/r1cc4r2o/DhW/blob/main/experiments/20_face_recog_data_preproc.py): Pre-process face recognition dataset v2
- [21_face_recog_data_morlet_preproc.py](https://github.com/r1cc4r2o/DhW/blob/main/experiments/21_face_recog_data_morlet_preproc.py): Pre-process face recognition dataset with morlet wavelet transform v2
---
### Trained Models
- [6_latent_representation_visualstimuli.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/6_latent_representation_visualstimuli.ipynb): Simple transformer-based architecture to perform inference
- [11_weighting_block.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/11_weighting_block.ipynb): Design Weighting Block
- [14_architecture_visual_stimuli_preprocessed_bilinear.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/14_architecture_visual_stimuli_preprocessed_bilinear.ipynb): Experiments feature extractor
- [14_architecture_visual_stimuli_preprocessed_cat.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/14_architecture_visual_stimuli_preprocessed_cat.ipynb): Experiments feature extractor
- [14_architecture_visual_stimuli_preprocessed_shift.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/14_architecture_visual_stimuli_preprocessed_shift.ipynb): Experiments feature extractor
- [16_architecture_face_recognition.ipynb](https://github.com/r1cc4r2o/DhW/blob/main/experiments/14_architecture_visual_stimuli_preprocessed_shift.ipynb): Attempt 1 architecture 2
- The code for the architecture 1 and 2 is in the [main/src](https://github.com/r1cc4r2o/DhW/tree/main/src) folder
---

REMINDER: I released the dataset in a gdrive folder here is the [link](https://drive.google.com/drive/folders/11qvooftCfDQvlYDtIOJpJTIxlppzgWFW?usp=sharing). 