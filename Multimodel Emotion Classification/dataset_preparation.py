# -*- coding: utf-8 -*-
"""Dataset Preparation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HcfMXvhqNJ75G6lqvXdaYBg9Yc5ooL8B

Import all the nessecary libraries that will be used for the following modules:


1.   Dataset preprocessing
2.   Feature extraction
3.   VAEGAN Translator
4.   Multimodal Transformer
5.   Model Training and Evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report

"""# **Dataset Preparation**

Defines a custom PyTorch dataset class (EmotionDataset) and uses it with a DataLoader which extracted the following features:

**audio_features:** Array of audio features (e.g., extracted using COVAREP).

**visual_features:** Array of visual features (e.g., extracted using OpenFace).

**text_features:** Array of text features (e.g., extracted using GloVe embeddings).

**labels:** Array of labels corresponding to the emotion class of each sample.

Converts features (audio, visual, text) and label for the given index into PyTorch tensors
"""

class EmotionDataset(Dataset):
    def __init__(self, audio_features, visual_features, text_features, labels):
        self.audio = audio_features
        self.visual = visual_features
        self.text = text_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "audio": torch.tensor(self.audio[idx], dtype=torch.float32),
            "visual": torch.tensor(self.visual[idx], dtype=torch.float32),
            "text": torch.tensor(self.text[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Example data loading
# Replace with the actual paths to your feature files
audio_features = dataset['COVAREP'].data  # Access COVAREP features
visual_features = dataset['OpenFace'].data  # Access OpenFace features
text_features = dataset['glove_vectors'].data  # Access GloVe vectors
labels = dataset['Opinion Segment Labels'].data# Update this with the path to your labels file.

dataset = EmotionDataset(audio_features, visual_features, text_features, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)