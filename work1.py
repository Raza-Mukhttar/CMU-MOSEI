import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

import random
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset which already downloaded in the directory
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

file_path = './cmu-mosei/unaligned_50.pkl'
data = load_data(file_path)

# Inspect the data structure
print("Keys in dataset:", data.keys())

print(data['train'].keys())

train_data = data['train']
print("Keys in train data:", train_data.keys())
print("Example raw text:", train_data['raw_text'][:5])  # View the first 5 entries of raw text
print("Example audio shape:", train_data['audio'][:5])  # View the first 5 audio feature entries
print("Example vision shape:", train_data['vision'][:5])  # View the first 5 vision feature entries
print("Example labels:", train_data['classification_labels'][:5])  # View the first 5 labels

audio_data = train_data['audio']
visual_data = train_data['vision']
text_data = train_data['text']  # Or 'text_bert' if you prefer BERT embeddings
labels = train_data['classification_labels']

dataset = list(zip(audio_data, visual_data, text_data, labels))  # Combine modalities and labels

# Print the shapes and labels for the first 5 entries
for i, (audio, visual, text, label) in enumerate(dataset[:5]):  # Limit to first 5
    print(f"Entry {i + 1}:")
    print(f"Audio Shape: {audio.shape}")
    print(f"Visual Shape: {visual.shape}")
    print(f"Text: {text}")
    print(f"Label: {label}\n")


# Define the embedding dimension and number of heads
embed_dim = 128  # Must be divisible by num_heads
num_heads = 4    # Adjust this to ensure embed_dim % num_heads == 0


# Example Transformer layer
transformer_layer = torch.nn.Transformer(
    d_model=embed_dim,  # Embedding dimension
    nhead=num_heads,    # Number of attention heads
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=512,
    dropout=0.1,
    activation='relu',
)

print(f"embed_dim: {embed_dim}, num_heads: {num_heads}")


if torch.cuda.is_available():
    print("CUDA is available. GPU is being used!")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. CPU is being used.")



import torchaudio
from torchaudio.transforms import Resample, Vol, TimeStretch, MelSpectrogram
from torchvision.transforms import Compose
from random import randint

# Augmentation Functions

def augment_audio(audio):
    # Apply random pitch shift
    sample_rate = 16000  # Assuming 16 kHz sample rate for the audio
    pitch_shift = randint(-2, 2)  # Shift by -2 to 2 semitones
    audio = torchaudio.transforms.PitchShift(sample_rate, n_steps=pitch_shift)(audio)
    
    # Apply random volume adjustment
    volume = random.uniform(0.7, 1.3)  # Random scaling between 70% and 130%
    audio = Vol(volume)(audio)

    # Optionally, add background noise (not implemented here but could be added)
    return audio


def augment_text(text):
    # Simple text augmentation: Randomly shuffle words in the text
    words = text.split()
    if len(words) > 2:
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)

# Step 1: Dataset Preparation
class CMUMOSEIDataset(Dataset):
    def __init__(self, audio, vision, text, labels, augment_audio_fn=None, augment_text_fn=None):
        self.audio = audio
        self.vision = vision
        self.text = text
        self.labels = labels
        self.augment_audio_fn = augment_audio_fn
        self.augment_text_fn = augment_text_fn

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = torch.tensor(self.audio[idx], dtype=torch.float32)
        vision = torch.tensor(self.vision[idx], dtype=torch.float32)
        text = torch.tensor(self.text[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Apply augmentations
        if self.augment_audio_fn:
            audio = self.augment_audio_fn(audio)
        
        if self.augment_text_fn:
            text = self.augment_text_fn(text)

        return audio, vision, text, label

    

# Prepare the dataset
def prepare_dataloader(data, batch_size, augment_audio_fn=None, augment_text_fn=None):
    dataset = CMUMOSEIDataset(
        audio=data['audio'],
        vision=data['vision'],
        text=data['text'],
        labels=data['classification_labels'],
        augment_audio_fn= augment_audio_fn,
        augment_text_fn= augment_text_fn
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = prepare_dataloader(data['train'], batch_size=64, augment_audio_fn=augment_audio, augment_text_fn=augment_text)

# Step 2: Multimodal Transformer
class MultimodalTransformer(nn.Module):
    def __init__(self, input_dim_audio, input_dim_vision, input_dim_text, hidden_dim, num_classes):
        super(MultimodalTransformer, self).__init__()

        self.audio_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim_audio, nhead=2, dim_feedforward=hidden_dim),
            num_layers=2
        )

        self.vision_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim_vision, nhead=5, dim_feedforward=hidden_dim),
            num_layers=2
        )

        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim_text, nhead=2, dim_feedforward=hidden_dim),
            num_layers=2
        )

        self.fusion = nn.Linear(input_dim_audio + input_dim_vision + input_dim_text, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, audio, vision, text):
        audio = audio.permute(1, 0, 2)
        vision = vision.permute(1, 0, 2)
        text = text.permute(1, 0, 2)

        audio_out = self.audio_transformer(audio).mean(dim=0)
        vision_out = self.vision_transformer(vision).mean(dim=0)
        text_out = self.text_transformer(text).mean(dim=0)

        fusion_out = torch.cat([audio_out, vision_out, text_out], dim=1)
        fusion_out = self.fusion(fusion_out)
        logits = self.classifier(fusion_out)

        return logits

def train_model(data, input_dims, hidden_dim, num_classes, batch_size, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalTransformer(
        input_dim_audio=input_dims['audio'],
        input_dim_vision=input_dims['vision'],
        input_dim_text=input_dims['text'],
        hidden_dim=hidden_dim,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = prepare_dataloader(data['train'], batch_size=batch_size)
    test_loader = prepare_dataloader(data['test'], batch_size=batch_size)

    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for audio, vision, text, labels in train_loader:
            audio, vision, text, labels = audio.to(device), vision.to(device), text.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(audio, vision, text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Training accuracy
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            total_loss += loss.item()

        # Calculate metrics for training
        train_loss = total_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate on test set
        model.eval()
        total_test_loss = 0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for audio, vision, text, labels in test_loader:
                audio, vision, text, labels = audio.to(device), vision.to(device), text.to(device), labels.to(device)

                outputs = model(audio, vision, text)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_test += (predicted == labels).sum().item()
                total_test += labels.size(0)

        # Calculate metrics for testing
        test_loss = total_test_loss / len(test_loader)
        test_accuracy = correct_test / total_test
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Hyperparameters
input_dims = {'audio': 74, 'vision': 35, 'text': 768}
hidden_dim = 128
num_classes = 6
batch_size = 64
num_epochs = 10
learning_rate = 0.001

train_model(data, input_dims, hidden_dim, num_classes, batch_size, num_epochs, learning_rate)
