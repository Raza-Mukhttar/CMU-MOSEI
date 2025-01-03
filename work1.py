import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torchaudio
from torchaudio.transforms import Vol
from sklearn.metrics import accuracy_score

# --- Data Loading ---
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

file_path = './cmu-mosei/unaligned_50.pkl'
data = load_data(file_path)

# --- VAEGAN Implementation (Simplified Example) ---
class VAEGAN(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAEGAN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

# Apply VAEGAN for missing modality augmentation
def apply_vaegan(data, vaegan_model, weights):
    original = torch.tensor(data, dtype=torch.float32)
    generated, _ = vaegan_model(original)
    combined = weights[0] * original + weights[1] * generated
    return combined.detach().numpy()

# --- Data Augmentation ---
def augment_audio(audio):
    sample_rate = 16000
    pitch_shift = random.randint(-2, 2)
    volume = random.uniform(0.7, 1.3)
    audio = torchaudio.transforms.PitchShift(sample_rate, n_steps=pitch_shift)(audio)
    audio = Vol(volume)(audio)
    return audio

# --- Dataset ---
class CMUMOSEIDataset(Dataset):
    def __init__(self, audio, vision, text, labels):
        self.audio = audio
        self.vision = vision
        self.text = text
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = torch.tensor(self.audio[idx], dtype=torch.float32)
        vision = torch.tensor(self.vision[idx], dtype=torch.float32)
        text = torch.tensor(self.text[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return audio, vision, text, label

def prepare_dataloader(audio, vision, text, labels, batch_size):
    dataset = CMUMOSEIDataset(audio, vision, text, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Multimodal Transformer ---
class MultimodalTransformer(nn.Module):
    def __init__(self, input_dim_audio, input_dim_vision, input_dim_text, hidden_dim, num_classes):
        super(MultimodalTransformer, self).__init__()
        self.audio_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim_audio, nhead=1, dim_feedforward=hidden_dim),
            num_layers=2
        )
        self.vision_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim_vision, nhead=1, dim_feedforward=hidden_dim),
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

# --- Training Function ---
def train_model(audio, vision, text, labels, input_dims, hidden_dim, num_classes, batch_size, num_epochs, learning_rate):
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
    train_loader = prepare_dataloader(audio, vision, text, labels, batch_size)
    test_loader = prepare_dataloader(audio,vision, text, labels, batch_size=batch_size)

    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct_train, total_train = 0, 0, 0

        for audio, vision, text, labels in train_loader:
            audio, vision, text, labels = audio.to(device), vision.to(device), text.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(audio, vision, text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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


    return model

# --- Evaluation Function ---
def evaluate_model(model, audio, vision, text, labels, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    eval_loader = prepare_dataloader(audio, vision, text, labels, batch_size)

    correct, total = 0, 0
    with torch.no_grad():
        for audio, vision, text, labels in eval_loader:
            audio, vision, text, labels = audio.to(device), vision.to(device), text.to(device), labels.to(device)
            outputs = model(audio, vision, text)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Evaluation Accuracy: {correct / total:.4f}")

# --- Main Pipeline ---
input_dims = {'audio': 74, 'vision': 35, 'text': 768}
hidden_dim = 128
num_classes = 6
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# VAEGAN model and weights
audio_vaegan = VAEGAN(input_dim=74, latent_dim=32)
vision_vaegan = VAEGAN(input_dim=35, latent_dim=16)
text_vaegan = VAEGAN(input_dim=768, latent_dim=64)
weights = [0.7, 0.3]

# Combine data using VAEGAN
audio_combined = apply_vaegan(data['train']['audio'], audio_vaegan, weights)
vision_combined = apply_vaegan(data['train']['vision'], vision_vaegan, weights)
text_combined = apply_vaegan(data['train']['text'], text_vaegan, weights)
labels = data['train']['classification_labels']

# Train and evaluate
print("Training...")
trained_model = train_model(audio_combined, vision_combined, text_combined, labels, input_dims, hidden_dim, num_classes, batch_size, num_epochs, learning_rate)

print("\nEvaluating...")
test_audio_combined = apply_vaegan(data['test']['audio'], audio_vaegan, weights)
test_vision_combined = apply_vaegan(data['test']['vision'], vision_vaegan, weights)
test_text_combined = apply_vaegan(data['test']['text'], text_vaegan, weights)
test_labels = data['test']['classification_labels']

evaluate_model(trained_model, test_audio_combined, test_vision_combined, test_text_combined, test_labels, batch_size)
