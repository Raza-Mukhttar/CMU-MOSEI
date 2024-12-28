# -*- coding: utf-8 -*-
"""Training and Evaluation Pipeline.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HcfMXvhqNJ75G6lqvXdaYBg9Yc5ooL8B

# **Training and Evaluation Pipeline:**
Training and evaluation pipline for a multimodal neural network in PyTorch, leveraging audio, visual, and text inputs. The train_model function optimizes the model parameters over batches using backpropagation, while the evaluate_model function computes performance metrics, such as precision, recall, and F1-score, using sklearn.metrics.classification_report.

The use of GPU acceleration ensures efficient computation for large datasets. This modular implementation provides a robust foundation for multimodal deep learning tasks, facilitating both effective training and detailed evaluation.
"""

from sklearn.metrics import classification_report

def train_model(dataloader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        audio = batch["audio"].to(device)
        visual = batch["visual"].to(device)
        text = batch["text"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(audio, visual, text)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(dataloader, model, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            text = batch["text"].to(device)
            labels = batch["label"].to(device)

            outputs = model(audio, visual, text)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return classification_report(true_labels, predictions)