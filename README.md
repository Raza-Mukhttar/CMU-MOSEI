# CMU-MOSEI
Multimodal Emotion Recognition Using Cross-Modal Translation based on three modalities (visual, audio and text) with CMU-MOSEI dataset. Detailed discription of the working and issues faced are discussed below:
1. Dataset Preparation and Preprocessing
# Libraries Installed:
Multimodal-SDK, h5py, and pandas to work with CMU-MOSEI datasets.
CMU-MultimodalSDK for downloading and processing multimodal datasets.
# 1. Data Downloading:
High-level features such as audio (COVAREP), visual (OpenFace), and text (GloVe embeddings) are downloaded using CMU-MultimodalSDK.
Challenges Faced: System crashes during alignment of GloVe vectors with other modalities due to computational limits and incomplete metadata for computational sequences.
# 2. Feature Alignment
Goal: Align all modalities (audio, visual, text) using the same interval and feature set to synchronize them.
Issues: Some dataset entries were not shared across sequences, leading to their removal.
The system failed due to resource limitations during alignment.
# 3. Custom Dataset Class (EmotionDataset)
Functionality: Handles input features (audio, visual, text) and corresponding labels.
Converts these features into PyTorch tensors for model consumption.
Challenges: Errors in loading data due to improper alignment of dataset features.
# 4. Feature Extraction Module
Components: Two separate feedforward neural networks process audio and visual features independently.
Both are mapped into a 300-dimensional embedding space to ensure consistency for fusion.
Architecture: Linear layers with ReLU activation are used for transformation.
# 5. VAEGAN Translator
Structure: Combines Encoder, Decoder, and Discriminator to process input features.
Uses softmax activation in encoding and decoding stages and sigmoid activation in the discriminator.
Purpose: Generates latent representations and reconstructs data for meaningful feature translation.
#6. Multimodal Transformer
Design: Separate Transformer Encoder modules for audio, visual, and text.
Fully connected layers for fusing and classifying concatenated features.
Workflow: Each modality is transformed independently.
Outputs are concatenated and passed through a fusion layer for classification.
# 7. Training and Evaluation Pipeline
Training: Combines VAEGAN Translators and Multimodal Transformer for end-to-end learning.
Uses Adam optimizer and CrossEntropyLoss.
Evaluation: Uses metrics such as precision, recall, and F1-score through sklearn's classification report.
Challenges: Errors in defining the dataloader due to incomplete data preprocessing.
Warnings related to tensor processing in the Transformer module.
# 8. Overall Challenges
Dataset Issues: Misaligned features and incomplete metadata caused interruptions.
Significant data entries were removed due to lack of shared sequences.
Resource Constraints: System crashes during large operations, indicating insufficient computational resources.
# Challenges ca be Resoled by:
Fix dataset alignment by resolving metadata issues.
Use a high-performance system to handle computational requirements.
Debug and test each module independently before full integration to avoid cascading errors.
