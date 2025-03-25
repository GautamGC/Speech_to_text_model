# Natural Language Processing (NLP) Project 
# Speech to text Model 

## Overview
This project implements an end-to-end Automatic Speech Recognition (ASR) system that converts spoken language into text. The system is built using PyTorch and follows a deep learning approach with recurrent neural networks. It takes raw audio as input and produces text transcriptions as output, eliminating the need for separate acoustic and language models that were common in traditional ASR systems.

## Model Used
The speech recognition model uses a deep bidirectional recurrent neural network 
-Input Layer:
--Input dimensions: [batch_size, 1, n_mels=128, time_steps]
--Reshaped to [batch_size, time_steps, n_mels=128]
-Recurrent Layers:
--Type: Bidirectional Gated Recurrent Unit (GRU)
--Number of layers: 3
--Hidden size: 256 per direction (512 combined)
--Dropout: 0.1 between layers (not applied to the last layer)
--Sequence packing: Uses PyTorch's pack_padded_sequence for efficient processing
-Output Layer:
--Fully connected layer: 512 → 29 (vocabulary size)
--Maps RNN outputs to character probabilities
--Applied to each time step independently
-Activation Function:
-Log softmax applied to output layer
-Provides log probabilities for each character at each time step
-Compatible with CTC loss function

## Model Parameters
-**Total Parameters:** Approximately 3-4 million
-**Input Features:** 128 mel-frequency bins
-**Hidden Size:** 256 per direction (512 total for bidirectional)
-**Output Classes:** 29 (26 letters + space + apostrophe + blank symbol)

## Dataset
- **Dataset Name:** LibriSpeech 
- **Training Dataset:** train-clean.tar.gz
- **Size:** 6.3Gb
- **Download Link:** https://www.openslr.org/resources/12/train-clean-100.tar.gz
- **Testing Dataset:** test-clean.tar.gz
- **Size:** 346Mb
- **Download Link:**https://www.openslr.org/resources/12/test-clean.tar.gz
- **Preprocessing Steps:** [Tokenization, stopword removal, stemming/lemmatization]  

## Outcomes
- **Accuracy:** [Mention model performance metrics]
- ![image](https://github.com/user-attachments/assets/899dc2b6-bf82-4baf-a4eb-879fdb7f5d23)
-Implementation:
  Custom levenshtein_distance function computes edit distance
  Calculated on word level, not character level
  Lower WER indicates better performance
- **Evaluation Metrics:** [F1-score, Precision, Recall, BLEU score, etc.]  
- **Inference Time:** [If applicable]  
![image](https://github.com/user-attachments/assets/8525d27b-0b26-447e-bd13-d71cb65e349c)

## Installation
To run this project locally, follow these steps:
Download the dataset from the given download link and chanfge its loaction accordingly.
Run Speech Generator once before running the model.ipynb
Run the Model.ipynb

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
pip install -r requirements.txt
