# Natural Language Processing (NLP) Project 
# Speech to text Model 

## Overview
This project implements an end-to-end Automatic Speech Recognition (ASR) system that converts spoken language into text. The system is built using PyTorch and follows a deep learning approach with recurrent neural networks. It takes raw audio as input and produces text transcriptions as output, eliminating the need for separate acoustic and language models that were common in traditional ASR systems.

## Model Used
- **Model Type:** [e.g., Transformer-based model (BERT, GPT), LSTM, CNN]  
- **Framework:** [e.g., TensorFlow, PyTorch, Hugging Face]  
- **Pretrained or Custom:** [Specify if it’s a fine-tuned pretrained model or built from scratch]  
- **Training Details:** [Brief info on hyperparameters, epochs, batch size, optimizer]  

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
