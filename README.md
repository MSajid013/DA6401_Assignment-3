# DA6401_Assignment-3
# Muhammad Sajid (MA23M013)
# Seq2Seq Transliteration Model (Urdu - Dakshina Dataset)
This repository contains the code and resources for a sequence-to-sequence (seq2seq) transliteration model developed using the Dakshina dataset for the Urdu language. The primary goal of this project is to transliterate text from Latin script (romanized Urdu) to native Urdu script.

# Overview
Transliteration is the task of converting text from one script to another while preserving phonetic characteristics. This project implements a sequence-to-sequence model with an attention mechanism to improve accuracy and handle ambiguities where the same input characters may have different pronunciations.

# Dataset
The model uses the Dakshina dataset provided by Google Research, specifically the Urdu transliteration pairs. The dataset includes parallel data (Latin → Urdu) suitable for training and evaluating character-level sequence models.

# Model Architecture
The transliteration model follows a typical Encoder-Decoder structure with an Attention mechanism:

- Encoder: Encodes input Latin-script characters into context vectors.

- Decoder: Predicts the corresponding Urdu characters one step at a time using the encoder's context and attention.

Key Components:
- Embedding Layer: Converts input characters to dense vectors.

- Recurrent Layers: LSTM,GRU or RNN layers for both encoder and decoder to model sequences.

- Attention Layer: Enables decoder to focus on specific encoder outputs at each timestep.

Training Strategy:
To improve performance and training efficiency:

Teacher Forcing with probability 0.5.

Bayesian Hyperparameter Optimization using Weights & Biases (WandB) sweeps.

Dropout Regularization to prevent overfitting.

Bidirectional Encoder for capturing both forward and backward context.

# Best Hyperparameters
- input_size = 30         # Number of unique Latin characters
- output_size = 58        # Number of unique Urdu characters
- embed_size = 64
- hidden_size = 128
- encoder_layers = 3
- decoder_layers = 3
- cell_type = 'lstm'
- batch_size = 64
- num_epochs = 10
- drop_prob = 0.2
- learning_rate = 0.001
- bidirectional = True

# Best Hyperparameters (Attention Model)
- input_size = 30         # Number of unique Latin characters
- output_size = 58        # Number of unique Urdu characters
- embed_size = 64
- hidden_size = 128
- encoder_layers = 1
- decoder_layers = 1
- cell_type = 'lstm'
- batch_size = 64
- num_epochs = 10
- drop_prob = 0.3
- learning_rate = 0.001
- bidirectional = True

