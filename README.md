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
- input_vocab_size = 30         # Number of Latin script characters
- output_vocab_size = 58        # Number of Urdu script characters
- embedding_dim = 64           # Embedding dimension for both encoder and decoder
- hidden_size = 128             # Hidden state size for RNN cells
- encoder_layers = 3            # Number of layers in encoder RNN
- decoder_layers = 3            # Number of layers in decoder RNN
- cell_type = 'lstm'            # RNN cell type: 'rnn', 'gru', or 'lstm'
- batch_size = 64               # Batch size during training
- num_epochs = 10               # Total number of training epochs
- dropout = 0.2                 # Dropout probability
- learning_rate = 0.001         # Learning rate for optimizer
- bidirectional = True

# Best Hyperparameters (Attention Model)
- input_vocab_size = 30         # Number of Latin script characters
- output_vocab_size = 58        # Number of Urdu script characters
- embedding_dim = 64           # Embedding dimension for both encoder and decoder
- hidden_size = 128             # Hidden state size for RNN cells
- encoder_layers = 1            # Number of layers in encoder RNN
- decoder_layers = 1            # Number of layers in decoder RNN
- cell_type = 'lstm'            # RNN cell type: 'rnn', 'gru', or 'lstm'
- batch_size = 64               # Batch size during training
- num_epochs = 10               # Total number of training epochs
- dropout = 0.3                 # Dropout probability
- learning_rate = 0.001         # Learning rate for optimizer
- bidirectional = True

# WandB Report Link:
https://wandb.ai/ma23m013-iit-madras/DA6401_Assignment-3/reports/DA6401-Assignment-3--VmlldzoxMjM5MDgzNg?accessToken=xhilhlf90j29ynzwnwy0jb29xf7y9vty1o1u1upl5rpfpsank0ncntti7feeozsj
