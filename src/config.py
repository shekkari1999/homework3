import torch 
import torch.nn as nn
import random
import numpy as np

# Fix random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = 'cpu' #'mps' if torch.mps.is_available() else 'cpu'
criterion = nn.BCELoss()
hidden_size = 64
num_layers = 2
dropout = 0.5
vocab_size = 10000
emb_dim = 100
seed = 42
batch_size = 32
epochs = 10

experiments = [

# ---- Short seq (25) ----
{ "model": "rnn", "activation": "tanh",    "seq_len": 25, "optimizer": "Adam",    "strategy": "clipping" },
{ "model": "rnn", "activation": "tanh",    "seq_len": 25, "optimizer": "Adam",    "strategy": "no clipping" },
{ "model": "rnn", "activation": "relu",    "seq_len": 25, "optimizer": "RMSprop", "strategy": "clipping" },
{ "model": "rnn", "activation": "relu",    "seq_len": 25, "optimizer": "RMSprop", "strategy": "no clipping" },
{ "model": "rnn", "activation": "sigmoid", "seq_len": 25, "optimizer": "SGD",     "strategy": "clipping" },
{ "model": "rnn", "activation": "sigmoid", "seq_len": 25, "optimizer": "SGD",     "strategy": "no clipping" },

# ---- Medium seq (50) ----
{ "model": "rnn", "activation": "tanh",    "seq_len": 50, "optimizer": "SGD",     "strategy": "clipping" },
{ "model": "rnn", "activation": "tanh",    "seq_len": 50, "optimizer": "SGD",     "strategy": "no clipping" },
{ "model": "rnn", "activation": "relu",    "seq_len": 50, "optimizer": "Adam",    "strategy": "clipping" },
{ "model": "rnn", "activation": "relu",    "seq_len": 50, "optimizer": "Adam",    "strategy": "no clipping" },
{ "model": "rnn", "activation": "sigmoid", "seq_len": 50, "optimizer": "RMSprop", "strategy": "clipping" },
{ "model": "rnn", "activation": "sigmoid", "seq_len": 50, "optimizer": "RMSprop", "strategy": "no clipping" },

# ---- Long seq (100) ----
{ "model": "rnn", "activation": "tanh",    "seq_len": 100, "optimizer": "RMSprop", "strategy": "clipping" },
{ "model": "rnn", "activation": "tanh",    "seq_len": 100, "optimizer": "RMSprop", "strategy": "no clipping" },
{ "model": "rnn", "activation": "relu",    "seq_len": 100, "optimizer": "SGD",     "strategy": "clipping" },
{ "model": "rnn", "activation": "relu",    "seq_len": 100, "optimizer": "SGD",     "strategy": "no clipping" },
{ "model": "rnn", "activation": "relu",    "seq_len": 100, "optimizer": "Adam",    "strategy": "clipping" },
{ "model": "rnn", "activation": "relu",    "seq_len": 100, "optimizer": "Adam",    "strategy": "no clipping" },

# ---- Mixed edge tests ----
{ "model": "rnn", "activation": "tanh",    "seq_len": 50,  "optimizer": "Adam",    "strategy": "clipping" },
{ "model": "rnn", "activation": "relu",    "seq_len": 25,  "optimizer": "SGD",     "strategy": "no clipping" },

# ---- Short seq (25) ----
{ "model": "lstm", "activation": "tanh",    "seq_len": 25, "optimizer": "Adam",    "strategy": "clipping" },
{ "model": "lstm", "activation": "tanh",    "seq_len": 25, "optimizer": "Adam",    "strategy": "no clipping" },
{ "model": "lstm", "activation": "relu",    "seq_len": 25, "optimizer": "RMSprop", "strategy": "clipping" },
{ "model": "lstm", "activation": "relu",    "seq_len": 25, "optimizer": "RMSprop", "strategy": "no clipping" },
{ "model": "lstm", "activation": "sigmoid", "seq_len": 25, "optimizer": "SGD",     "strategy": "clipping" },
{ "model": "lstm", "activation": "sigmoid", "seq_len": 25, "optimizer": "SGD",     "strategy": "no clipping" },

# ---- Medium seq (50) ----
{ "model": "lstm", "activation": "tanh",    "seq_len": 50, "optimizer": "SGD",     "strategy": "clipping" },
{ "model": "lstm", "activation": "tanh",    "seq_len": 50, "optimizer": "SGD",     "strategy": "no clipping" },
{ "model": "lstm", "activation": "relu",    "seq_len": 50, "optimizer": "Adam",    "strategy": "clipping" },
{ "model": "lstm", "activation": "relu",    "seq_len": 50, "optimizer": "Adam",    "strategy": "no clipping" },
{ "model": "lstm", "activation": "sigmoid", "seq_len": 50, "optimizer": "RMSprop", "strategy": "clipping" },
{ "model": "lstm", "activation": "sigmoid", "seq_len": 50, "optimizer": "RMSprop", "strategy": "no clipping" },

# ---- Long seq (100) ----
{ "model": "lstm", "activation": "tanh",    "seq_len": 100, "optimizer": "RMSprop", "strategy": "clipping" },
{ "model": "lstm", "activation": "tanh",    "seq_len": 100, "optimizer": "RMSprop", "strategy": "no clipping" },
{ "model": "lstm", "activation": "relu",    "seq_len": 100, "optimizer": "SGD",     "strategy": "clipping" },
{ "model": "lstm", "activation": "relu",    "seq_len": 100, "optimizer": "SGD",     "strategy": "no clipping" },
{ "model": "lstm", "activation": "relu",    "seq_len": 100, "optimizer": "Adam",    "strategy": "clipping" },
{ "model": "lstm", "activation": "relu",    "seq_len": 100, "optimizer": "Adam",    "strategy": "no clipping" },

# ---- Mixed edge tests ----
{ "model": "lstm", "activation": "tanh",    "seq_len": 50,  "optimizer": "Adam",    "strategy": "clipping" },
{ "model": "lstm", "activation": "relu",    "seq_len": 25,  "optimizer": "SGD",     "strategy": "no clipping" },


# ---- Short seq (25) ----
{ "model": "bilstm", "activation": "tanh",    "seq_len": 25, "optimizer": "Adam",    "strategy": "clipping" },
{ "model": "bilstm", "activation": "tanh",    "seq_len": 25, "optimizer": "Adam",    "strategy": "no clipping" },
{ "model": "bilstm", "activation": "relu",    "seq_len": 25, "optimizer": "RMSprop", "strategy": "clipping" },
{ "model": "bilstm", "activation": "relu",    "seq_len": 25, "optimizer": "RMSprop", "strategy": "no clipping" },
{ "model": "bilstm", "activation": "sigmoid", "seq_len": 25, "optimizer": "SGD",     "strategy": "clipping" },
{ "model": "bilstm", "activation": "sigmoid", "seq_len": 25, "optimizer": "SGD",     "strategy": "no clipping" },

# ---- Medium seq (50) ----
{ "model": "bilstm", "activation": "tanh",    "seq_len": 50, "optimizer": "SGD",     "strategy": "clipping" },
{ "model": "bilstm", "activation": "tanh",    "seq_len": 50, "optimizer": "SGD",     "strategy": "no clipping" },
{ "model": "bilstm", "activation": "relu",    "seq_len": 50, "optimizer": "Adam",    "strategy": "clipping" },
{ "model": "bilstm", "activation": "relu",    "seq_len": 50, "optimizer": "Adam",    "strategy": "no clipping" },
{ "model": "bilstm", "activation": "sigmoid", "seq_len": 50, "optimizer": "RMSprop", "strategy": "clipping" },
{ "model": "bilstm", "activation": "sigmoid", "seq_len": 50, "optimizer": "RMSprop", "strategy": "no clipping" },

# ---- Long seq (100) ----
{ "model": "bilstm", "activation": "tanh",    "seq_len": 100, "optimizer": "RMSprop", "strategy": "clipping" },
{ "model": "bilstm", "activation": "tanh",    "seq_len": 100, "optimizer": "RMSprop", "strategy": "no clipping" },
{ "model": "bilstm", "activation": "relu",    "seq_len": 100, "optimizer": "SGD",     "strategy": "clipping" },
{ "model": "bilstm", "activation": "relu",    "seq_len": 100, "optimizer": "SGD",     "strategy": "no clipping" },
{ "model": "bilstm", "activation": "relu",    "seq_len": 100, "optimizer": "Adam",    "strategy": "clipping" },
{ "model": "bilstm", "activation": "relu",    "seq_len": 100, "optimizer": "Adam",    "strategy": "no clipping" },

# ---- Mixed edge tests ----
{ "model": "bilstm", "activation": "tanh",    "seq_len": 50,  "optimizer": "Adam",    "strategy": "clipping" },
{ "model": "bilstm", "activation": "relu",    "seq_len": 25,  "optimizer": "SGD",     "strategy": "no clipping" }
]