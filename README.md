# Comparative Analysis of RNN Architectures for Sentiment Classification

This project implements and evaluates multiple Recurrent Neural Network (RNN) architectures for sentiment classification on the IMDb Movie Review Dataset.

## Getting Started

### Clone the Repository

```bash
git clone <repository-url>
cd homework3
```

Replace `<repository-url>` with your actual repository URL.

### Python Version
- Python >= 3.11

### Virtual Environment Setup

It's recommended to use a virtual environment to isolate project dependencies.

#### Option 1: Using `uv` (Recommended)

If you have `uv` installed:

```bash
# Create and activate virtual environment
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate

# Install dependencies
uv sync
```

#### Option 2: Using Standard `venv`

If you prefer using Python's built-in `venv`:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: After activation, you should see `(.venv)` in your terminal prompt.

### Deactivate Virtual Environment

When you're done working, you can deactivate the virtual environment:

```bash
deactivate
```

## Setup Instructions

### Dependencies

If you haven't installed dependencies during virtual environment setup, you can install them now:

**Using pip:**
```bash
pip install -r requirements.txt
```

**Using uv:**
```bash
uv sync
```

### Dependencies List
- torch >= 2.9.0
- pandas >= 2.3.3
- matplotlib >= 3.8.0
- nltk >= 3.9.2
- scikit-learn >= 1.7.2
- tqdm >= 4.66.0
- ipykernel >= 7.1.0

## Project Structure

```
├── data/
│   ├── imdb.csv              # IMDb dataset (50,000 reviews)
│   ├── train_labels.pt       # Preprocessed labels
│   └── train_seqs.*.pt      # Preprocessed sequences (25, 50, 100)
├── src/
│   ├── preprocess.py         # Data preprocessing functions
│   ├── model.py              # RNN, LSTM, and BiLSTM implementations
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation functions
│   ├── utils.py              # Utility functions (plotting, metrics)
│   └── config.py             # Configuration and hyperparameters
├── results/
│   ├── metrics.csv           # Experiment results
│   └── plots/                # Generated plots
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## How to Run

### 1. Prepare the Dataset
Place the `imdb.csv` file in the `data/` directory. The dataset should contain 50,000 reviews with columns:
- `review`: Text of the movie review
- `sentiment`: Either 'positive' or 'negative'

### 2. Run Training and Evaluation

Make sure your virtual environment is activated, then:

```bash
cd src
python train.py
```

Or from the project root:

```bash
python src/train.py
```

The script will:
1. Preprocess the data (lowercase, remove punctuation, tokenize, keep top 10k words)
2. Create padded sequences for lengths 25, 50, and 100
3. Run all experiments defined in `config.py`
4. Generate metrics and plots automatically

### 3. Expected Runtime
- **Hardware**: CPU only (configurable in `config.py`)
- **Runtime**: Depends on number of experiments and epochs
  - Each experiment: ~5-15 minutes per epoch (CPU)
  - Total experiments: 60+ configurations
  - Estimated total time: 10-20 hours (depending on CPU)

### 4. Output Files

After training completes, you'll find:

**Results:**
- `results/metrics.csv`: Summary table with all experiment results
  - Columns: Model, Activation, Optimizer, Seq Length, Grad Clipping, Accuracy, F1, Epoch Time (s)

**Plots:**
- `results/plots/accuracy_f1_vs_seq_length.png`: Accuracy and F1 vs sequence length
- `results/plots/training_loss_{MODEL}_best.png`: Training loss for best model
- `results/plots/training_loss_{MODEL}_worst.png`: Training loss for worst model

## Configuration

Edit `src/config.py` to modify:
- Number of epochs (default: 10)
- Batch size (default: 32)
- Hidden size (default: 64)
- Number of layers (default: 2)
- Dropout rate (default: 0.5)
- Embedding dimension (default: 100)
- Experiment configurations

## Reproducibility

Random seeds are fixed in `config.py` and `train.py`:
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- Python random: `random.seed(42)`

## Model Architectures

The project evaluates:
- **RNN**: Vanilla Recurrent Neural Network
- **LSTM**: Long Short-Term Memory
- **BiLSTM**: Bidirectional LSTM

Each model uses:
- Embedding layer (100 dimensions)
- 2 hidden layers (64 units each)
- Dropout (0.5)
- Sigmoid output for binary classification
- Binary cross-entropy loss

## Experimental Variations

The project systematically tests:
- **Architectures**: RNN, LSTM, BiLSTM
- **Activations**: Tanh, ReLU, Sigmoid
- **Optimizers**: Adam, SGD, RMSprop
- **Sequence Lengths**: 25, 50, 100
- **Gradient Clipping**: With and without

## Evaluation Metrics

- **Accuracy**: Classification accuracy
- **F1 Score**: Macro-averaged F1 score
- **Epoch Time**: Training time per epoch in seconds

## Hardware Requirements

- **CPU**: Any modern CPU (tested on CPU-only setup)
- **RAM**: Minimum 8GB recommended
- **Storage**: ~500MB for dataset and results

## Notes

- The dataset is split 50/50 (25k training, 25k testing) as per requirements
- All preprocessing is done automatically on first run
- Results are saved incrementally, so you can stop and resume training

