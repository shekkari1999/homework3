import pandas as pd
import torch
import time
import random
import numpy as np
from pathlib import Path

# Fix random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
from preprocess import pad_seq, preprocess, IMDBDataset, split_data
from config import device, criterion, epochs, experiments
from model import MultiLayerRNN, MultiLayerLSTM, MultiLayerBiDrectionalLSTM
import torch.optim as optim
from tqdm import tqdm
from evaluate import evaluate_model
from utils import evaluate_experiment, save_metrics_to_csv, plot_training_loss, plot_accuracy_f1_vs_seq_length, print_summary_table

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

def main():

    df = pd.read_csv(DATA_DIR / 'imdb.csv')
    #do all preprocessing at once and save all the files with .pt(even padding for 25, 50 and 100)
    df_with_ids, word_to_id = preprocess(df['review'])
    ytrain = [0 if sentiment == 'negative' else 1 for sentiment in df['sentiment']]
    ytrain = torch.tensor(ytrain)
    torch.save(ytrain, DATA_DIR / 'train_labels.pt')
    sequence_lengths = [25, 50, 100]
    for seq_len in sequence_lengths:
        padded_sequences = pad_seq(df_with_ids, seq_len)
        padded_sequences = torch.tensor(padded_sequences)
        torch.save(padded_sequences, DATA_DIR / f'train_seqs.{seq_len}.pt')
    print(f'Done with tokenizing and padding')

    # List to store all experiment metrics
    all_metrics = []
    # Store training losses for each experiment (to plot best/worst later)
    experiment_losses = []
    
    for experiment in experiments:
        '''
        For each experiment, read the config and get the corresponding dataset and model config like activation.
        optimizer and clipping will be mentioned in training loop
        '''
        # Print current experiment config
        print(f"Running experiment: {experiment}") 
        # Print current experiment 
        seq_len = experiment['seq_len']
        # Load the appropriate dataset based on sequence length
        imdb_dataset = IMDBDataset(str(DATA_DIR / f'train_seqs.{seq_len}.pt'), str(DATA_DIR / 'train_labels.pt'))
        train_dl, val_dl = split_data(imdb_dataset)

        # +1 for padding index 0
        vocab_size = 10000 + 1 
        if experiment['model'] == 'rnn':
            # Assuming MultiLayerRNN is your RNN implementation
            model = MultiLayerRNN(vocab_size=vocab_size, activation=experiment['activation']) 
        elif experiment['model'] == 'lstm':
            model = MultiLayerLSTM(vocab_size=vocab_size, activation=experiment['activation']) 
        elif experiment['model'] == 'bilstm':
            model = MultiLayerBiDrectionalLSTM(vocab_size=vocab_size, activation=experiment['activation']) 
        model.to(device)

        #### optimizer
        if  experiment['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr = 0.01)
        elif experiment['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr = 0.001)
        elif experiment['optimizer'] == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr = 0.01)

        ## clipping
        clipping = experiment['strategy'] == 'clipping' 
        # Track losses and times for plotting
        train_losses = []
        val_losses = []
        epoch_times = []

        for epoch in range(epochs):
            '''
            set model to train mode at the beggining of each epoch
            '''
            epoch_start_time = time.time()
            
            model.train()
            epoch_loss = 0
            for i, (xb, yb) in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")):
                xb = xb.to(device)
                yb = yb.to(device).float().unsqueeze(1) # Ensure yb is float and has shape (batch_size, 1)
                preds = model(xb)
                optimizer.zero_grad()
                loss = criterion(preds, yb)
                epoch_loss += loss.item() # Use .item() to get the scalar loss value
                loss.backward()

                # Apply gradient clipping if strategy is 'clipping'
                if clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            # set to eval mode
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb = xb.to(device)
                    yb = yb.to(device).float().unsqueeze(1) # Ensure yb is float and has shape (batch_size, 1)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_epoch_loss = epoch_loss / len(train_dl)
            avg_val_loss = val_loss / len(val_dl)
            train_losses.append(avg_epoch_loss)
            val_losses.append(avg_val_loss)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            # Evaluate model using evaluate.py functions
            accuracy, f1 = evaluate_model(model, val_dl, device)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}, Time: {epoch_time:.2f}s")
        
        # After training, get final metrics and save
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        experiment_metrics = evaluate_experiment(model, val_dl, device, experiment, epoch_time=avg_epoch_time)
        all_metrics.append(experiment_metrics)
        
        # Store training losses for this experiment (to plot best/worst later)
        model_name = f"{experiment['model'].upper()}_{experiment['activation']}_{experiment['optimizer']}_seq{experiment['seq_len']}"
        experiment_losses.append({
            'model_name': model_name,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'f1': experiment_metrics['F1'],
            'accuracy': experiment_metrics['Accuracy']
        })
        
        print(f"Completed experiment: {experiment}")
        print(f"Final Accuracy: {experiment_metrics['Accuracy']:.4f}, F1: {experiment_metrics['F1']:.4f}, Avg Epoch Time: {avg_epoch_time:.2f}s\n")
    
    # Save all metrics to CSV
    if all_metrics:
        df = save_metrics_to_csv(all_metrics, 'metrics.csv')
        print(f"\nAll experiments completed! Metrics saved to results/metrics.csv")
        
        # Plot accuracy and F1 vs sequence length
        plot_accuracy_f1_vs_seq_length(df)
        
        # Plot training loss for best and worst models (based on F1 score)
        if len(experiment_losses) > 0:
            # Sort by F1 score to find best and worst
            sorted_experiments = sorted(experiment_losses, key=lambda x: x['f1'])
            worst_exp = sorted_experiments[0]
            best_exp = sorted_experiments[-1]
            
            # Plot worst model
            plot_training_loss(
                worst_exp['train_losses'], 
                worst_exp['val_losses'], 
                epochs, 
                f"{worst_exp['model_name']}_worst"
            )
            
            # Plot best model
            plot_training_loss(
                best_exp['train_losses'], 
                best_exp['val_losses'], 
                epochs, 
                f"{best_exp['model_name']}_best"
            )
            
            print(f"\nPlotted training loss for:")
            print(f"  Worst model (F1={worst_exp['f1']:.4f}): {worst_exp['model_name']}")
            print(f"  Best model (F1={best_exp['f1']:.4f}): {best_exp['model_name']}")
        
        # Print summary table
        print_summary_table(df)

if __name__ == "__main__":
    main()
