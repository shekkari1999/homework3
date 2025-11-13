
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from evaluate import evaluate_model

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
PLOTS_DIR = RESULTS_DIR / 'plots'

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

def save_metrics_to_csv(metrics_list, filename='metrics.csv'):
    """
    Save evaluation metrics to a CSV file.
    
    Args:
        metrics_list (list of dict): List of dictionaries containing metrics for each experiment
        filename (str): Name of the CSV file to save
    """
    df = pd.DataFrame(metrics_list)
    filepath = RESULTS_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")
    return df


def load_metrics_from_csv(filename='metrics.csv'):
    """
    Load evaluation metrics from a CSV file.
    
    Args:
        filename (str): Name of the CSV file to load
    
    Returns:
        pd.DataFrame: DataFrame containing the metrics
    """
    filepath = RESULTS_DIR / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        return df
    else:
        print(f"File {filepath} does not exist.")
        return None


def plot_accuracy_f1_vs_seq_length(df, save_path=None):
    """
    Plot Accuracy and F1 score vs Sequence Length.
    
    Args:
        df (pd.DataFrame): DataFrame containing metrics with 'seq_len' or 'Seq Length', 'Accuracy', 'F1' columns
        save_path (str or Path): Path to save the plot. If None, saves to results/plots/
    """
    if save_path is None:
        save_path = PLOTS_DIR / 'accuracy_f1_vs_seq_length.png'
    
    # Handle different column name formats
    df_plot = df.copy()
    if 'Seq Length' in df_plot.columns and 'seq_len' not in df_plot.columns:
        df_plot = df_plot.rename(columns={'Seq Length': 'seq_len'})
    
    # Group by sequence length and calculate mean
    grouped = df_plot.groupby('seq_len')[['Accuracy', 'F1']].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(grouped['seq_len'], grouped['Accuracy'], marker='o', label='Accuracy', linewidth=2)
    ax.plot(grouped['seq_len'], grouped['F1'], marker='s', label='F1 Score', linewidth=2)
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Accuracy and F1 Score vs Sequence Length', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(grouped['seq_len'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_training_loss(train_losses, val_losses, epochs, model_name, save_path=None):
    """
    Plot training loss vs epochs.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        epochs (int): Number of epochs
        model_name (str): Name of the model for the title
        save_path (str or Path): Path to save the plot. If None, saves to results/plots/
    """
    if save_path is None:
        save_path = PLOTS_DIR / f'training_loss_{model_name.replace(" ", "_")}.png'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    epoch_numbers = range(1, len(train_losses) + 1)
    ax.plot(epoch_numbers, train_losses, marker='o', label='Train Loss', linewidth=2)
    ax.plot(epoch_numbers, val_losses, marker='s', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training Loss vs Epochs - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def create_summary_table(df):
    """
    Create a formatted summary table from metrics DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing metrics
    
    Returns:
        pd.DataFrame: Formatted summary table
    """
    # Select and order columns for the summary table
    summary_cols = ['Model', 'Activation', 'Optimizer', 'Seq Length', 'Grad Clipping', 
                    'Accuracy', 'F1', 'Epoch Time (s)']
    
    # Map column names if needed
    column_mapping = {
        'model': 'Model',
        'activation': 'Activation',
        'optimizer': 'Optimizer',
        'seq_len': 'Seq Length',
        'Seq Length': 'Seq Length',  # Already formatted
        'strategy': 'Grad Clipping',
        'Accuracy': 'Accuracy',
        'F1': 'F1',
        'epoch_time': 'Epoch Time (s)',
        'Epoch Time (s)': 'Epoch Time (s)'  # Already formatted
    }
    
    # Create a copy and rename columns
    summary_df = df.copy()
    # Only rename columns that exist and need renaming
    rename_dict = {k: v for k, v in column_mapping.items() if k in summary_df.columns and k != v}
    summary_df = summary_df.rename(columns=rename_dict)
    
    # Format strategy column
    if 'Grad Clipping' in summary_df.columns:
        summary_df['Grad Clipping'] = summary_df['Grad Clipping'].apply(
            lambda x: 'Yes' if str(x).lower() == 'clipping' else 'No'
        )
    
    # Capitalize model names
    if 'Model' in summary_df.columns:
        summary_df['Model'] = summary_df['Model'].str.upper()
    
    # Round numeric columns
    numeric_cols = ['Accuracy', 'F1', 'Epoch Time (s)']
    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(4)
    
    # Select only the columns we want
    available_cols = [col for col in summary_cols if col in summary_df.columns]
    summary_df = summary_df[available_cols]
    
    return summary_df


def print_summary_table(df):
    """
    Print a nicely formatted summary table.
    
    Args:
        df (pd.DataFrame): DataFrame containing metrics
    """
    summary_df = create_summary_table(df)
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100 + "\n")


def evaluate_experiment(model, dataloader, device, experiment_config, epoch_time=None):
    """
    Evaluate a single experiment and return metrics dictionary.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation
        device (torch.device): Device to run on
        experiment_config (dict): Configuration dictionary for the experiment
        epoch_time (float): Average time per epoch in seconds (optional)
    
    Returns:
        dict: Dictionary containing all metrics and experiment configuration
    """
    accuracy, f1 = evaluate_model(model, dataloader, device)
    
    metrics = {
        'Model': experiment_config.get('model', 'unknown'),
        'Activation': experiment_config.get('activation', 'unknown'),
        'Optimizer': experiment_config.get('optimizer', 'unknown'),
        'Seq Length': experiment_config.get('seq_len', 'unknown'),
        'Grad Clipping': experiment_config.get('strategy', 'unknown'),
        'Accuracy': accuracy,
        'F1': f1,
    }
    
    if epoch_time is not None:
        metrics['Epoch Time (s)'] = epoch_time
    
    return metrics
