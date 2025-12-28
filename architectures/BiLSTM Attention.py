#!/usr/bin/env python3
"""
================================================================================
ECT DEFECT CLASSIFICATION - STEP 4: RETROSPECTIVE LSTM WITH ATTENTION
================================================================================
Purpose: ECT Data Analysis Pipeline - Deep Learning Classification

DESCRIPTION:
    This script implements a deep learning model for classifying defective vs.
    good/compliant layers in ECT data from powder bed fusion additive manufacturing.
    It uses a Bidirectional LSTM with Multi-Head Attention mechanism to capture
    temporal patterns across consecutive layers.

ARCHITECTURE:
    The model processes temporal sequences of 3 consecutive layers [t-2, t-1, t]
    to classify whether the current layer t is defective or good.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ INPUT: (batch_size, 3 timesteps, 14 features)                          │
    │ └─ 3 timesteps: [t-2, t-1, t], 14 selected features per timestep       │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ BIDIRECTIONAL LSTM:                                                     │
    │ - Hidden size: 128, Num layers: 2, Dropout: 0.2                        │
    │ - Output: (batch_size, 3, 256) [128*2 for bidirectional]               │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ MULTI-HEAD ATTENTION (8 heads):                                         │
    │ - Self-attention on LSTM outputs                                        │
    │ - Learns which timesteps are most important for classification          │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ RESIDUAL CONNECTION + LAYER NORM → MEAN POOLING                        │
    │ - Output: (batch_size, 256)                                             │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ FULLY CONNECTED LAYERS:                                                 │
    │ - FC1: 256 → 128 (ReLU + Dropout 0.3)                                  │
    │ - FC2: 128 → 64  (ReLU + Dropout 0.3)                                  │
    │ - FC3: 64 → 32   (ReLU + Dropout 0.3)                                  │
    │ - FC4: 32 → 1    (Raw logits → Sigmoid for probability)                │
    └─────────────────────────────────────────────────────────────────────────┘

RETROSPECTIVE TEMPORAL APPROACH:
    - Pattern: [t-2, t-1, t] where t is the layer to classify
    - Only PAST context is used (realistic for real-time monitoring)
    - Same geometry + same channel for all timesteps in a sequence
    - Example: To classify DB1 Ch0 Layer 64, uses [Layer 62, 63, 64] of DB1 Ch0

FEATURES EXTRACTED (per layer):
    Statistical features from real and imaginary signals:
    - Mean, standard deviation, variance
    - Skewness, kurtosis
    - FFT mean (frequency domain)
    - Gradient mean (temporal derivative)
    - Peak/valley counts (structural)
    - Energy, power

    Top 14 features are selected using mutual information.

TRAINING CONFIGURATION:
    - Optimizer: Adam (lr=0.001)
    - Loss: BCEWithLogitsLoss (numerically stable)
    - LR Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)
    - Early Stopping: patience=15 epochs
    - Max Epochs: 100
    - Batch Size: 32
    - Data Balancing: Undersampling majority class

INPUT:
    Excel file containing labeled ECT data with columns:
    - geometry_name: Geometry identifier (CB, DB1, DB2, DB3, DB4, XCT)
    - channel: Channel number (0 or 1)
    - buildjob: Build job identifier
    - layer_index: Layer number
    - layer_type: Label ('good', 'compliant', or 'defective')
    - real: Real component of the signal
    - imag: Imaginary component of the signal

OUTPUT:
    Results saved to 4.Results_RETROSPECTIVE/V{n}_RETRO_{timestamp}/:
    - confusion_matrix.png: Classification performance visualization
    - training_history.png: Loss and accuracy curves over epochs
    - attention_weights.png: Visualization of attention focus per timestep
    - misclassified_sequences.xlsx: Detailed analysis of classification errors
    - attention_lstm_model.pth: Saved model weights for inference

================================================================================
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATASET CLASS
# =============================================================================

class TemporalDataset(Dataset):
    """
    PyTorch Dataset for temporal sequences.
    
    Wraps numpy arrays of sequences and labels into a format suitable
    for PyTorch DataLoader.
    """
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class AttentionLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier with Multi-Head Attention mechanism.
    
    Architecture:
        1. Bidirectional LSTM processes temporal sequences
        2. Multi-Head Self-Attention focuses on important timesteps
        3. Residual connection + Layer Normalization for stability
        4. Fully connected layers for classification
    
    Args:
        input_size: Number of features per timestep
        hidden_size: LSTM hidden state size (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.2)
        num_heads: Number of attention heads (default: 8)
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout=0.2, num_heads=8):
        super(AttentionLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Raw logits of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, seq_len, hidden_size*2)
        
        # Multi-Head Self-Attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection + Layer Normalization
        residual_out = self.layer_norm(lstm_out + attn_out)
        
        # Mean pooling across timesteps
        weighted_output = residual_out.mean(dim=1)  # (batch_size, hidden_size*2)
        
        # Fully connected layers
        x = self.relu(self.fc1(weighted_output))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)  # Raw logits
        
        return x
    
    def get_attention_weights(self, x):
        """
        Extract attention weights for interpretability.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Attention weights of shape (batch_size, seq_len) showing
            the importance of each timestep.
        """
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            _, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            # Average across heads: (batch_size, num_heads, seq_len, seq_len) 
            # → (batch_size, seq_len, seq_len)
            attn_weights_avg = attn_weights.mean(dim=1)
            # Average across all timesteps to get importance per timestep
            return attn_weights_avg.mean(dim=-1)  # (batch_size, seq_len)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors validation loss and stops training if no improvement
    is observed for a specified number of epochs.
    
    Args:
        patience: Number of epochs to wait before stopping (default: 15)
        min_delta: Minimum change to qualify as improvement (default: 0.001)
    """
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_normalized_data():
    """
    Load normalized ECT data from Excel file.
    
    Filters data to include only layers from index 40 onwards,
    as earlier layers may contain initialization artifacts.
    
    Returns:
        DataFrame containing the loaded and filtered data
    """
    print("📂 Loading normalized data...")
    
    data_path = '/Users/gianalbertocoldani/Desktop/Detrendare/All Boxes Universal LABELING and DETAIL.xlsx'
    
    try:
        df = pd.read_excel(data_path)
        print(f"📊 Dataset loaded: {len(df)} datapoints")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Filter only layers from 40 onwards
        df = df[df['layer_index'] >= 40]
        print(f"📊 After filtering (layer >= 40): {len(df)} datapoints")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ File not found: {data_path}")
        return pd.DataFrame()


def extract_features_per_geometry(df):
    """
    Extract statistical features for each geometry, channel, and layer combination.
    
    For each layer, computes features from both real and imaginary signal components:
    - Statistical moments (mean, std, variance, skewness, kurtosis)
    - Frequency domain (FFT mean)
    - Temporal derivatives (gradient mean)
    - Structural features (peaks, valleys count)
    - Energy metrics (energy, power)
    
    Args:
        df: DataFrame containing raw ECT data
        
    Returns:
        DataFrame with one row per layer containing extracted features
    """
    print("🔧 Extracting features per geometry and channel...")
    
    all_features = []
    
    for (geometry, channel, buildjob), group in df.groupby(['geometry_name', 'channel', 'buildjob']):
        print(f"📊 Processing {geometry} Channel {channel} {buildjob}: {len(group)} datapoints")
        
        for layer, layer_data in group.groupby('layer_index'):
            real_values = layer_data['real'].values
            imag_values = layer_data['imag'].values
            
            # Handle cases with insufficient data
            if len(real_values) == 0 or len(imag_values) == 0:
                continue
            
            features = {
                'geometry_name': geometry,
                'channel': channel,
                'buildjob': buildjob,
                'layer_index': layer,
                'layer_type': layer_data['layer_type'].iloc[0],
                'Defect': 1 if layer_data['layer_type'].iloc[0] == 'defective' else 0,
                
                # Statistical features - Real component
                'mean_real': np.mean(real_values),
                'std_real': np.std(real_values),
                'var_real': np.var(real_values),
                'skewness_real': pd.Series(real_values).skew() if len(real_values) > 2 and np.std(real_values) > 0 else 0.0,
                'kurtosis_real': pd.Series(real_values).kurtosis() if len(real_values) > 2 and np.std(real_values) > 0 else 0.0,
                
                # Statistical features - Imaginary component
                'mean_imag': np.mean(imag_values),
                'std_imag': np.std(imag_values),
                'var_imag': np.var(imag_values),
                'skewness_imag': pd.Series(imag_values).skew() if len(imag_values) > 2 and np.std(imag_values) > 0 else 0.0,
                'kurtosis_imag': pd.Series(imag_values).kurtosis() if len(imag_values) > 2 and np.std(imag_values) > 0 else 0.0,
                
                # Frequency domain features (FFT)
                'fft_real': np.abs(np.fft.fft(real_values)).mean(),
                'fft_imag': np.abs(np.fft.fft(imag_values)).mean(),
                
                # Temporal derivative features
                'gradient_real': np.gradient(real_values).mean() if len(real_values) > 1 else 0.0,
                'gradient_imag': np.gradient(imag_values).mean() if len(imag_values) > 1 else 0.0,
                
                # Structural features - peaks and valleys
                'peaks_real': len(np.where(np.diff(np.sign(np.diff(real_values))) < 0)[0]) if len(real_values) > 2 else 0.0,
                'valleys_real': len(np.where(np.diff(np.sign(np.diff(real_values))) > 0)[0]) if len(real_values) > 2 else 0.0,
                'peaks_imag': len(np.where(np.diff(np.sign(np.diff(imag_values))) < 0)[0]) if len(imag_values) > 2 else 0.0,
                'valleys_imag': len(np.where(np.diff(np.sign(np.diff(imag_values))) > 0)[0]) if len(imag_values) > 2 else 0.0,
                
                # Placeholder for KernelPCA features
                'kpca_real': 0.0,
                'kpca_imag': 0.0,
                
                # Energy and power features
                'energy_real': np.sum(real_values**2),
                'energy_imag': np.sum(imag_values**2),
                'power_real': np.mean(real_values**2),
                'power_imag': np.mean(imag_values**2),
            }
            
            all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    print(f"📊 Features extracted: {len(features_df)} sequences")
    print(f"📊 Label distribution: {features_df['Defect'].value_counts().to_dict()}")
    
    return features_df


def create_retrospective_temporal_sequences(features_df, sequence_length=3):
    """
    Create RETROSPECTIVE temporal sequences using pattern [t-2, t-1, t].
    
    For each layer t to classify, creates a sequence containing:
    - t-2: Two layers before
    - t-1: One layer before  
    - t: Current layer (label comes from this layer)
    
    Only consecutive layers within the same geometry/channel/buildjob
    are used to form valid sequences.
    
    Args:
        features_df: DataFrame with extracted features per layer
        sequence_length: Number of timesteps (default: 3)
        
    Returns:
        Tuple of (sequences, labels, metadata):
        - sequences: numpy array of shape (n_sequences, seq_length, n_features)
        - labels: numpy array of shape (n_sequences,)
        - metadata: list of dicts with geometry/channel/layer info
    """
    print(f"🔄 Creating RETROSPECTIVE temporal sequences (length: {sequence_length})...")
    print("📊 Pattern: [t-2, t-1, t] where t is the layer to classify")
    
    sequences = []
    labels = []
    metadata = []
    
    # Group by geometry, channel AND buildjob
    for (geometry, channel, buildjob), group in features_df.groupby(['geometry_name', 'channel', 'buildjob']):
        group_sorted = group.sort_values('layer_index')
        layers = group_sorted['layer_index'].values
        
        # Find consecutive layer groups
        consecutive_groups = []
        current_group = [layers[0]]
        
        for i in range(1, len(layers)):
            if layers[i] == layers[i-1] + 1:  # Consecutive
                current_group.append(layers[i])
            else:  # Gap found
                if len(current_group) >= sequence_length:
                    consecutive_groups.append(current_group)
                current_group = [layers[i]]
        
        # Add the last group if it's long enough
        if len(current_group) >= sequence_length:
            consecutive_groups.append(current_group)
        
        # Create RETROSPECTIVE sequences from each consecutive group
        for group_layers in consecutive_groups:
            group_data = group_sorted[group_sorted['layer_index'].isin(group_layers)]
            
            # Need at least 3 layers for [t-2, t-1, t] pattern
            for i in range(2, len(group_layers)):
                # RETROSPECTIVE pattern: [t-2, t-1, t]
                sequence_layers = [
                    group_layers[i-2],  # t-2
                    group_layers[i-1],  # t-1
                    group_layers[i]     # t (current layer to classify)
                ]
                
                sequence_data = group_data[group_data['layer_index'].isin(sequence_layers)]
                sequence_data = sequence_data.sort_values('layer_index')
                
                if len(sequence_data) == sequence_length:
                    # Extract features for the sequence
                    feature_cols = [col for col in sequence_data.columns if col not in 
                                 ['geometry_name', 'channel', 'buildjob', 'layer_index', 'layer_type', 'Defect']]
                    
                    sequence_features = sequence_data[feature_cols].values
                    label = sequence_data.iloc[-1]['Defect']  # Label from current layer t
                    
                    sequences.append(sequence_features)
                    labels.append(label)
                    metadata.append({
                        'geometry': geometry,
                        'channel': channel,
                        'buildjob': buildjob,
                        'layer': group_layers[i],
                        'layer_type': sequence_data.iloc[-1]['layer_type']
                    })
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    print(f"📊 RETROSPECTIVE sequences created: {len(sequences)}")
    print(f"📊 Sequence shape: {sequences.shape}")
    print(f"📊 Label distribution: {np.bincount(labels.astype(int))}")
    
    return sequences, labels, metadata


def balance_sequences(sequences, labels, metadata):
    """
    Balance sequences by undersampling the majority class.
    
    Ensures equal representation of good/compliant and defective samples
    to prevent model bias toward the majority class.
    
    Args:
        sequences: numpy array of sequences
        labels: numpy array of labels
        metadata: list of metadata dicts
        
    Returns:
        Tuple of (balanced_sequences, balanced_labels, balanced_metadata)
    """
    print("⚖️ Balancing sequences...")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame({
        'sequence_idx': range(len(sequences)),
        'label': labels,
        'geometry': [m['geometry'] for m in metadata],
        'channel': [m['channel'] for m in metadata],
        'buildjob': [m['buildjob'] for m in metadata],
        'layer': [m['layer'] for m in metadata],
        'layer_type': [m['layer_type'] for m in metadata]
    })
    
    # Count good/compliant and defective
    good_count = len(df[(df['layer_type'] == 'good') | (df['layer_type'] == 'compliant')])
    defective_count = len(df[df['layer_type'] == 'defective'])
    
    print(f"📊 Before balancing: Good/Compliant={good_count}, Defective={defective_count}")
    
    if good_count > defective_count:
        # Sample good sequences to match defective count
        good_df = df[(df['layer_type'] == 'good') | (df['layer_type'] == 'compliant')]
        sampled_good = good_df.sample(n=defective_count, random_state=42)
        balanced_df = pd.concat([sampled_good, df[df['layer_type'] == 'defective']])
    else:
        # Sample defective sequences to match good count
        defective_df = df[df['layer_type'] == 'defective']
        sampled_defective = defective_df.sample(n=good_count, random_state=42)
        balanced_df = pd.concat([df[(df['layer_type'] == 'good') | (df['layer_type'] == 'compliant')], 
                                sampled_defective])
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Extract balanced data
    balanced_indices = balanced_df['sequence_idx'].values
    balanced_sequences = sequences[balanced_indices]
    balanced_labels = labels[balanced_indices]
    balanced_metadata = [metadata[i] for i in balanced_indices]
    
    print(f"📊 After balancing: {len(balanced_sequences)} sequences")
    print(f"📊 Balanced label distribution: {np.bincount(balanced_labels.astype(int))}")
    
    return balanced_sequences, balanced_labels, balanced_metadata


def feature_selection(X, y, top_k=14):
    """
    Select top features using mutual information scoring.
    
    Mutual information measures the dependency between each feature
    and the target variable, selecting features with highest predictive power.
    
    Args:
        X: DataFrame of features
        y: Array of labels
        top_k: Number of top features to select (default: 14)
        
    Returns:
        Tuple of (top_features_indices, top_features_names)
    """
    print(f"🔍 Selecting top {top_k} features using mutual information...")
    
    # Handle NaN values
    X = X.fillna(0)
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y)
    
    # Get top features
    top_features_idx = np.argsort(mi_scores)[-top_k:]
    top_features_names = X.columns[top_features_idx]
    
    print(f"📊 Top {top_k} features:")
    for i, (idx, name) in enumerate(zip(top_features_idx, top_features_names)):
        print(f"   {i+1:2d}. {name}: {mi_scores[idx]:.4f}")
    
    return top_features_idx, top_features_names


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_model(model, train_loader, test_loader, device, results_dir):
    """
    Train the Attention LSTM model with learning rate scheduling and early stopping.
    
    Args:
        model: AttentionLSTMClassifier model
        train_loader: DataLoader for training data
        test_loader: DataLoader for validation data
        device: torch device (cuda or cpu)
        results_dir: Directory to save training plots
        
    Returns:
        Trained model
    """
    print("🚀 Training Attention LSTM model...")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      patience=10, factor=0.5)
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(100):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted.squeeze() == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), labels)
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted.squeeze() == labels).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/100:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, results_dir)
    
    return model


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, results_dir):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        results_dir: Directory to save the plot
        
    Returns:
        Confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Good', 'Defective'], 
                yticklabels=['Good', 'Defective'])
    plt.title('Confusion Matrix - RETROSPECTIVE Temporal LSTM with Attention')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def plot_training_history(train_losses, val_losses, train_accs, val_accs, results_dir):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        results_dir: Directory to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss - RETROSPECTIVE Temporal with Attention')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy - RETROSPECTIVE Temporal with Attention')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_weights(model, test_loader, results_dir, num_samples=5):
    """
    Visualize attention weights to show which timesteps the model focuses on.
    
    Args:
        model: Trained AttentionLSTMClassifier
        test_loader: DataLoader for test data
        results_dir: Directory to save the plot
        num_samples: Number of samples to visualize (default: 5)
    """
    print("🎯 Plotting attention weights...")
    
    model.eval()
    attention_samples = []
    
    with torch.no_grad():
        for i, (sequences, labels) in enumerate(test_loader):
            if i >= num_samples:
                break
            
            sequences = sequences.to(next(model.parameters()).device)
            attention_weights = model.get_attention_weights(sequences)
            
            # Convert to numpy for plotting
            attention_np = attention_weights.cpu().numpy()
            attention_samples.append(attention_np[0])  # Take first sample from batch
    
    # Plot attention weights
    plt.figure(figsize=(12, 8))
    
    for i, weights in enumerate(attention_samples):
        plt.subplot(num_samples, 1, i+1)
        timesteps = ['t-2', 't-1', 't']
        plt.bar(timesteps, weights)
        plt.title(f'Sample {i+1} - Attention Weights')
        plt.ylabel('Attention Weight')
        plt.xlabel('Timestep')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'attention_weights.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Attention weights plot saved!")


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_misclassified_sequences(model, test_loader, metadata, test_indices, results_dir):
    """
    Analyze and export details of misclassified sequences.
    
    Identifies which sequences were incorrectly classified and saves
    detailed information for further analysis.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        metadata: List of metadata dicts for all sequences
        test_indices: Indices of test sequences in the balanced dataset
        results_dir: Directory to save results
        
    Returns:
        DataFrame with misclassified sequence details
    """
    print("\n🔍 ANALYZING MISCLASSIFIED SEQUENCES")
    print("=" * 60)
    
    model.eval()
    misclassified_sequences = []
    
    with torch.no_grad():
        for batch_idx, (sequences, labels) in enumerate(test_loader):
            sequences = sequences.to(next(model.parameters()).device)
            outputs = model(sequences)
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            
            for idx in range(len(labels)):
                if predicted[idx].item() != labels[idx].item():
                    # Get the actual index in the balanced dataset
                    actual_idx = test_indices[batch_idx * test_loader.batch_size + idx]
                    meta = metadata[actual_idx]
                    
                    confidence = torch.sigmoid(outputs[idx]).item()
                    true_label = 'Good' if labels[idx].item() == 0 else 'Defective'
                    predicted_label = 'Good' if predicted[idx].item() == 0 else 'Defective'
                    
                    misclassified_sequences.append({
                        'geometry_name': meta['geometry'],
                        'channel': meta['channel'],
                        'layer_index': meta['layer'],
                        'layer_type': meta['layer_type'],
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'confidence': confidence
                    })
    
    if misclassified_sequences:
        misclassified_df = pd.DataFrame(misclassified_sequences)
        misclassified_df.to_excel(os.path.join(results_dir, 'misclassified_sequences.xlsx'), 
                                  index=False)
        
        print(f"📊 Total misclassified sequences: {len(misclassified_sequences)}")
        print(f"📊 Misclassified by geometry:")
        print(misclassified_df['geometry_name'].value_counts().to_string())
        print(f"📊 Misclassified by layer type:")
        print(misclassified_df['layer_type'].value_counts().to_string())
        
        return misclassified_df
    else:
        print("✅ No misclassified sequences found!")
        return pd.DataFrame()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_results_folder():
    """
    Create versioned results folder with timestamp.
    
    Creates a folder structure: 4.Results_RETROSPECTIVE/V{n}_RETRO_{timestamp}/
    where n is automatically incremented based on existing versions.
    
    Returns:
        Path to the created results directory
    """
    base_dir = '/Users/gianalbertocoldani/Desktop/Detrendare/4.Results_RETROSPECTIVE'
    
    # Check existing versions
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    existing_versions = []
    for item in os.listdir(base_dir):
        if item.startswith('V') and os.path.isdir(os.path.join(base_dir, item)):
            try:
                version_part = item.split('_')[0]  # Extract "V1" from "V1_RETRO_timestamp"
                version_num = int(version_part[1:])  # Remove 'V' and convert to int
                existing_versions.append(version_num)
            except (ValueError, IndexError):
                continue
    
    # Determine next version
    if existing_versions:
        next_version = max(existing_versions) + 1
    else:
        next_version = 1
    
    # Create folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    version_folder = f"V{next_version}_RETRO_{timestamp}"
    results_dir = os.path.join(base_dir, version_folder)
    
    os.makedirs(results_dir)
    print(f"📁 Results folder created: {results_dir}")
    return results_dir


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main execution function for the RETROSPECTIVE LSTM with Attention pipeline.
    
    Executes the complete workflow:
    1. Load and preprocess data
    2. Extract features per layer
    3. Create retrospective temporal sequences
    4. Balance dataset
    5. Select top features
    6. Train model
    7. Evaluate and visualize results
    """
    print("🚀 ECT DEFECT CLASSIFICATION - RETROSPECTIVE TEMPORAL LSTM WITH ATTENTION")
    print("=" * 80)
    print("🧠 Features: Bidirectional LSTM + Multi-Head Attention + RETROSPECTIVE Sequences")
    print("🎯 Pattern: [t-2, t-1, t] with attention weights")
    print("=" * 80)
    
    # Create results folder
    results_dir = create_results_folder()
    
    # Load data
    df = load_normalized_data()
    if df.empty:
        print("❌ No data loaded. Exiting.")
        return
    
    # Extract features
    features_df = extract_features_per_geometry(df)
    
    # Create RETROSPECTIVE temporal sequences
    sequences, labels, metadata = create_retrospective_temporal_sequences(
        features_df, sequence_length=3)
    
    # Balance sequences
    sequences_balanced, labels_balanced, metadata_balanced = balance_sequences(
        sequences, labels, metadata)
    
    # Feature selection using last timestep
    feature_cols = [col for col in features_df.columns if col not in 
                   ['geometry_name', 'channel', 'buildjob', 'layer_index', 'layer_type', 'Defect']]
    
    last_features = sequences_balanced[:, -1, :]  # Shape: (n_sequences, n_features)
    X = pd.DataFrame(last_features, columns=feature_cols)
    
    top_features_idx, top_features_names = feature_selection(X, labels_balanced, top_k=14)
    
    # Select top features from all timesteps
    sequences_filtered = sequences_balanced[:, :, top_features_idx]
    
    print(f"📊 Final sequence shape: {sequences_filtered.shape}")
    print(f"📊 Selected features: {top_features_names.tolist()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        sequences_filtered, labels_balanced, np.arange(len(sequences_filtered)),
        test_size=0.2, random_state=42, stratify=labels_balanced
    )
    
    print(f"📊 Training set: {len(X_train)} sequences")
    print(f"📊 Test set: {len(X_test)} sequences")
    
    # Create datasets and dataloaders
    train_dataset = TemporalDataset(X_train, y_train)
    test_dataset = TemporalDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    model = AttentionLSTMClassifier(
        input_size=len(top_features_idx),
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        num_heads=8
    ).to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trained_model = train_model(model, train_loader, test_loader, device, results_dir)
    
    # Evaluate model
    print("\n📊 EVALUATING MODEL")
    print("=" * 60)
    
    trained_model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = trained_model(sequences)
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🎯 Recall: {recall:.4f}")
    print(f"🎯 F1-Score: {f1:.4f}")
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(all_labels, all_predictions, results_dir)
    
    # Plot attention weights
    plot_attention_weights(trained_model, test_loader, results_dir, num_samples=5)
    
    # Analyze misclassified sequences
    misclassified_df = analyze_misclassified_sequences(
        trained_model, test_loader, metadata_balanced, test_indices, results_dir)
    
    # Save model
    torch.save(trained_model.state_dict(), 
               os.path.join(results_dir, 'attention_lstm_model.pth'))
    
    print(f"\n✅ Training completed!")
    print(f"📁 Results saved in: {results_dir}")
    print(f"🎯 Final Accuracy: {accuracy:.4f}")
    print(f"🎯 Final F1-Score: {f1:.4f}")


if __name__ == "__main__":
    main()
