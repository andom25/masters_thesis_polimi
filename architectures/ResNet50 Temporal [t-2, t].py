"""
Temporal ResNet50 for Binary Classification of Manufacturing Defects

This script implements a deep learning pipeline for binary classification of manufacturing defects
using temporal sequences of images. The architecture combines a ResNet50 backbone for feature extraction
with an LSTM network and multi-head attention mechanisms to capture temporal dependencies across
consecutive manufacturing layers. The model processes short temporal sequences [t-2, t-1, t] to classify
each layer as either defective or compliant, leveraging spatial and temporal information for improved
classification accuracy. The implementation includes adaptive device detection (GPU/CPU), stratified
data splitting, class-weighted loss functions, and comprehensive evaluation metrics including per-detail
performance analysis.
"""

import os
import sys
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_auc_score
import random
from tqdm import tqdm
import torchvision.models as models
import numpy as np
import warnings
import shutil
from PIL import Image
import pandas as pd
warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION - MODIFY HERE TO CHANGE PATHS
# ============================================================================
# Path to folder containing all images (both compliant and defective)
IMAGES_DIR = r"C:\Users\NUM\Desktop\Coldani_Domini\computer_vision_new\total_images"

# Path to Excel file with labels
LABELS_EXCEL_PATH = r"C:\Users\NUM\Desktop\final_ETH_analysis\images_sequences_labels_v1_POROSITY.xlsx"

# Base path for results
RESULTS_BASE_DIR = r"C:\Users\NUM\Desktop\final_ETH_analysis\trials\ResNet50_temporal_short_v0\251113-t1-ResNet50_temporal_short_run005"

# Results folder (will be created automatically)
RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "251113-t1-ResNet50_temporal_short")
# ============================================================================

# ============================================================================
# CLASSES AND FUNCTIONS - DEFINED OUTSIDE if __name__ == '__main__' FOR MULTIPROCESSING ON WINDOWS
# ============================================================================

# -------------------------------
# CUSTOM DATASET
# -------------------------------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dict, details_dict=None, transform=None):
        self.transform = transform
        self.samples = []
        self.image_paths = []
        self.details_dict = details_dict if details_dict is not None else {}
        
        # Map textual labels to numeric
        # defective -> 0, compliant -> 1
        label_to_num = {'defective': 0, 'compliant': 1}
        
        # Separate images by class
        defective_samples = []
        compliant_samples = []
        
        # Load all images from folder
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in image_files:
            # Search for label in dictionary
            if filename in labels_dict:
                label_text = labels_dict[filename].lower().strip()
                
                # Convert textual label to numeric
                if label_text in label_to_num:
                    label_num = label_to_num[label_text]
                    full_path = os.path.join(images_dir, filename)
                    
                    # Get detail if available
                    detail = self.details_dict.get(filename, 'unknown')
                    if pd.isna(detail):
                        detail = 'unknown'
                    else:
                        detail = str(detail).strip()
                    
                    if label_num == 0:  # defective
                        defective_samples.append((full_path, label_num, filename, detail))
                    else:  # compliant
                        compliant_samples.append((full_path, label_num, filename, detail))
                else:
                    print(f"Warning: Label '{label_text}' not recognized for {filename}. Skipping.")
            else:
                print(f"Warning: Image {filename} not found in Excel file. Skipping.")
        
        # Use ALL available data (unbalanced)
        # Balancing will be handled via class weights in the loss function
        num_defective = len(defective_samples)
        num_compliant_total = len(compliant_samples)
        
        # Combine all images (unbalanced)
        self.samples = defective_samples + compliant_samples
        # Shuffle the dataset
        random.shuffle(self.samples)
        
        # Extract paths for reference
        self.image_paths = [sample[0] for sample in self.samples]
        
        # Create filename -> idx dictionary for fast access (WITHOUT loading images)
        self.filename_to_idx = {filename: idx for idx, (_, _, filename, _) in enumerate(self.samples)}
        
        self.classes = ['defective', 'compliant']
        count_defective = len(defective_samples)
        count_compliant = len(compliant_samples)
        
        print(f"Dataset loaded: {count_defective} defective, {count_compliant} compliant")
        print(f"Total images loaded: {len(self.samples)}")
        print(f"Ratio: {count_defective/len(self.samples)*100:.1f}% defective, {count_compliant/len(self.samples)*100:.1f}% compliant")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label, filename, detail = self.samples[idx]
        image = datasets.folder.default_loader(path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, path, filename, detail
    
    def get_filename(self, idx):
        """Gets the filename without loading the image (OPTIMIZED)"""
        _, _, filename, _ = self.samples[idx]
        return filename

# -------------------------------
# TEMPORAL SEQUENCE WRAPPER
# -------------------------------
class TemporalSequenceWrapper(torch.utils.data.Dataset):
    """Wrapper that creates temporal sequences from the original dataset"""
    def __init__(self, base_dataset, temporal_dict, labels_dict, details_dict=None, sequence_length=3):
        self.base_dataset = base_dataset
        self.temporal_dict = temporal_dict
        self.labels_dict = labels_dict
        self.details_dict = details_dict if details_dict is not None else {}
        self.sequence_length = sequence_length
        
        # Create dictionary: image_name -> idx in base_dataset (OPTIMIZED - without loading images)
        print("Creating image_name -> idx dictionary...")
        if hasattr(base_dataset, 'filename_to_idx'):
            # Use the dictionary already created in CustomDataset (MUCH FASTER)
            self.name_to_idx = base_dataset.filename_to_idx.copy()
            print(f"Dictionary created quickly: {len(self.name_to_idx)} images mapped")
        else:
            # Fallback: create the dictionary (slower, but compatible)
            print("Fallback: slow dictionary creation...")
            self.name_to_idx = {}
            for idx in tqdm(range(len(base_dataset)), desc="Mapping images", ncols=120, colour='cyan'):
                if hasattr(base_dataset, 'get_filename'):
                    filename = base_dataset.get_filename(idx)
                else:
                    _, _, _, filename = base_dataset[idx]
                self.name_to_idx[filename] = idx
        
        # Create temporal sequences
        print("Creating temporal sequences...")
        self.sequences = self._create_temporal_sequences()
        
        self.classes = base_dataset.classes
        
        print(f"Temporal dataset loaded: {len(self.sequences)} sequences")
        count_defective = sum(1 for s in self.sequences if s['label'] == 0)
        count_compliant = sum(1 for s in self.sequences if s['label'] == 1)
        print(f"Sequences: {count_defective} defective, {count_compliant} compliant")
    
    def _create_temporal_sequences(self):
        """Creates SHORT temporal sequences [t-2, t-1, t]"""
        sequences = []
        label_to_num = {'defective': 0, 'compliant': 1}
        
        # Group by buildjob, geometry_name and channel
        print("  Grouping images by (buildjob, geometry, channel)...")
        grouped = {}
        for image_name, label in tqdm(self.labels_dict.items(), desc="  Grouping", ncols=120, colour='yellow', leave=False):
            if image_name not in self.temporal_dict:
                continue
            if image_name not in self.name_to_idx:
                continue
            
            temp_info = self.temporal_dict[image_name]
            buildjob = temp_info['buildjob']
            geometry = temp_info['geometry_name']
            channel = temp_info['channel']
            layer_idx = temp_info['layer_index']
            
            key = (buildjob, geometry, channel)
            if key not in grouped:
                grouped[key] = []
            
            label_text = label.lower().strip()
            if label_text in label_to_num:
                grouped[key].append({
                    'image_name': image_name,
                    'layer_index': layer_idx,
                    'label': label_to_num[label_text],
                    'idx': self.name_to_idx[image_name]
                })
        
        # Create SHORT sequences for each group
        print(f"  Creating sequences from {len(grouped)} groups...")
        for (buildjob, geometry, channel), items in tqdm(grouped.items(), desc="  Creating sequences", ncols=120, colour='green', leave=False):
            # Sort by layer_index
            items_sorted = sorted(items, key=lambda x: x['layer_index'])
            
            # Find consecutive groups
            consecutive_groups = []
            current_group = [items_sorted[0]]
            
            for i in range(1, len(items_sorted)):
                if items_sorted[i]['layer_index'] == items_sorted[i-1]['layer_index'] + 1:
                    current_group.append(items_sorted[i])
                else:
                    if len(current_group) >= self.sequence_length:
                        consecutive_groups.append(current_group)
                    current_group = [items_sorted[i]]
            
            if len(current_group) >= self.sequence_length:
                consecutive_groups.append(current_group)
            
            # Create SHORT sequences [t-2, t-1, t]
            for group in consecutive_groups:
                for i in range(2, len(group)):
                    sequence_layers = [
                        group[i-2],  # t-2
                        group[i-1],  # t-1
                        group[i]     # t (current)
                    ]
                    
                    # Extract indices and label of current layer (t)
                    indices = [item['idx'] for item in sequence_layers]
                    center_label = group[i]['label']
                    center_image_name = group[i]['image_name']
                    
                    # Get detail of current layer
                    center_detail = self.details_dict.get(center_image_name, 'unknown')
                    if pd.isna(center_detail):
                        center_detail = 'unknown'
                    else:
                        center_detail = str(center_detail).strip()
                    
                    sequences.append({
                        'indices': indices,
                        'label': center_label,
                        'buildjob': buildjob,
                        'geometry': geometry,
                        'channel': channel,
                        'center_layer': group[i]['layer_index'],
                        'detail': center_detail,
                        'center_image_name': center_image_name
                    })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        indices = seq_info['indices']
        label = seq_info['label']
        
        # Load images from base dataset (already preprocessed)
        # Returns list of images (padding will be done in collate_fn)
        images = []
        paths = []
        filenames = []
        
        for img_idx in indices:
            image, _, path, filename, detail = self.base_dataset[img_idx]
            images.append(image)
            paths.append(path)
            filenames.append(filename)
        
        # Returns list of images (not stacked) - collate_fn will do padding and stack
        # The detail is that of the central layer (already in seq_info)
        return images, label, paths, seq_info

# -------------------------------
# RESNET MODEL WITH TEMPORAL AGGREGATION
# -------------------------------
class TemporalResNetModel(nn.Module):
    """ResNet50 with temporal aggregation for image sequences"""
    def __init__(self, num_classes=2, pretrained=True, model_name='resnet50', 
                 sequence_length=3, hidden_size=256, num_lstm_layers=2, dropout=0.3):
        super(TemporalResNetModel, self).__init__()
        
        self.sequence_length = sequence_length
        
        # ResNet50 backbone (without final classifier)
        if model_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            num_features = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
            backbone = models.resnet50(pretrained=pretrained)
            num_features = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # LSTM for temporal aggregation
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Attention pooling for temporal aggregation (V2 - BETTER PERFORMANCE)
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Query for attention pooling (uses central timestep as query)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_size * 2))
        
        # Final classifier (V2 - less dropout for more learning)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # REDUCED from 0.6
            nn.Linear(hidden_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),  # REDUCED from 0.5
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # REDUCED from 0.4
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, C, H, W)
        batch_size, seq_len, C, H, W = x.size()
        
        # Reshape to process all images: (batch_size * seq_len, C, H, W)
        x_reshaped = x.view(batch_size * seq_len, C, H, W)
        
        # Extract features with ResNet for each image
        features = self.backbone(x_reshaped)  # (batch_size * seq_len, num_features, 1, 1)
        features = features.view(batch_size * seq_len, -1)  # (batch_size * seq_len, num_features)
        
        # Reshape for sequence: (batch_size, seq_len, num_features)
        features_seq = features.view(batch_size, seq_len, -1)
        
        # LSTM for temporal aggregation
        lstm_out, _ = self.lstm(features_seq)  # (batch_size, seq_len, hidden_size*2)
        
        # Multi-Head Attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection + Layer Normalization
        residual_out = self.layer_norm(lstm_out + attn_out)
        
        # V2: Attention Pooling instead of simple mean (BETTER PERFORMANCE)
        # Uses central timestep as query to weight all sequences
        query = self.pool_query.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_size*2)
        pooled_out, pool_weights = self.attention_pool(query, residual_out, residual_out)
        aggregated = pooled_out.squeeze(1)  # (batch_size, hidden_size*2)
        
        # Final classifier
        output = self.classifier(aggregated)
        
        return output

# -------------------------------
# EARLY STOPPING
# -------------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.005, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.best_model_state = None
        self.epochs_without_improvement = 0

    def __call__(self, val_loss, val_acc, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.best_model_state = model.state_dict().copy()
            self.epochs_without_improvement = 0
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.counter = 0
            self.epochs_without_improvement = 0
            self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1
            self.epochs_without_improvement += 1
        
        # Stop if no improvement for patience epochs
        if self.epochs_without_improvement >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_model_state)
            return True
        return False

# -------------------------------
# MIXUP FOR REGULARIZATION
# -------------------------------
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -------------------------------
# COLLATE FUNCTION FOR TEMPORAL SEQUENCES
# -------------------------------
def custom_collate_fn(batch):
    """Collate function for temporal sequences - OPTIMIZED (images already fixed size)"""
    sequences = []
    labels = []
    paths_list = []
    metadata_list = []
    
    for item in batch:
        images, label, paths, metadata = item
        sequences.append(images)  # images is a list of 3 images
        labels.append(label)
        paths_list.append(paths)
        metadata_list.append(metadata)
    
    # OPTIMIZATION: Images are already resized to (img_size, img_size) so they all have the same size!
    # Direct stack of sequences (much faster, no padding needed)
    stacked_sequences = []
    for seq_images in sequences:
        # Stack images of the sequence: (sequence_length, C, H, W)
        # All images already have the same size after Resize((img_size, img_size))
        seq_tensor = torch.stack(seq_images)
        stacked_sequences.append(seq_tensor)
    
    # Stack sequences: (batch_size, sequence_length, C, H, W)
    sequences = torch.stack(stacked_sequences)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return sequences, labels, paths_list, metadata_list

# ============================================================================
# Protection for multiprocessing on Windows
# ============================================================================
if __name__ == '__main__':
    # Add freeze_support for Windows
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Import matplotlib and seaborn here to avoid multiprocessing issues on Windows/Python 3.13
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create results folder if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results folder created/verified: {RESULTS_DIR}")

    # -------------------------------
    # 1. INTELLIGENT DEVICE SETUP
    # -------------------------------
    def setup_device():
        """Intelligent device setup with automatic GPU/CPU detection"""
        print("="*80)
        print("HARDWARE DETECTION")
        print("="*80)
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"GPU DETECTED: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"PyTorch CUDA: {torch.cuda.is_available()}")
            
            # Quick test to verify GPU works
            try:
                test_tensor = torch.randn(100, 100).to(device)
                result = torch.mm(test_tensor, test_tensor)
                print("GPU Test: SUCCESS")
                return device, "gpu", gpu_memory
            except Exception as e:
                print(f"GPU Test: FAILED - {str(e)}")
                print("Fallback to CPU...")
                device = torch.device("cpu")
                return device, "cpu", 0
        else:
            print("GPU not available")
            print("Using CPU...")
            device = torch.device("cpu")
            return device, "cpu", 0

    # -------------------------------
    # 2. ADAPTIVE PARAMETERS BASED ON DEVICE - MAXIMUM REGULARIZATION
    # -------------------------------
    def get_adaptive_params(device_type, gpu_memory=0):
        """Optimized parameters based on available device with MAXIMUM REGULARIZATION"""
        
        # Detect if we're on Windows (especially Python 3.13 has multiprocessing issues)
        is_windows = platform.system() == 'Windows'
        python_version = sys.version_info
        is_python313 = python_version.major == 3 and python_version.minor == 13
        
        if device_type == "gpu":
            # Optimized parameters for GPU - VERSION V2 (BALANCED SPEED/PERFORMANCE)
            if gpu_memory >= 8:  # GPU with at least 8GB
                batch_size = 32  # INCREASED for speed (sequences = 3 images, so 96 effective images)
                num_workers = 4  # INCREASED for parallel loading
                img_size = 384  # Fixed resolution required
            elif gpu_memory >= 4:  # GPU with 4-8GB
                batch_size = 24  # INCREASED for speed
                num_workers = 2  # INCREASED for parallel loading
                img_size = 384  # Fixed resolution required
            else:  # GPU with less than 4GB
                batch_size = 16  # INCREASED for speed
                num_workers = 2  # INCREASED for parallel loading
                img_size = 384  # Fixed resolution required
            
            pin_memory = True
            num_epochs = 30  # INCREASED to allow more learning
            
        else:  # CPU
            batch_size = 8  # Kept low for CPU
            num_workers = 2  # INCREASED - can work with 2 on Windows too
            img_size = 384  # Fixed resolution required
            pin_memory = False
            num_epochs = 25  # INCREASED
        
        # FORCE num_workers=0 on Windows (especially Python 3.13) to avoid multiprocessing issues
        if is_windows:
            if is_python313:
                print(f"Windows + Python 3.13 detected: num_workers forced to 0 (PyTorch compatibility)")
            else:
                print(f"Windows detected: num_workers forced to 0 (multiprocessing compatibility)")
            num_workers = 0
            pin_memory = False  # Also disable pin_memory on Windows with num_workers=0
        
        # Common parameters - BALANCED REGULARIZATION (V2 - BETTER PERFORMANCE)
        learning_rate = 1e-4      # INCREASED for faster learning
        weight_decay = 5e-4       # REDUCED to allow more learning
        dropout_rate = 0.4        # REDUCED to allow more learning
        
        print(f"Parameters: Batch={batch_size}, Workers={num_workers}, Size={img_size}")
        print(f"Epochs: {num_epochs}, Pin Memory: {pin_memory}")
        print(f"V2 - BALANCED Regularization: LR={learning_rate}, WD={weight_decay}, Dropout={dropout_rate}")
        
        return {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'img_size': img_size,
            'pin_memory': pin_memory,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'dropout_rate': dropout_rate
        }

    # -------------------------------
    # 3. SETUP DEVICE AND PARAMETERS
    # -------------------------------
    device, device_type, gpu_memory = setup_device()
    params = get_adaptive_params(device_type, gpu_memory)

    # Extract parameters
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']
    img_size = params['img_size']
    weight_decay = params['weight_decay']
    dropout_rate = params['dropout_rate']
    num_workers = params['num_workers']
    pin_memory = params['pin_memory']

    # -------------------------------
    # 4. LOAD LABELS FROM EXCEL WITH TEMPORAL INFORMATION
    # -------------------------------
    print("\nLoading labels from Excel...")
    try:
        df_labels = pd.read_excel(LABELS_EXCEL_PATH)
        print(f"Excel file loaded: {len(df_labels)} rows")
        print(f"Columns found: {list(df_labels.columns)}")
        
        # Verify that columns exist
        required_cols = ['image_name']
        if 'image_name' not in df_labels.columns:
            raise ValueError("Column 'image_name' must be present in Excel file")
        
        # Check if 'label' or 'labels' exists
        label_column = None
        if 'labels' in df_labels.columns:
            label_column = 'labels'
        elif 'label' in df_labels.columns:
            label_column = 'label'
        else:
            raise ValueError("One of columns 'label' or 'labels' must be present in Excel file")
        
        # Verify temporal columns (buildjob, geometry_name, channel, layer_index or layer)
        # Accept both 'layer_index' and 'layer' as column name
        required_temporal_cols = ['buildjob', 'geometry_name', 'channel']
        layer_col = None
        if 'layer_index' in df_labels.columns:
            layer_col = 'layer_index'
        elif 'layer' in df_labels.columns:
            layer_col = 'layer'
        
        missing_temporal = [col for col in required_temporal_cols if col not in df_labels.columns]
        if layer_col is None:
            missing_temporal.append('layer_index/layer')
        
        if missing_temporal:
            print(f"WARNING: Missing temporal columns: {missing_temporal}")
            print("Temporal sequences require: buildjob, geometry_name, channel, layer_index (or layer)")
            print("If missing, sequences will be created based only on image_name")
            has_temporal_info = False
        else:
            has_temporal_info = True
            temporal_cols = required_temporal_cols + [layer_col]
            print(f"Temporal columns found: {temporal_cols}")
        
        # Verify presence of 'detail' column
        if 'detail' not in df_labels.columns:
            raise ValueError("Column 'detail' must be present in Excel file")
        
        print(f"Using label column: '{label_column}'")
        print(f"Using detail column: 'detail'")
        
        # Create dictionaries for fast lookup
        labels_dict = {}
        details_dict = {}
        temporal_dict = {}  # For temporal information
        
        for idx, row in df_labels.iterrows():
            image_name = row['image_name']
            label = row[label_column]
            detail = row['detail']
            labels_dict[image_name] = label
            details_dict[image_name] = detail
            
            if has_temporal_info:
                temporal_dict[image_name] = {
                    'buildjob': row['buildjob'],
                    'geometry_name': row['geometry_name'],
                    'channel': row['channel'],
                    'layer_index': row[layer_col]  # Use layer_col (can be 'layer' or 'layer_index')
                }
        
        print(f"Labels loaded: {len(labels_dict)} images")
        print(f"Details loaded: {len(details_dict)} images")
        print(f"Example labels: {list(set(labels_dict.values()))}")
        
        # Show detail types found
        unique_details = df_labels['detail'].dropna().unique()
        print(f"Detail types found: {len(unique_details)}")
        print(f"Example details: {sorted(unique_details)[:10]}")  # Show first 10
        
    except Exception as e:
        print(f"ERROR loading Excel file: {str(e)}")
        raise  

    # -------------------------------
    # 5. TRANSFORMATIONS WITH CONSERVATIVE AUGMENTATIONS FOR LPBF
    # -------------------------------
    # Augmentation for training - CONSERVATIVE TO PRESERVE SMALL DEFECTS
    train_transform = transforms.Compose([
        # FORCED resize to fixed size (DOES NOT maintain aspect ratio - faster!)
        transforms.Resize((img_size, img_size)),  # Force fixed size (img_size x img_size)
        
        # CONSERVATIVE augmentations to preserve small defects
        transforms.RandomHorizontalFlip(p=0.5),  # OK - does not hide defects
        transforms.RandomVerticalFlip(p=0.5),    # OK - does not hide defects
        transforms.RandomRotation(degrees=20),   # Kept - OK for LPBF layers
        
        # Color augmentations - kept original for metallic layers
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        
        # VERY light affine transformation
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Drastically reduced
        
        # CONSERVATIVE AUGMENTATIONS - REMOVED DANGEROUS ONES
        # REMOVED: RandomPerspective - too much distortion
        # REMOVED: RandomErasing - hides defects!
        
        # Add safe augmentations for texture instead
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))  # Light blur for noise
        ], p=0.2),
        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Transformations for validation/test - FIXED SIZE
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Force fixed size (img_size x img_size)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # -------------------------------
    # 10. DATASET SETUP (SAME AS RESNET50_V4)
    # -------------------------------
    print("\nLoading dataset...")
    base_dataset = CustomDataset(IMAGES_DIR, labels_dict, details_dict=details_dict, transform=None)

    # Stratified split to balance classes
    # Uses train_test_split to maintain class proportions in each split
    from sklearn.model_selection import train_test_split
    
    total_samples = len(base_dataset)
    
    # Extract labels for all samples for stratification
    labels = []
    for idx in range(total_samples):
        _, label, _, _, _ = base_dataset[idx]
        labels.append(label)
    
    # First split: separate train (80%) from val+test (20%)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        list(range(total_samples)), labels, 
        test_size=0.2, 
        stratify=labels
        # random_state removed: different split each time, but proportions maintained thanks to stratify
    )
    
    # Second split: separate val (10%) from test (10%) from remaining 20%
    # Note: 10% of total = 50% of remaining 20%
    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices, temp_labels,
        test_size=0.5,  # 50% of 20% = 10% of total
        stratify=temp_labels
        # random_state removed: different split each time, but proportions maintained thanks to stratify
    )
    
    print(f"Stratified split completed:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")
    
    # Verify class distribution per split
    train_class_dist = {0: train_labels.count(0), 1: train_labels.count(1)}
    val_class_dist = {0: val_labels.count(0), 1: val_labels.count(1)}
    test_class_dist = {0: test_labels.count(0), 1: test_labels.count(1)}
    print(f"  Train class distribution: defective={train_class_dist[0]}, compliant={train_class_dist[1]}")
    print(f"  Val class distribution: defective={val_class_dist[0]}, compliant={val_class_dist[1]}")
    print(f"  Test class distribution: defective={test_class_dist[0]}, compliant={test_class_dist[1]}")

    # Create base dataset with appropriate transformations
    train_base = Subset(base_dataset, train_indices)
    val_base = Subset(base_dataset, val_indices)
    test_base = Subset(base_dataset, test_indices)

    # Apply transformations
    for i in range(len(train_base)):
        train_base.dataset.transform = train_transform
    for i in range(len(val_base)):
        val_base.dataset.transform = val_transform
    for i in range(len(test_base)):
        test_base.dataset.transform = val_transform

    # Create wrapper for temporal sequences
    print("\nCreating temporal sequences...")
    if has_temporal_info:
        full_dataset = TemporalSequenceWrapper(base_dataset, temporal_dict, labels_dict, details_dict=details_dict, sequence_length=3)
        
        # Split sequences based on central layer label
        # Uses train_test_split to maintain class proportions in each split
        from sklearn.model_selection import train_test_split
        
        total_sequences = len(full_dataset)
        
        # Extract labels for all sequences for stratification
        sequence_labels = [seq['label'] for seq in full_dataset.sequences]
        
        # First split: separate train (80%) from val+test (20%)
        train_sequences, temp_sequences, train_labels, temp_labels = train_test_split(
            list(range(total_sequences)), sequence_labels, 
            test_size=0.2, 
            stratify=sequence_labels
            # random_state removed: different split each time, but proportions maintained thanks to stratify
        )
        
        # Second split: separate val (10%) from test (10%) from remaining 20%
        # Note: 10% of total = 50% of remaining 20%
        val_sequences, test_sequences, val_labels, test_labels = train_test_split(
            temp_sequences, temp_labels,
            test_size=0.5,  # 50% of 20% = 10% of total
            stratify=temp_labels
            # random_state removed: different split each time, but proportions maintained thanks to stratify
        )
        
        print(f"Stratified sequence split completed:")
        print(f"  Train: {len(train_sequences)} sequences")
        print(f"  Val: {len(val_sequences)} sequences")
        print(f"  Test: {len(test_sequences)} sequences")
        
        # Verify class distribution per split
        train_class_dist = {0: train_labels.count(0), 1: train_labels.count(1)}
        val_class_dist = {0: val_labels.count(0), 1: val_labels.count(1)}
        test_class_dist = {0: test_labels.count(0), 1: test_labels.count(1)}
        print(f"  Train class distribution: defective={train_class_dist[0]}, compliant={train_class_dist[1]}")
        print(f"  Val class distribution: defective={val_class_dist[0]}, compliant={val_class_dist[1]}")
        print(f"  Test class distribution: defective={test_class_dist[0]}, compliant={test_class_dist[1]}")
        
        train_dataset = Subset(full_dataset, train_sequences)
        val_dataset = Subset(full_dataset, val_sequences)
        test_dataset = Subset(full_dataset, test_sequences)
    else:
        print("WARNING: No temporal information, using base dataset")
        train_dataset = train_base
        val_dataset = val_base
        test_dataset = test_base

    # DataLoader with parameters optimized for device
    # NOTE: num_workers is already set to 0 on Windows in get_adaptive_params function
    # to avoid multiprocessing issues (especially Python 3.13)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    # Val and Test always with num_workers=0 for stability
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=custom_collate_fn, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=custom_collate_fn, num_workers=0, pin_memory=False)
    print(f"DataLoader created: Train with {num_workers} workers, Val/Test with 0 workers")

    print(f"Dataset split: Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # -------------------------------
    # 11. TEMPORAL MODEL INITIALIZATION
    # -------------------------------
    model = TemporalResNetModel(
        num_classes=2, 
        pretrained=True, 
        model_name='resnet50',
        sequence_length=3,
        hidden_size=256,
        num_lstm_layers=2,
        dropout=dropout_rate  # Use dropout_rate from parameters (V2)
    ).to(device)
    
    # V2: Use Mixed Precision Training for speed (if GPU supports)
    if device_type == "gpu":
        try:
            from torch.cuda.amp import autocast, GradScaler
            use_amp = True
            scaler = GradScaler()
            print("Mixed Precision Training enabled (speeds up ~2x)")
        except:
            use_amp = False
            scaler = None
            print("Mixed Precision not available")
    else:
        use_amp = False
        scaler = None

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTemporal model initialized on: {device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # -------------------------------
    # 12. LOSS, OPTIMIZER AND SCHEDULER OPTIMIZED
    # -------------------------------
    # Calculate class weights based on DETAIL distribution in dataset
    # OPTIMIZED: uses information already available in sequences instead of loading images
    print("\nCalculating class weights based on DETAIL (optimized)...")
    detail_counts_by_class = {0: {}, 1: {}}  # 0=defective, 1=compliant
    class_counts = {0: 0, 1: 0}
    
    if has_temporal_info:
        for seq_info in full_dataset.sequences:
            label = seq_info['label']
            detail = seq_info.get('detail', 'unknown')
            class_counts[label] = class_counts.get(label, 0) + 1
            # Also count details for each class
            if detail not in detail_counts_by_class[label]:
                detail_counts_by_class[label][detail] = 0
            detail_counts_by_class[label][detail] += 1
    else:
        # Fallback: use base dataset
        for idx in range(len(base_dataset)):
            _, label, _, _, detail = base_dataset[idx]
            class_counts[label] = class_counts.get(label, 0) + 1
            # Also count details for each class
            if detail not in detail_counts_by_class[label]:
                detail_counts_by_class[label][detail] = 0
            detail_counts_by_class[label][detail] += 1
    
    total_samples = len(full_dataset)
    
    # Calculate weights considering detail distribution
    # Weights are inversely proportional to number of samples per class
    # calculated considering detail distribution
    if 0 in class_counts and 1 in class_counts and class_counts[0] > 0 and class_counts[1] > 0:
        # Weights are calculated based on class counts (which now include details)
        weight_0 = total_samples / (2.0 * class_counts[0])  # defective
        weight_1 = total_samples / (2.0 * class_counts[1])  # compliant
        
        class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)
        print(f"Class weights calculated based on DETAIL in dataset:")
        print(f"  Defective: {weight_0:.3f} (samples: {class_counts[0]}, unique details: {len(detail_counts_by_class[0])})")
        print(f"  Compliant: {weight_1:.3f} (samples: {class_counts[1]}, unique details: {len(detail_counts_by_class[1])})")
        print(f"  Top 5 defective details: {dict(sorted(detail_counts_by_class[0].items(), key=lambda x: x[1], reverse=True)[:5])}")
        print(f"  Top 5 compliant details: {dict(sorted(detail_counts_by_class[1].items(), key=lambda x: x[1], reverse=True)[:5])}")
    else:
        class_weights = None
        print("Class weights not calculated (using uniform weights)")
    
    # Loss with class weights to handle imbalance + REDUCED label smoothing (V2)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # REDUCED from 0.3

    # Optimizer with warmup
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Scheduler - uses CosineAnnealingWarmRestarts for stability on Windows
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2)

    # More aggressive early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.005)

    # -------------------------------
    # 13. OPTIMIZED TRAINING LOOP WITH MIXUP
    # -------------------------------
    print(f"\n{'='*80}")
    print(f"STARTING TEMPORAL TRAINING - {device_type.upper()} MODE")
    print(f"{'='*80}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Workers: {num_workers}")
    print(f"Pin Memory: {pin_memory}")
    print(f"Image Size: {img_size}")
    print(f"Device: {device}")
    print(f"Temporal sequences: [t-2, t-1, t]")
    print("="*80)
    print("IMAGE TRANSFORMATIONS:")
    print("- FORCED resize to fixed size (img_size x img_size)")
    print("- DOES NOT maintain aspect ratio (faster, no padding needed)")
    print("- CONSERVATIVE augmentations to preserve small defects")
    print("- WITHOUT MIXUP (version v2 - BETTER PERFORMANCE)")
    print("="*80)

    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                          leave=False, ncols=120, colour='green')
        
        for sequences, labels, paths_list, metadata_list in train_pbar:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass with Mixed Precision (V2 - faster)
            if use_amp:
                with autocast():
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Normal forward pass
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.1f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                        leave=False, ncols=120, colour='blue')
        
        with torch.no_grad():
            for sequences, labels, paths_list, metadata_list in val_pbar:
                sequences, labels = sequences.to(device), labels.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.1f}%'
                })

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update scheduler
        scheduler.step()  # For CosineAnnealingWarmRestarts
        
        # Calculate train/val gap to monitor overfitting
        gap = train_acc - val_acc
        
        # Check early stopping
        if early_stopping(val_loss, val_acc, model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
        
        # Early stop if gap too large
        if gap > 0.03:
            print(f"Gap too large ({gap:.4f}), considering early stop...")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_filename = os.path.join(RESULTS_DIR, f'best_model_temporal_short_{device_type}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'device_type': device_type,
                'params': params
            }, model_filename)

        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} "
              f"Gap: {gap:.4f} LR: {optimizer.param_groups[0]['lr']:.2e}")

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")

    # -------------------------------
    # 13.5. SAVE TRAINING SUMMARY
    # -------------------------------
    print("\nSaving training summary...")
    training_summary_filename = os.path.join(RESULTS_DIR, f"training_summary_temporal_short_{device_type}.txt")
    with open(training_summary_filename, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("TEMPORAL TRAINING SUMMARY - EPOCH BY EPOCH\n")
        f.write("="*80 + "\n\n")
        f.write(f"Device: {device_type.upper()}\n")
        f.write(f"Total epochs: {len(train_losses)}\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"Temporal sequences: [t-2, t-1, t]\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}\n")
        f.write("-"*80 + "\n")
        
        for epoch in range(len(train_losses)):
            f.write(f"{epoch+1:<8} {train_losses[epoch]:<12.6f} {train_accs[epoch]:<12.6f} "
                   f"{val_losses[epoch]:<12.6f} {val_accs[epoch]:<12.6f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILS PER EPOCH\n")
        f.write("="*80 + "\n\n")
        
        for epoch in range(len(train_losses)):
            f.write(f"Epoch {epoch+1}/{len(train_losses)}:\n")
            f.write(f"  Train Loss: {train_losses[epoch]:.6f}\n")
            f.write(f"  Train Accuracy: {train_accs[epoch]:.6f} ({train_accs[epoch]*100:.2f}%)\n")
            f.write(f"  Val Loss: {val_losses[epoch]:.6f}\n")
            f.write(f"  Val Accuracy: {val_accs[epoch]:.6f} ({val_accs[epoch]*100:.2f}%)\n")
            if epoch < len(train_losses) - 1:
                f.write("\n")
    
    # Also save as CSV for analysis
    training_summary_csv = os.path.join(RESULTS_DIR, f"training_summary_temporal_short_{device_type}.csv")
    df_summary = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    })
    df_summary.to_csv(training_summary_csv, index=False)
    print(f"Training summary saved in: {training_summary_filename}")
    print(f"Training summary CSV saved in: {training_summary_csv}")

    # -------------------------------
    # 14. LOAD BEST MODEL FOR TEST
    # -------------------------------
    model_filename = os.path.join(RESULTS_DIR, f'best_model_temporal_short_{device_type}.pth')
    checkpoint = torch.load(model_filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model loaded for final test: {model_filename}")

    # -------------------------------
    # 15. FINAL TEST WITH ADVANCED METRICS
    # -------------------------------
    print("\nStarting final test...")
    model.eval()
    test_correct = 0
    test_total = 0
    y_true = []
    y_pred = []
    y_probs = []
    y_details = []  # Add list for details
    misclassified_sequences = []

    test_pbar = tqdm(test_loader, desc="Final Test", ncols=120, colour='red')

    with torch.no_grad():
        for sequences, labels, paths_list, metadata_list in test_pbar:
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # Track misclassified sequences
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                prob = probs[i, 1].item()  # Positive class probability
                meta = metadata_list[i]
                detail = meta.get('detail', 'unknown')  # Get detail of central layer
                
                y_true.append(true_label)
                y_pred.append(pred_label)
                y_probs.append(prob)
                y_details.append(detail)  # Add detail to list
                
                if true_label != pred_label:
                    true_class = full_dataset.classes[true_label]
                    pred_class = full_dataset.classes[pred_label]
                    
                    misclassified_sequences.append({
                        'buildjob': meta.get('buildjob', 'N/A'),
                        'geometry': meta.get('geometry', 'N/A'),
                        'channel': meta.get('channel', 'N/A'),
                        'center_layer': meta.get('center_layer', 'N/A'),
                        'true_class': true_class,
                        'predicted_class': pred_class,
                        'detail': detail,
                        'confidence': prob if pred_label == 1 else 1-prob,
                        'image_names': meta.get('image_names', []),
                        'center_image_name': meta.get('center_image_name', 'N/A')
                    })
            
            test_pbar.set_postfix({
                'Acc': f'{100.*test_correct/test_total:.1f}%',
                'Errors': len(misclassified_sequences)
            })

    test_accuracy = test_correct / test_total
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # -------------------------------
    # 16. ADVANCED METRICS
    # -------------------------------
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    auc = roc_auc_score(y_true, y_probs)

    print(f"\n{'='*80}")
    print(f"ADVANCED METRICS - {device_type.upper()} MODE")
    print(f"{'='*80}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    # -------------------------------
    # 16.5. ADD FINAL METRICS TO TRAINING SUMMARY
    # -------------------------------
    print("\nAdding final metrics to training summary...")
    with open(training_summary_filename, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write("FINAL PERFORMANCE METRICS - TEST SET\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Accuracy:  {test_accuracy:.6f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"Precision:      {precision:.6f} ({precision*100:.2f}%)\n")
        f.write(f"Recall:         {recall:.6f} ({recall*100:.2f}%)\n")
        f.write(f"F1-Score:       {f1:.6f} ({f1*100:.2f}%)\n")
        f.write(f"AUC (ROC):      {auc:.6f}\n")
        f.write(f"\nTotal test sequences: {test_total}\n")
        f.write(f"Correctly classified sequences: {test_correct}\n")
        f.write(f"Misclassified sequences: {len(misclassified_sequences)}\n")
        f.write("="*80 + "\n")
    
    # Also save final metrics in a separate CSV file
    final_metrics_csv = os.path.join(RESULTS_DIR, f"final_metrics_temporal_short_{device_type}.csv")
    df_final = pd.DataFrame({
        'metric': ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc', 'test_total', 'test_correct', 'test_errors'],
        'value': [test_accuracy, precision, recall, f1, auc, test_total, test_correct, len(misclassified_sequences)]
    })
    df_final.to_csv(final_metrics_csv, index=False)
    print(f"Final metrics added to training summary: {training_summary_filename}")
    print(f"Final metrics also saved in: {final_metrics_csv}")

    # -------------------------------
    # 17. PRINT MISCLASSIFIED SEQUENCES
    # -------------------------------
    print(f"\n{'='*80}")
    print(f"MISCLASSIFIED SEQUENCES ({len(misclassified_sequences)} out of {test_total})")
    print(f"{'='*80}")

    if misclassified_sequences:
        # Sort by confidence (more confident = more serious errors)
        misclassified_sequences.sort(key=lambda x: x['confidence'], reverse=True)
        
        for i, item in enumerate(misclassified_sequences[:20], 1):  # Show only first 20
            print(f"{i:3d}. BuildJob: {item['buildjob']}, Geometry: {item['geometry']}, Channel: {item['channel']}, Layer: {item['center_layer']}")
            print(f"   True: {item['true_class']} | Predicted: {item['predicted_class']}")
            print(f"   Confidence: {item['confidence']:.3f}")
            print()
    else:
        print("No misclassified sequences!")

    # -------------------------------
    # 18. SAVE MISCLASSIFIED SEQUENCES
    # -------------------------------
    error_filename = os.path.join(RESULTS_DIR, f"misclassified_sequences_temporal_short_{device_type}.xlsx")
    print(f"Saving error list in: {error_filename}")
    
    if misclassified_sequences:
        # Create DataFrame with required columns
        error_data = []
        for i, item in enumerate(misclassified_sequences, 1):
            error_data.append({
                'center_image_name': item.get('center_image_name', 'N/A'),
                'buildjob': item.get('buildjob', 'N/A'),
                'geometry': item.get('geometry', 'N/A'),
                'channel': item.get('channel', 'N/A'),
                'center_layer': item.get('center_layer', 'N/A'),
                'true': item['true_class'],
                'predicted': item['predicted_class'],
                'detail': item.get('detail', 'N/A'),
                'confidence': item['confidence'],
                'image_names': ', '.join(item.get('image_names', []))
            })
        
        # Create DataFrame and save as Excel
        df_errors = pd.DataFrame(error_data)
        df_errors.to_excel(error_filename, index=False, engine='openpyxl')
        print(f"Excel file created with {len(df_errors)} rows")
    else:
        # Create empty DataFrame with correct columns
        df_errors = pd.DataFrame(columns=['center_image_name', 'buildjob', 'geometry', 'channel', 'center_layer', 'true', 'predicted', 'detail', 'confidence', 'image_names'])
        df_errors.to_excel(error_filename, index=False, engine='openpyxl')
        print("No misclassified sequences! Empty Excel file created.")

    # -------------------------------
    # 19. CONFUSION MATRIX & REPORT
    # -------------------------------
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    class_names = full_dataset.classes

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Temporal ResNet Short - {device_type.upper()} - Accuracy: {test_accuracy:.4f}")
    plt.tight_layout()
    cm_filename = os.path.join(RESULTS_DIR, f"confusion_matrix_temporal_short_{device_type}.png")
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # -------------------------------
    # 19.5. CONFUSION MATRIX AND METRICS FOR EACH DETAIL TYPE
    # -------------------------------
    print("\n" + "="*80)
    print("ANALYSIS BY DETAIL TYPE")
    print("="*80)
    
    # Group results by detail
    detail_groups = {}
    for i, detail in enumerate(y_details):
        if detail not in detail_groups:
            detail_groups[detail] = {'y_true': [], 'y_pred': [], 'y_probs': []}
        detail_groups[detail]['y_true'].append(y_true[i])
        detail_groups[detail]['y_pred'].append(y_pred[i])
        detail_groups[detail]['y_probs'].append(y_probs[i])
    
    # Sort details by number of samples (descending)
    sorted_details = sorted(detail_groups.items(), key=lambda x: len(x[1]['y_true']), reverse=True)
    
    print(f"\nFound {len(detail_groups)} detail types in test set")
    print(f"Sample distribution per detail:")
    for detail, data in sorted_details:
        print(f"  - {detail}: {len(data['y_true'])} samples")
    
    # Create DataFrame for metrics per detail
    detail_metrics_list = []
    
    # Generate confusion matrix and calculate metrics for each detail
    for detail, data in sorted_details:
        if len(data['y_true']) == 0:
            continue
        
        detail_y_true = data['y_true']
        detail_y_pred = data['y_pred']
        detail_y_probs = data['y_probs']
        
        # Calculate metrics for this detail
        detail_accuracy = sum(1 for i in range(len(detail_y_true)) if detail_y_true[i] == detail_y_pred[i]) / len(detail_y_true)
        
        # Precision, Recall, F1 for this detail
        try:
            detail_precision, detail_recall, detail_f1, _ = precision_recall_fscore_support(
                detail_y_true, detail_y_pred, average='weighted', zero_division=0
            )
        except:
            detail_precision = detail_recall = detail_f1 = 0.0
        
        # AUC if at least 2 classes are represented
        try:
            if len(set(detail_y_true)) > 1:
                detail_auc = roc_auc_score(detail_y_true, detail_y_probs)
            else:
                detail_auc = float('nan')
        except:
            detail_auc = float('nan')
        
        # Add to metrics
        detail_metrics_list.append({
            'detail': detail,
            'n_samples': len(detail_y_true),
            'accuracy': detail_accuracy,
            'precision': detail_precision,
            'recall': detail_recall,
            'f1_score': detail_f1,
            'auc': detail_auc
        })
        
        # Create confusion matrix for this detail
        detail_cm = confusion_matrix(detail_y_true, detail_y_pred)
        
        # Save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(detail_cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - Detail: {detail}\nAccuracy: {detail_accuracy:.4f} | Samples: {len(detail_y_true)}")
        plt.tight_layout()
        
        # Sanitize detail name for filename
        safe_detail_name = str(detail)
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            safe_detail_name = safe_detail_name.replace(char, '_')
        safe_detail_name = '_'.join(safe_detail_name.split())
        if len(safe_detail_name) > 100:
            safe_detail_name = safe_detail_name[:100]
        detail_cm_filename = os.path.join(RESULTS_DIR, f"confusion_matrix_detail_{safe_detail_name}_{device_type}.png")
        plt.savefig(detail_cm_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nDetail: {detail}")
        print(f"  Samples: {len(detail_y_true)}")
        print(f"  Accuracy: {detail_accuracy:.4f}")
        print(f"  Precision: {detail_precision:.4f}")
        print(f"  Recall: {detail_recall:.4f}")
        print(f"  F1-Score: {detail_f1:.4f}")
        auc_str = f"{detail_auc:.4f}" if not np.isnan(detail_auc) else "N/A"
        print(f"  AUC: {auc_str}")
        print(f"  Confusion Matrix saved: {detail_cm_filename}")
    
    # Create DataFrame with all metrics per detail
    df_detail_metrics = pd.DataFrame(detail_metrics_list)
    detail_metrics_csv = os.path.join(RESULTS_DIR, f"detail_metrics_{device_type}.csv")
    df_detail_metrics.to_csv(detail_metrics_csv, index=False)
    print(f"\nDetail metrics saved in: {detail_metrics_csv}")
    
    # Add detail metrics to training summary
    print("\nAdding detail metrics to training summary...")
    with open(training_summary_filename, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write("METRICS BY DETAIL TYPE - TEST SET\n")
        f.write("="*80 + "\n\n")
        
        for detail, data in sorted_details:
            if len(data['y_true']) == 0:
                continue
            
            detail_y_true = data['y_true']
            detail_y_pred = data['y_pred']
            detail_y_probs = data['y_probs']
            
            detail_accuracy = sum(1 for i in range(len(detail_y_true)) if detail_y_true[i] == detail_y_pred[i]) / len(detail_y_true)
            
            try:
                detail_precision, detail_recall, detail_f1, _ = precision_recall_fscore_support(
                    detail_y_true, detail_y_pred, average='weighted', zero_division=0
                )
            except:
                detail_precision = detail_recall = detail_f1 = 0.0
            
            try:
                if len(set(detail_y_true)) > 1:
                    detail_auc = roc_auc_score(detail_y_true, detail_y_probs)
                else:
                    detail_auc = float('nan')
            except:
                detail_auc = float('nan')
            
            f.write(f"Detail: {detail}\n")
            f.write(f"  Samples: {len(detail_y_true)}\n")
            f.write(f"  Accuracy:  {detail_accuracy:.6f} ({detail_accuracy*100:.2f}%)\n")
            f.write(f"  Precision: {detail_precision:.6f} ({detail_precision*100:.2f}%)\n")
            f.write(f"  Recall:    {detail_recall:.6f} ({detail_recall*100:.2f}%)\n")
            f.write(f"  F1-Score:  {detail_f1:.6f} ({detail_f1*100:.2f}%)\n")
            if not np.isnan(detail_auc):
                f.write(f"  AUC (ROC): {detail_auc:.6f}\n")
            else:
                f.write(f"  AUC (ROC): N/A\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"Detail metrics added to training summary: {training_summary_filename}")

    # -------------------------------
    # 20. TRAINING PROGRESS PLOT
    # -------------------------------
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Progress - Loss Temporal ResNet Short ({device_type.upper()})')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc', color='blue')
    plt.plot(val_accs, label='Val Acc', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training Progress - Accuracy Temporal ResNet Short ({device_type.upper()})')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    # Learning rate schedule
    epochs = range(1, len(train_losses) + 1)
    lrs = [scheduler.get_last_lr()[0] for _ in epochs]
    plt.plot(epochs, lrs, label='Learning Rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    progress_filename = os.path.join(RESULTS_DIR, f"training_progress_temporal_short_{device_type}.png")
    plt.savefig(progress_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print("\nAll completed!")
    print(f"All outputs saved in: {RESULTS_DIR}")
    print(f"\nSaved files:")
    print(f"  - Model: {model_filename}")
    print(f"  - Training summary (txt): {training_summary_filename}")
    print(f"  - Training summary (csv): {training_summary_csv}")
    print(f"  - Final metrics (csv): {final_metrics_csv}")
    print(f"  - Detail metrics (csv): {detail_metrics_csv}")
    print(f"  - Plots: {cm_filename}, {progress_filename}")
    print(f"  - Confusion matrix per detail: {len(detail_groups)} files")
    print(f"  - Errors (Excel): {error_filename}")

