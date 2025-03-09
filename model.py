import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import time

df = pd.read_csv('data/Processed/words.csv')
train_df = pd.read_csv('data/Processed/train_words.csv')
val_df = pd.read_csv('data/Processed/val_words.csv')

train_df = train_df[['Image', 'label']]
val_df = val_df[['Image', 'label']]

labels = [str(word) for word in df['label'].to_numpy()]

# Unique characters
unique_chars = set(char for word in labels for char in word)
n_classes = len(unique_chars)

# Show
print(f"Total number of unique characters : {n_classes}")
print(f"Unique Characters : \n{unique_chars}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset paths
IMAGE_DIR = "data/Processed/augmented_words"

# Character Mapping
characters = sorted(list(unique_chars))
char_to_idx = {c: i+1 for i, c in enumerate(characters)}  # Start index from 1 (0 reserved for blank)
char_to_idx['<blank>'] = 0  # Blank character
idx_to_char = {i+1: c for i, c in enumerate(characters)}
idx_to_char[0] = '<blank>'  # Blank character
NUM_CLASSES = len(characters) + 1  # Extra for blank


import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

class OCRDataset(Dataset):
    def __init__(self, dataframe, img_dir, char_to_idx, idx_to_char, transform=None, height=128):

        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.height = height
        
        # Character to index mapping
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and label
        img_name = self.dataframe.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        label = self.dataframe.iloc[idx, 1]
        
        # Read image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Store original dimensions
        original_height, original_width = image.shape
        
        # Resize to fixed height while maintaining aspect ratio
        aspect_ratio = original_width / original_height
        new_width = int(aspect_ratio * self.height)
        image = cv2.resize(image, (new_width, self.height))
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        
        # Convert label to indices
        label_indices = [self.char_to_idx[c] for c in label]
        label_tensor = torch.tensor(label_indices, dtype=torch.long)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return {
            'image': image_tensor,
            'label': label_tensor,
            'width': new_width,
            'text': label
        }
    
    def get_vocab_size(self):
        return len(self.char_to_idx)
    

import torch
import torch.nn.functional as F

def collate_fn(batch):

    # Extract items
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    widths = [item['width'] for item in batch]
    texts = [item['text'] for item in batch]
    
    # Find max width in the batch
    max_width = max(widths)
    batch_size = len(images)
    
    # Pad images to max width in the batch (not globally)
    # This is more memory efficient than padding all images to a global max width
    padded_images = []
    for image in images:
        _, h, w = image.size()
        padding_size = max_width - w
        if padding_size > 0:
            # Pad the width dimension to match max_width
            padded_image = F.pad(image, (0, padding_size, 0, 0), "constant", 0)
        else:
            padded_image = image
        padded_images.append(padded_image)
    
    # Stack images to form a batch
    images_tensor = torch.stack(padded_images)
    
    # Find max label length in the batch
    max_label_len = max([len(label) for label in labels])
    
    # Pad labels to max length
    padded_labels = []
    label_lengths = []
    for label in labels:
        label_lengths.append(len(label))
        # Pad with zeros (index for blank)
        if len(label) < max_label_len:
            padded_label = torch.cat([
                label, 
                torch.zeros(max_label_len - len(label), dtype=torch.long)
            ])
        else:
            padded_label = label
        padded_labels.append(padded_label)
    
    # Convert to tensors
    padded_labels = torch.stack(padded_labels)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    widths = torch.tensor(widths, dtype=torch.long)
    
    return {
        'images': images_tensor,
        'labels': padded_labels,
        'label_lengths': label_lengths,
        'widths': widths,
        'texts': texts
    }
    
    
# DataLoaders
train_dataset = OCRDataset(train_df, IMAGE_DIR,char_to_idx, idx_to_char, height=128)
val_dataset = OCRDataset(val_df, IMAGE_DIR, char_to_idx, idx_to_char, height=128)

train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    collate_fn=collate_fn,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False, 
    collate_fn=collate_fn,
    pin_memory=True
)

class CRNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2):
        super(CRNN, self).__init__()
        
        # Reduced CNN Feature Extractor
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 128 -> 64
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64 -> 32
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)  # 32 -> 16
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Simplified LSTM
        self.lstm = nn.LSTM(64 * 16, 128, bidirectional=True, batch_first=True)
        
        # Output Layer
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # CNN layers with fewer filters
        x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool3(nn.functional.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        # Reshape for LSTM
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c*h)  # (batch, width, features)
        
        # Single LSTM layer
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x  # (batch, seq_len, num_classes)
 
 
 # Decode predictions (greedy decoding)
def decode_predictions(predictions):
    # Simple greedy decoding
    decoded_preds = []
    for pred in predictions:
        # Remove repeated characters
        collapsed = []
        prev_char = -1
        for p in pred:
            if p != prev_char:  # CTC collapse
                collapsed.append(p)
            prev_char = p
        
        # Remove blank tokens (0)
        result = [idx_to_char[c] for c in collapsed if c > 0]
        decoded_preds.append(''.join(result))
    
    return decoded_preds

# Calculate Character Error Rate (CER)
def calculate_cer(pred_texts, true_texts):
    total_distance = 0
    total_length = 0
    
    for pred, true in zip(pred_texts, true_texts):
        # Calculate Levenshtein distance
        distance = levenshtein_distance(pred, true)
        total_distance += distance
        total_length += len(true)
    
    return total_distance / total_length if total_length > 0 else 1.0

# Levenshtein distance implementation
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, path='models/'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        
        
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    all_pred_texts = []
    all_true_texts = []
    correct_count = 0
    total_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Get batch data
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            label_lengths = batch['label_lengths'].to(device)
            texts = batch['texts']  # Original text labels
            
            # Forward pass
            outputs = model(images).log_softmax(2)
            outputs_for_loss = outputs.permute(1, 0, 2)  # (T, N, C)
            
            # Calculate loss
            input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long, device=device)
            
            # Calculate loss
            loss = criterion(outputs_for_loss, labels, input_lengths, label_lengths)
            val_loss += loss.item()
            
            # Get predictions
            predictions = outputs.argmax(2).detach().cpu().numpy()
            pred_texts = decode_predictions(predictions)
            
            # Calculate accuracy
            for pred, true in zip(pred_texts, texts):
                total_count += 1
                if pred == true:
                    correct_count += 1
            
            all_pred_texts.extend(pred_texts)
            all_true_texts.extend(texts)
    
    # Calculate metrics
    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct_count / total_count if total_count > 0 else 0
    cer = calculate_cer(all_pred_texts, all_true_texts)
    
    return avg_val_loss, cer, accuracy, all_pred_texts, all_true_texts


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience,min_delta,checkpoint_dir='models/'):
    os.makedirs(checkpoint_dir, exist_ok=True)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_cer': [],
        'val_accuracy': [],
        'learning_rates': []
    }

    # Early stopping variables
    best_val_loss = float('inf')
    counter = 0
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

        for batch_idx, batch in enumerate(train_pbar):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            label_lengths = batch['label_lengths'].to(device)

            outputs = model(images).log_softmax(2)
            outputs_for_loss = outputs.permute(1, 0, 2)

            input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long, device=device)

            loss = criterion(outputs_for_loss, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # Use the validate function here
        avg_val_loss, cer, accuracy, all_pred_texts, all_true_texts = validate(model, val_loader, criterion)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_cer'].append(cer)
        history['val_accuracy'].append(accuracy)
        history['learning_rates'].append(current_lr)

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}, CER: {cer:.4f}, LR: {current_lr:.6f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            counter = 0
            # Save best model (based on validation loss)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'cer': cer,
                'accuracy': accuracy,
                'val_loss': avg_val_loss
            }, best_model_path)
            
            print(f"Saved best model with val_loss: {best_val_loss:.4f}")
        else:
            counter += 1
            
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load the best model weights (based on validation loss)
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    final_cer = best_checkpoint['cer']
    final_accuracy = best_checkpoint['accuracy']
    best_epoch = best_checkpoint['epoch']

    print(f"\nTraining completed. Restored best model from epoch {best_epoch}")
    print(f"Final CER: {final_cer:.4f}, Final Accuracy: {final_accuracy:.4f}")

    return history


# Initialize Model
model = CRNN(NUM_CLASSES).to(device)
criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
optimizer = optim.Adam(
    model.parameters(), 
    lr=0.001, 
    weight_decay=1e-4
)
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5,
    patience=3
)


# Set hyperparameters
NUM_EPOCHS = 100
PATIENCE = 5
CHECKPOINT_DIR = 'models/'

# Train the model
history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=NUM_EPOCHS,
    patience=PATIENCE,
    checkpoint_dir=CHECKPOINT_DIR
)


    
trained_model = CRNN(NUM_CLASSES).to(device)
trained_model.load_state_dict(torch.load('models/best_accuracy_model.pth')['model_state_dict'])


def plot_training_metrics(history):
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Plot Training and Validation Loss
    axs[0, 0].plot(history['train_loss'], label='Train Loss')
    axs[0, 0].plot(history['val_loss'], label='Validation Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot Character Error Rate (CER)
    axs[0, 1].plot(history['val_cer'], label='CER', color='red')
    axs[0, 1].set_title('Character Error Rate')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('CER')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot Accuracy
    axs[1, 0].plot(history['val_accuracy'], label='Accuracy', color='green')
    axs[1, 0].set_title('Validation Accuracy')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot Learning Rate
    axs[1, 1].plot(history['learning_rates'], label='Learning Rate', color='purple')
    axs[1, 1].set_title('Learning Rate')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Learning Rate')
    axs[1, 1].set_yscale('log')  # Log scale for better visualization
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
    plt.savefig('images/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def predict_and_visualize_random(model, data_loader, idx_to_char, device, num_samples=10, save_path=None, seed=None):

    import random
    import numpy as np
    import torch.nn.functional as F
    
    # Set random seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    model.eval()
    
    # Get all samples from the loader
    all_data = []
    for batch in data_loader:
        batch_size = len(batch['texts'])
        for i in range(batch_size):
            item = {
                'image': batch['images'][i],
                'text': batch['texts'][i]
            }
            all_data.append(item)
            
            # Break if we already have enough samples
            if len(all_data) >= num_samples * 10:  # Get 10x more than needed to have a good random selection
                break
        
        if len(all_data) >= num_samples * 10:
            break
    
    # Select random samples
    num_samples = min(num_samples, len(all_data))
    random_indices = random.sample(range(len(all_data)), num_samples)
    selected_samples = [all_data[i] for i in random_indices]
    
    # Process each image individually
    predictions = []
    texts = [sample['text'] for sample in selected_samples]
    images_for_display = [sample['image'].clone() for sample in selected_samples]  # Clone to avoid modifying originals
    
    # Get predictions one by one to avoid padding issues
    correct_count = 0
    total_count = 0
    
    with torch.no_grad():
        for i, sample in enumerate(selected_samples):
            # Process single image
            image = sample['image'].unsqueeze(0).to(device)  # Add batch dimension
            output = model(image).log_softmax(2)
            prediction = output.argmax(2).cpu().numpy()[0]  # Get first (only) item from batch
            
            # Decode prediction
            decoded_pred = decode_single_prediction(prediction, idx_to_char)
            predictions.append(decoded_pred)
            
            # Check if prediction is correct
            if decoded_pred == sample['text']:
                correct_count += 1
            total_count += 1
    
    accuracy = correct_count / total_count
    print(f"Sample accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    
    # Display samples
    rows = (num_samples + 4) // 5  # Ensure 5 samples per row
    fig, axs = plt.subplots(rows, 5, figsize=(16, 2 * rows))
    
    # Handle the case when rows=1
    if rows == 1:
        axs = np.array([axs]).reshape(1, -1)
    
    axs = axs.flatten()
    
    for i in range(num_samples):
        # Get the image and convert to numpy for display
        image = images_for_display[i].squeeze(0).cpu().numpy()
        
        # Display the image
        axs[i].imshow(image, cmap="gray")
        
        # Color code the title based on whether prediction is correct
        title_color = 'green' if predictions[i] == texts[i] else 'red'
        
        axs[i].set_title(f"Pred: {predictions[i]}\nTrue: {texts[i]}", 
                         fontsize=9, color=title_color)
        axs[i].axis("off")
    
    # Hide unused subplots
    for i in range(len(axs)):
        if i >= num_samples:
            axs[i].axis("off")
    
    plt.suptitle("OCR Model Predictions (Random Samples)", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.show()

def decode_single_prediction(prediction, idx_to_char):
    """
    Decode a single prediction using CTC decoding
    
    Args:
        prediction: Raw prediction from the model
        idx_to_char: Mapping from indices to characters
        
    Returns:
        String containing the decoded text
    """
    # Remove repeated characters
    collapsed = []
    prev_char = -1
    for p in prediction:
        if p != prev_char:  # CTC collapse
            collapsed.append(p)
        prev_char = p
    
    # Remove blank tokens (0)
    result = [idx_to_char[c] for c in collapsed if c > 0]
    return ''.join(result)

# Example usage:
predict_and_visualize_random(
    model=model,
    data_loader=val_loader,
    idx_to_char=idx_to_char,
    device=device,
    num_samples=20,
    save_path='random_predictions.png',
    seed=2  # For reproducible results
)

# After training the model and having the history dictionary
plot_training_metrics(history)
