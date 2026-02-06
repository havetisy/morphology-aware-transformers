import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Model configuration"""
    # Model
    MODEL_NAME = 'deepset/gbert-base'
    MAX_LENGTH = 256
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    EPOCHS = 5
    
    # Data split
    TEST_SIZE = 0.0001  # Minimal test set
    VAL_SIZE = 0.1      # 10% validation
    RANDOM_STATE = 1
    
    # KldB digit level (change this to modify classification granularity)
    # (0,1) = 1st digit (~10 classes)
    # (0,2) = 1st-2nd digits (~40 classes)
    # (0,3) = 1st-3rd digits (~140 classes)
    # (0,5) = All 5 digits (~37,000 classes)
    KLDB_START_DIGIT = 0
    KLDB_END_DIGIT = 1
    
    # Files
    TRAIN_DATA = 'dataset.csv'
    TEST_DATA = 'TEST.csv'


# ============================================================================
# Utility Functions
# ============================================================================

def kldb_devision(_data, _start_digit, _end_digit):
    """
    Extract specific digits from KldB code for hierarchical classification
    
    Args:
        _data: DataFrame with 'kldb' column
        _start_digit: Start position (0-indexed)
        _end_digit: End position (exclusive)
    
    Returns:
        Modified DataFrame with extracted KldB digits
    
    Examples:
        (0, 1) → 1st digit (occupational area)
        (0, 2) → 1st-2nd digits (main group)
        (0, 3) → 1st-3rd digits (sub-group)
    """
    _data['col'] = _data['kldb'].astype(str)
    _data['kldb_new'] = _data['col'].str[_start_digit:_end_digit]
    return _data['kldb_new']


def setup_device():
    """Initialize GPU if available"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('No GPU available, using CPU')
        device = torch.device('cpu')
    return device


# ============================================================================
# Metrics
# ============================================================================

def f1_score_func(preds, labels):
    """Calculate weighted F1 score"""
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def recall_score_func(preds, labels):
    """Calculate weighted recall"""
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return recall_score(labels_flat, preds_flat, average='weighted')


def precision_score_func(preds, labels):
    """Calculate weighted precision"""
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return precision_score(labels_flat, preds_flat, average='weighted')


# ============================================================================
# Training Functions
# ============================================================================

def set_seed(seed_val=17):
    """Set seed for reproducibility"""
    import random
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def evaluate(model, dataloader_val, device):
    """
    Evaluate model on validation set
    
    Returns:
        loss_val_avg: Average validation loss
        predictions: Model predictions
        true_vals: True labels
    """
    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2],
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total / len(dataloader_val)
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    
    return loss_val_avg, predictions, true_vals


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    """Main training and evaluation pipeline"""
    
    print("="*70)
    print("German Occupation Coding via BERT")
    print("="*70)
    
    # Set seed for reproducibility
    set_seed()
    
    # Setup device
    device = setup_device()
    
    # ========================================================================
    # Load and prepare data
    # ========================================================================
    print(f"\nLoading training data from {Config.TRAIN_DATA}...")
    df = pd.read_csv(Config.TRAIN_DATA, encoding="utf-8")
    print(f"Loaded {len(df)} records")
    
    # Extract KldB digit level
    print(f"\nExtracting KldB digits {Config.KLDB_START_DIGIT}:{Config.KLDB_END_DIGIT}...")
    df.kldb = kldb_devision(df, Config.KLDB_START_DIGIT, Config.KLDB_END_DIGIT)
    
    # Check sequence lengths
    seq_len = [len(i.split()) for i in df.occupation]
    print(f"Max sequence length in data: {max(seq_len)} words")
    
    # Show class distribution
    print(f"\nClass distribution:")
    print(df['kldb'].value_counts())
    
    # ========================================================================
    # Encode labels
    # ========================================================================
    print("\nEncoding labels...")
    possible_labels = df.kldb.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    
    df['label'] = df.kldb.replace(label_dict)
    
    print(f"Number of classes: {len(label_dict)}")
    print(f"Label mapping (sample): {dict(list(label_dict.items())[:5])}")
    
    # ========================================================================
    # Train/Val/Test split
    # ========================================================================
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df.index.values,
        df.label.values,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=Config.VAL_SIZE,
        random_state=Config.RANDOM_STATE
    )
    
    df['data_type'] = ['not_set'] * df.shape[0]
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'
    df.loc[X_test, 'data_type'] = 'test'
    
    print(df.groupby(['data_type']).count())
    
    # ========================================================================
    # Tokenization
    # ========================================================================
    print(f"\nLoading tokenizer: {Config.MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(
        Config.MODEL_NAME,
        do_lower_case=True
    )
    
    print("Tokenizing data...")
    
    # Tokenize train set
    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type == 'train'].occupation.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=Config.MAX_LENGTH,
        return_tensors='pt'
    )
    
    # Tokenize validation set
    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type == 'val'].occupation.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=Config.MAX_LENGTH,
        return_tensors='pt'
    )
    
    # Tokenize test set
    encoded_data_test = tokenizer.batch_encode_plus(
        df[df.data_type == 'test'].occupation.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=Config.MAX_LENGTH,
        return_tensors='pt'
    )
    
    # Create tensors
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type == 'train'].label.values)
    
    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type == 'val'].label.values)
    
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(df[df.data_type == 'test'].label.values)
    
    # Create datasets
    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    
    print("✓ Tokenization complete")
    
    # ========================================================================
    # Initialize model
    # ========================================================================
    print(f"\nInitializing model: {Config.MODEL_NAME}...")
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(label_dict),
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)
    
    print(f"Number of labels: {len(label_dict)}")
    
    # ========================================================================
    # Prepare dataloaders
    # ========================================================================
    print("\nPreparing dataloaders...")
    
    dataloader_train = DataLoader(
        dataset_train,
        sampler=RandomSampler(dataset_train),
        batch_size=Config.BATCH_SIZE
    )
    
    dataloader_validation = DataLoader(
        dataset_val,
        sampler=SequentialSampler(dataset_val),
        batch_size=Config.BATCH_SIZE
    )
    
    dataloader_test = DataLoader(
        dataset_test,
        sampler=SequentialSampler(dataset_test),
        batch_size=Config.BATCH_SIZE
    )
    
    # ========================================================================
    # Setup optimizer and scheduler
    # ========================================================================
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader_train) * Config.EPOCHS
    )
    
    # ========================================================================
    # Training loop
    # ========================================================================
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    for epoch in tqdm(range(1, Config.EPOCHS + 1)):
        model.train()
        loss_train_total = 0
        
        progress_bar = tqdm(
            dataloader_train,
            desc=f'Epoch {epoch:1d}',
            leave=False,
            disable=False
        )
        
        for batch in progress_bar:
            model.zero_grad()
            
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2],
            }
            
            outputs = model(**inputs)
            
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({
                'training_loss': '{:.3f}'.format(loss.item() / len(batch))
            })
        
        # Save model checkpoint
        torch.save(model.state_dict(), f'model_epoch_{epoch:02d}.pt')
        
        print(f'\nEpoch {epoch}')
        
        loss_train_avg = loss_train_total / len(dataloader_train)
        print(f'Training loss: {loss_train_avg}')
        
        # Validation
        val_loss, predictions, true_vals = evaluate(model, dataloader_validation, device)
        val_f1 = f1_score_func(predictions, true_vals)
        val_recall = recall_score_func(predictions, true_vals)
        val_precision = precision_score_func(predictions, true_vals)
        
        print(f'Validation loss: {val_loss}')
        print(f'F1 Score (Weighted): {val_f1}')
        print(f'Recall (Weighted): {val_recall}')
        print(f'Precision (Weighted): {val_precision}')
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    
    # ========================================================================
    # Save final model
    # ========================================================================
    print("\nSaving final model...")
    torch.save(model.state_dict(), 'final_model.pt')
    print("✓ Model saved to final_model.pt")


def predict_test_set(model_path='final_model.pt'):
    """
    Load saved model and make predictions on test set
    
    Args:
        model_path: Path to saved model weights
    """
    print("\n" + "="*70)
    print("Testing on External Test Set")
    print("="*70)
    
    # Setup
    device = setup_device()
    
    # Load test data
    print(f"\nLoading test data from {Config.TEST_DATA}...")
    test = pd.read_csv(Config.TEST_DATA)
    
    # Extract KldB digits
    test.kldb = kldb_devision(test, Config.KLDB_START_DIGIT, Config.KLDB_END_DIGIT)
    
    # Tokenize
    print("Tokenizing test data...")
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME, do_lower_case=True)
    
    X_data = test.index.values
    test.loc[X_data, 'data_type'] = 'test'
    
    encoded_data = tokenizer.batch_encode_plus(
        test[test.data_type == 'test'].cberuf.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=Config.MAX_LENGTH,
        return_tensors='pt'
    )
    
    input_ids_data = encoded_data['input_ids']
    attention_masks_data = encoded_data['attention_mask']
    
    dataset_data = TensorDataset(input_ids_data, attention_masks_data)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    
    # First, we need to know the number of labels from training
    # This would typically be saved along with the model
    # For now, assuming it's loaded from the training data label_dict
    
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=1020,  # Adjust this to match your training
        output_attentions=False,
        output_hidden_states=False
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create dataloader
    dataloader_input = DataLoader(
        dataset_data,
        sampler=SequentialSampler(dataset_data),
        batch_size=16
    )
    
    # Predict
    print("Making predictions...")
    predictions = []
    
    for batch in tqdm(dataloader_input):
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1]
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    
    predictions = np.concatenate(predictions, axis=0)
    preds_flat = np.argmax(predictions, axis=1)
    
    # Save results
    print("\nSaving predictions...")
    test['predictions'] = preds_flat
    test[['occupation', 'predictions', 'kldb']].to_csv('result.csv', index=False)
    
    print("✓ Predictions saved to result.csv")
    print("\n" + "="*70)


if __name__ == "__main__":
    # Train the model
    main()
    
    # Uncomment to run predictions on test set after training
    # predict_test_set('final_model.pt')
