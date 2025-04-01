import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the Dataset class
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

def train_model(model, train_loader, val_loader, device, num_epochs=5):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        # Create progress bar for training
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]', 
                            leave=True, position=0)
        
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze()
            loss = F.mse_loss(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_progress.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        model.eval()
        total_val_loss = 0
        
        # Create progress bar for validation
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]', 
                          leave=True, position=0)
        
        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze()
                loss = F.mse_loss(logits, labels)
                
                total_val_loss += loss.item()
                val_progress.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}\n')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def main():
    # Load and preprocess data
    df = pd.read_csv('labelled_news.csv')
    texts = df['title'].tolist()
    labels = df[['positive_score', 'neutral_score', 'negative_score']].values

    # Split the data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3,  # Three outputs for positive, neutral, and negative scores
        problem_type="regression"
    )

    # Create datasets and dataloaders
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train the model
    train_model(model, train_loader, val_loader, device)

    print("Training completed! The best model has been saved as 'best_model.pth'")

if __name__ == "__main__":
    main()
