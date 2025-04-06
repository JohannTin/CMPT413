import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

class HealthcareSentimentDataset(Dataset):
    def __init__(self, texts, labels, entities, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.entities = entities
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        entity_list = self.entities[idx]

        # Create entity mask
        entity_mask = torch.zeros(self.max_length, dtype=torch.long)
        if entity_list:
            for entity, entity_type in entity_list:
                entity_tokens = self.tokenizer.tokenize(entity)
                entity_ids = self.tokenizer.convert_tokens_to_ids(entity_tokens)
                # Find positions of entity in the text
                start_pos = text.find(entity)
                if start_pos != -1:
                    # Get the token positions
                    encoding = self.tokenizer.encode_plus(
                        text[:start_pos],
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True,
                        return_tensors=None
                    )
                    entity_start = len(encoding['input_ids'])
                    entity_end = min(entity_start + len(entity_ids), self.max_length)
                    entity_mask[entity_start:entity_end] = 1

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'entity_mask': entity_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

class HealthcareBERT(nn.Module):
    def __init__(self, num_labels=3):
        super(HealthcareBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.entity_embedding = nn.Embedding(2, self.bert.config.hidden_size)  # Binary mask for entities
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, num_labels)  # Double size for entity features
        
    def forward(self, input_ids, attention_mask, entity_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        
        # Get entity embeddings
        entity_embeddings = self.entity_embedding(entity_mask)
        
        # Combine BERT output with entity embeddings
        combined_output = torch.cat([
            sequence_output.mean(dim=1),  # Average pooling of BERT output
            entity_embeddings.mean(dim=1)  # Average pooling of entity embeddings
        ], dim=1)
        
        combined_output = self.dropout(combined_output)
        logits = self.classifier(combined_output)
        return logits

def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=2e-5, patience=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity_mask = batch['entity_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, entity_mask=entity_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                entity_mask = batch['entity_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, entity_mask=entity_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'hbert_best_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

def main():
    # Load and preprocess data
    df = pd.read_csv('Data/Processed_Data/healthcare_articles_with_sentiment(manual).csv')
    
    # Combine title and content for better context
    df['text'] = df['title'] + ' ' + df['content']
    
    # Convert sentiment scores to labels (0: positive, 1: neutral, 2: negative)
    df['label'] = df.apply(lambda x: np.argmax([x['sentiment_positive'], x['sentiment_neutral'], x['sentiment_negative']]), axis=1)
    
    # Convert entities from string to list of tuples
    df['entities'] = df['entities'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    
    # Split data
    train_texts, val_texts, train_labels, val_labels, train_entities, val_entities = train_test_split(
        df['text'].values,
        df['label'].values,
        df['entities'].values,
        test_size=0.2,
        random_state=42
    )
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = HealthcareBERT()
    
    # Create datasets
    train_dataset = HealthcareSentimentDataset(train_texts, train_labels, train_entities, tokenizer)
    val_dataset = HealthcareSentimentDataset(val_texts, val_labels, val_entities, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_model(model, train_loader, val_loader, device)

if __name__ == '__main__':
    main()
