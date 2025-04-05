import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import numpy as np

# Load the dataset
df = pd.read_csv('Data/Datasets/Recent_News.csv')

# Load the HBERT model and tokenizer
model_path = 'hbert_best_model.pth'
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load(model_path))
model.eval()

# Function to get sentiment scores
def get_sentiment_scores(text):
    if pd.isna(text) or not isinstance(text, str):
        return pd.Series({'positive_score': None, 'neutral_score': None, 'negative_score': None})
    
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
        # Apply softmax to get probabilities
        probs = F.softmax(outputs.logits, dim=1).squeeze()
    
    # Return probabilities for each class (Positive=0, Neutral=1, Negative=2)
    return pd.Series({
        'positive_score': probs[0].item(),
        'neutral_score': probs[1].item(),
        'negative_score': probs[2].item()
    })

# Apply sentiment analysis to the content column
print("Processing articles...")
sentiment_scores = df['content'].apply(get_sentiment_scores)
df = pd.concat([df, sentiment_scores], axis=1)

# Save the results
output_path = 'Data/Processed_Data/Recent_News_HBERT.csv'
df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
