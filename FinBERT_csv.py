import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

class FinBERTAnalyzer:
    def __init__(self):
        print("\nInitializing FinBERT model and tokenizer...")
        # Initialize FinBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.labels = ['negative', 'neutral', 'positive']  # Correct order of labels
        print("Model and tokenizer initialized successfully")

    def get_sentiment_score(self, text):
        if not isinstance(text, str) or not text.strip():
            return {
                'positive': 0,
                'neutral': 0,
                'negative': 0
            }
            
        # Prepare the text for the model
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get model outputs
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert predictions to scores
        scores = predictions[0].tolist()
        sentiment_scores = {
            'positive': scores[2],
            'neutral': scores[1],
            'negative': scores[0]
        }
        
        return sentiment_scores

    def analyze_text(self, text, article_id=None):
        try:
            # Get sentiment scores
            scores = self.get_sentiment_score(text)
            
            return {
                'article_id': article_id,
                'scores': scores
            }
        except Exception as e:
            print(f"Error processing article {article_id}: {str(e)}")
            return None

def process_csv():
    # Initialize the analyzer
    print("\nInitializing FinBERT analyzer...")
    analyzer = FinBERTAnalyzer()
    
    # Path to the CSV file
    csv_path = "Data/Processed_Data/combined_english_news.csv"
    results = []

    try:
        # Read CSV file
        print(f"\nReading CSV file from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Successfully read CSV file with {len(df)} rows")
        
        # Process each article
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing articles"):
            # Combine title, description, and full_description
            title = str(row.get('title', '')).strip()
            description = str(row.get('description', '')).strip()
            full_description = str(row.get('full_description', '')).strip()
            
            # Combine all text fields with proper spacing
            text_parts = []
            if title:
                text_parts.append(title)
            if description:
                text_parts.append(description)
            if full_description and full_description != description:  # Avoid duplicating content
                text_parts.append(full_description)
            
            text = ' '.join(text_parts)
            
            if len(text.strip()) < 10:  # Skip very short texts
                continue
                
            article_id = str(index)  # Use index as article_id if none provided
            
            result = analyzer.analyze_text(text, article_id)
            if result:
                # Skip articles where all sentiment scores are 0
                scores = result['scores']
                if scores['negative'] == 0 and scores['neutral'] == 0 and scores['positive'] == 0:
                    continue
                    
                result['category'] = row.get('category', '')
                result['title'] = title
                result['date'] = row.get('date', '')
                results.append(result)
    
    except Exception as e:
        print(f"Error processing CSV file {csv_path}: {str(e)}")
        return None

    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {
            'article_id': r['article_id'],
            'category': r['category'],
            'title': r['title'],
            'date': r['date'],
            'positive_score': r['scores']['positive'],
            'neutral_score': r['scores']['neutral'],
            'negative_score': r['scores']['negative']
        }
        for r in results
    ])
    
    # Save results
    results_df.to_csv("Data/Processed_Data/combined_english_news_sentiment(FinBERT).csv", index=False)
    
    # Also save as JSON for detailed view
    with open('FinBERT_csv_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nAnalysis complete. Processed {len(results)} articles.")
    return results_df

if __name__ == "__main__":
    results_df = process_csv()
    if results_df is not None:
        print("\nSample of results:")
        print(results_df.head()) 