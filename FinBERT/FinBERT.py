import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

class FinBERTAnalyzer:
    def __init__(self):
        # Initialize FinBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.labels = ['negative', 'neutral', 'positive']

    def get_sentiment_score(self, text):
        if not isinstance(text, str) or not text.strip():
            return {
                'negative': 0,
                'neutral': 0,
                'positive': 0
            }
            
        # Prepare the text for the model
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get model outputs
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert predictions to scores
        scores = predictions[0].tolist()
        sentiment_scores = {
            'negative': scores[0],
            'neutral': scores[1],
            'positive': scores[2]
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
        
# Process all files in all subdirectories
# def process_datasets():
#     # Initialize the analyzer
#     analyzer = FinBERTAnalyzer()
    
#     # Path to the Datasets folder
#     datasets_path = "Datasets"
#     results = []

#     # Process all files in all subdirectories
#     for root, dirs, files in os.walk(datasets_path):
#         for file in files:
#             if file.endswith('.json'):
#                 category = os.path.basename(root)
#                 file_path = os.path.join(root, file)
#                 print(f"\nProcessing {file_path}")
                
#                 try:
#                     # Read JSON file
#                     with open(file_path, 'r', encoding='utf-8') as f:
#                         articles = json.load(f)
                    
#                     # Process each article
#                     for article in tqdm(articles, desc=f"Analyzing articles in {category}"):
#                         # Extract text from the article
#                         text = article.get('text', '') or article.get('content', '')
#                         article_id = article.get('id', '') or article.get('article_id', '')
                        
#                         result = analyzer.analyze_text(text, article_id)
#                         if result:
#                             #Skip articles where all sentiment scores are 0
#                             scores = result['scores']
#                             if scores['negative'] == 0 and scores['neutral'] == 0 and scores['positive'] == 0:
#                                 continue
                                
#                             result['category'] = category
#                             result['title'] = article.get('title', '')
#                             result['date'] = article.get('date', '') or article.get('published_date', '')
#                             results.append(result)
                
#                 except Exception as e:
#                     print(f"Error processing file {file_path}: {str(e)}")
#                     continue

def process_datasets():
    # Initialize the analyzer
    analyzer = FinBERTAnalyzer()
    
    # Path to the Datasets folder
    datasets_path = "Datasets"
    results = []

    # Specify the categories we want to process
    target_categories = ['Health_News', 'World_Politics_News']
    
    # Process only specified categories
    for category in target_categories:
        category_path = os.path.join(datasets_path, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category path {category_path} not found")
            continue
            
        # Process all JSON files in the category directory
        for file in os.listdir(category_path):
            if file.endswith('.json'):
                file_path = os.path.join(category_path, file)
                print(f"\nProcessing {file_path}")
                
                try:
                    # Read JSON file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        articles = json.load(f)
                    
                    # Process each article
                    for article in tqdm(articles, desc=f"Analyzing articles in {category}"):
                        # Extract text from the article
                        text = article.get('text', '') or article.get('content', '')
                        article_id = article.get('id', '') or article.get('article_id', '')
                        
                        result = analyzer.analyze_text(text, article_id)
                        if result:
                            # Skip articles where all sentiment scores are 0
                            scores = result['scores']
                            if scores['negative'] == 0 and scores['neutral'] == 0 and scores['positive'] == 0:
                                continue
                                
                            result['category'] = category
                            result['title'] = article.get('title', '')
                            result['date'] = article.get('date', '') or article.get('published_date', '')
                            results.append(result)
                
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
                    continue

    # Convert results to DataFrame
    df = pd.DataFrame([
        {
            'article_id': r['article_id'],
            'category': r['category'],
            'title': r['title'],
            'date': r['date'],
            'negative_score': r['scores']['negative'],
            'neutral_score': r['scores']['neutral'],
            'positive_score': r['scores']['positive']
        }
        for r in results
    ])
    
    # Save results
    df.to_csv('FinBERT_results.csv', index=False)
    
    # Also save as JSON for detailed view
    with open('FinBERT_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nAnalysis complete. Processed {len(results)} articles.")
    return df

if __name__ == "__main__":
    results_df = process_datasets()
    print("\nSample of results:")
    print(results_df.head())
