import pandas as pd
from langdetect import detect
import json

def filter_english_news(input_files=['Data/Datasets/Health_News/Health_News.csv', 'Data/Datasets/Covid_and_Vaccine_News/Covid_and_Vaccine_News.csv', 
                                     'Data/Datasets/Covid_News/Covid_News.csv', 'Data/Datasets/Science_&_Technology_News/Science_&_Technology_News.csv', 
                                     'Data/Datasets/World_Politics_News/World_Politics_News.csv']):
    try:
        all_english_news = []
        total_records = 0
        
        for input_file in input_files:
            # Read the CSV file
            df = pd.read_csv(input_file)
            total_records += len(df)
            
            if df.empty:
                print(f"Warning: The file '{input_file}' is empty")
                continue
            
            # Ensure required columns exist
            required_columns = ['title', 'content']
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: Missing required columns in '{input_file}'. File must contain {required_columns}")
                continue
            
            # Filter English news more efficiently using vectorized operations
            def detect_language_safe(text):
                if pd.isna(text):
                    return None
                try:
                    return detect(str(text))
                except:
                    return None
            
            # Apply language detection to both columns
            df['title_lang'] = df['title'].apply(detect_language_safe)
            df['content_lang'] = df['content'].apply(detect_language_safe)
            
            # Filter for English content and select 200 random articles
            english_mask = (df['title_lang'] == 'en') | (df['content_lang'] == 'en')
            english_df = df[english_mask]
            
            # If we have more than 200 articles, randomly sample 200
            if len(english_df) > 200:
                english_df = english_df.sample(n=200, random_state=42)  # random_state for reproducibility
            
            english_news = english_df.drop(['title_lang', 'content_lang'], axis=1).to_dict('records')
            all_english_news.extend(english_news)
            
            print(f"Processed '{input_file}': Selected {len(english_news)} English articles")
        
        if not all_english_news:
            print("Error: No English articles found in any of the input files")
            return []
        
        # Save combined results to CSV
        output_file = 'Data/Processed_Data/combined_english_news.csv'
        pd.DataFrame(all_english_news).to_csv(output_file, index=False)
        
        print(f"\nSummary:")
        print(f"Total input articles processed: {total_records}")
        print(f"Total English articles selected: {len(all_english_news)}")
        print(f"Results saved to {output_file}")
        
        return all_english_news
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return []
