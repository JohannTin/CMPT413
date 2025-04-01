import pandas as pd
from langdetect import detect
import glob
import os

def count_english_articles(csv_file):
    try:
        df = pd.read_csv(csv_file)
        english_count = 0
        total = len(df)
        
        print(f"\nProcessing {os.path.basename(csv_file)}:")
        print(f"Total articles: {total}")
        
        for idx, row in df.iterrows():
            try:
                # Combine title and description for better language detection
                text = str(row['title']) + ' ' + str(row['description'])
                if detect(text) == 'en':
                    english_count += 1
                
                # Show progress every 100 articles
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{total} articles...", end='\r')
                    
            except Exception as e:
                continue
                
        print(f"\nEnglish articles found: {english_count}")
        return english_count
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")
        return 0

def main():
    dataset_path = "Datasets/*/*.csv"
    total_english = 0
    
    print("Starting language detection analysis...")
    
    for csv_file in glob.glob(dataset_path):
        english_count = count_english_articles(csv_file)
        total_english += english_count
        
    print(f"\nTotal English articles across all files: {total_english}")

if __name__ == "__main__":
    main() 