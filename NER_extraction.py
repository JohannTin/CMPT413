import pandas as pd
import spacy

nlp = spacy.load("en_core_web_md") #set to "best-model" when running custom ner in jupyter notebook cell

input_files='Data/Processed_Data/combined_english_news.csv'
output_file = "healthcare_articles_with_entities.csv" #set to healthcare_articles_with_custom_entities.csv in jupyter notebook cell

def entities(text):
    doc = nlp(text)
    entity = [(ent.text, ent.label_) for ent in doc.ents]
    return entity

def extract():
    df = pd.read_csv(input_files)
    #drop rows where content in null
    df = df.dropna(subset=['content'])

    df['entities'] = df["content"].apply(entities)
    df.to_csv(output_file, index=False)