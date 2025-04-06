import pandas as pd
import spacy

nlp = spacy.load("best-model") #set to "en_core_web_lg" when running custom ner in jupyter notebook cell

input_files='Data/Processed_Data/combined_english_news.csv'
output_file = "Data/Processed_Data/healthcare_articles_with_custom_entities.csv" #set to healthcare_articles_with_entities.csv in jupyter notebook cell

def set_nlp(model):
    global nlp
    nlp = spacy.load(model)

def set_output(output):
    global output_file
    output_file = output

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