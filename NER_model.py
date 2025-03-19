import pandas as pd
import spacy
import argparse

nlp = spacy.load("en_core_web_sm")
file_path = "Datasets/Science_&_Technology_News/Science_&_Technology_News.csv"

keywords = {
    "companies": ["Pfizer", "Moderna", "Johnson & Johnson", "AstraZeneca", "Merck", "Roche", "GlaxoSmithKline", "UnitedHealth", "Cigna"],
    "drugs": ["Lipitor", "Zoloft", "Advil", "Tylenol", "Ibuprofen", "Aspirin"],
    "vaccines": ["COVID-19 vaccine", "Influenza vaccine", "HPV vaccine", "Pfizer-BioNTech", "Moderna vaccine"]
}

def is_healthcare(text):
    if not isinstance(text, str):
        return False
    #keywords = ["healthcare", "medical", "medicine"]
    return any(keyword.lower() in text.lower() for keyword in keywords)

def entities(text):
    doc = nlp(text)
    entity = [(ent.text, ent.label_) for ent in doc.ents]
    return entity

if __name__ == "__main__":

    df = pd.read_csv(file_path)

    df['is_healthcare'] = df["content"].apply(is_healthcare)

    healthcare_df = df[df['is_healthcare']].copy()

    healthcare_df['entities'] = healthcare_df["content"].apply(entities)

    healthcare_df.to_csv("healthcare_articles_with_entities.csv", index=False)