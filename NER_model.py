import pandas as pd
import spacy
import os
import glob

nlp = spacy.load("en_core_web_sm")

#ref https://stackoverflow.com/questions/77693774/copy-multiple-csv-files-from-multiple-subfolders-in-the-same-parent-directory-t
input_dir = os.path.join("Datasets", "*", "*.csv")
#file_path = "Datasets/Science_&_Technology_News/Science_&_Technology_News.csv"

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

    csv_files = glob.glob(input_dir)
    processed_df = []

    for file in csv_files:
        df = pd.read_csv(file)

        df['is_healthcare'] = df["content"].apply(is_healthcare)
        healthcare_df = df[df['is_healthcare']].copy()
        healthcare_df['entities'] = healthcare_df["content"].apply(entities)

        processed_df.append(healthcare_df)

    output_df = pd.concat(processed_df)
    output_df.to_csv("healthcare_articles_with_entities.csv", index=False)