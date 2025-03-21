import pandas as pd
import spacy
import os
import glob

nlp = spacy.load("en_core_web_sm")

#ref https://stackoverflow.com/questions/77693774/copy-multiple-csv-files-from-multiple-subfolders-in-the-same-parent-directory-t
input_dir = os.path.join("Datasets", "*", "*.csv")
#file_path = "Datasets/Science_&_Technology_News/Science_&_Technology_News.csv"

def entities(text):
    doc = nlp(text)
    entity = [(ent.text, ent.label_) for ent in doc.ents]
    return entity

if __name__ == "__main__":

    csv_files = glob.glob(input_dir)
    processed_df = []

    for file in csv_files:
        df = pd.read_csv(file)
        df = df.dropna(subset=['content'])

        df['entities'] = df["content"].apply(entities)

        processed_df.append(df)

    output_df = pd.concat(processed_df)
    output_df.to_csv("healthcare_articles_with_entities.csv", index=False)