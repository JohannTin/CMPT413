"""
reference for training custom NER 

https://github.com/amrrs/custom-ner-with-spacy3/blob/main/Custom_NER_with_Spacy3.ipynb
"""

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import json

nlp = spacy.blank("en") # load a new spacy model
db = DocBin() # create a DocBin object

def train_NER():
    with open('healthcare_annotations.json', encoding='utf-8') as f:
        TRAIN_DATA = json.load(f)
    
    #remove null values
    filtered_annotations = [item for item in TRAIN_DATA['annotations'] if item is not None]

    for text, annot in tqdm(filtered_annotations): 
        doc = nlp.make_doc(text) 
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        doc.ents = ents 
        db.add(doc)

    db.to_disk("./training_data.spacy") # save the docbin object