# CMPT413
Healthcare Sentiment Analysis for Market Trends

# Files
* Data\Datasets - contains the original datasets
* Data\Processed_Data - consists of all processed datas
* models - the trained models of LSTM+FinBERT, LSTM+HBERT, and LSTM

# Environment and Installations:
python3.10 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Large Models to import
Google Drive https://drive.google.com/drive/u/1/folders/16n_AqPpo2cGeJ03ymtdso-oFPtS51C10

# spaCy trained models (optional if you do not want to retrain spacy model in jupyter notebook)
import the folders model-best and model-last

# HBERTv2
import the train model hbert_best_model.pth before running the line %run HBERT_Recent_News.py in the notebook