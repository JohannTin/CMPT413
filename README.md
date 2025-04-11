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

# SFU Vault link to Large Models to import
https://vault.sfu.ca/index.php/s/ch6pqTL9Ni2mpo1

# spaCy trained models (optional if you do not want to retrain spacy model in jupyter notebook)
import the folders model-best and model-last

# HBERTv2
import the train model hbert_best_model.pth before running the line %run HBERT_Recent_News.py in the notebook
If downloading the model does not work, please run the HBERTv2.py file in a virtual environment