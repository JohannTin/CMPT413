import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_sentiment(true_sentiment, lstm_pred, hbert_pred):
    """
    Evaluate sentiment prediction metrics for both models
    """
    # Calculate metrics for LSTM-FinBERT
    lstm_accuracy = accuracy_score(true_sentiment, lstm_pred)
    lstm_precision = precision_score(true_sentiment, lstm_pred, average='weighted')
    lstm_recall = recall_score(true_sentiment, lstm_pred, average='weighted')
    lstm_f1 = f1_score(true_sentiment, lstm_pred, average='weighted')
    lstm_cm = confusion_matrix(true_sentiment, lstm_pred)
    
    # Calculate metrics for H-BERT
    hbert_accuracy = accuracy_score(true_sentiment, hbert_pred)
    hbert_precision = precision_score(true_sentiment, hbert_pred, average='weighted')
    hbert_recall = recall_score(true_sentiment, hbert_pred, average='weighted')
    hbert_f1 = f1_score(true_sentiment, hbert_pred, average='weighted')
    hbert_cm = confusion_matrix(true_sentiment, hbert_pred)
    
    # Print results
    print("LSTM-FinBERT Sentiment Metrics:")
    print(f"Accuracy: {lstm_accuracy:.4f}")
    print(f"Precision: {lstm_precision:.4f}")
    print(f"Recall: {lstm_recall:.4f}")
    print(f"F1 Score: {lstm_f1:.4f}")
    print("\nConfusion Matrix:")
    print(lstm_cm)
    
    print("\nH-BERT Sentiment Metrics:")
    print(f"Accuracy: {hbert_accuracy:.4f}")
    print(f"Precision: {hbert_precision:.4f}")
    print(f"Recall: {hbert_recall:.4f}")
    print(f"F1 Score: {hbert_f1:.4f}")
    print("\nConfusion Matrix:")
    print(hbert_cm)
    
    # Plot confusion matrices
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(lstm_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('LSTM-FinBERT Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(hbert_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('H-BERT Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig('Data/Figures/confusion_matrices.png')
    plt.close()

def evaluate_stock_predictions(true_prices, lstm_pred_prices, hbert_pred_prices):
    """
    Evaluate stock price prediction metrics for both models
    """
    # Calculate MAE and R2 for LSTM-FinBERT
    lstm_mae = mean_absolute_error(true_prices, lstm_pred_prices)
    lstm_r2 = r2_score(true_prices, lstm_pred_prices)
    
    # Calculate MAE and R2 for H-BERT
    hbert_mae = mean_absolute_error(true_prices, hbert_pred_prices)
    hbert_r2 = r2_score(true_prices, hbert_pred_prices)
    
    # Print results
    print("\nLSTM-FinBERT Stock Prediction Metrics:")
    print(f"Mean Absolute Error: {lstm_mae:.4f}")
    print(f"R-squared Score: {lstm_r2:.4f}")
    
    print("\nH-BERT Stock Prediction Metrics:")
    print(f"Mean Absolute Error: {hbert_mae:.4f}")
    print(f"R-squared Score: {hbert_r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(true_prices, label='Actual')
    plt.plot(lstm_pred_prices, label='LSTM-FinBERT')
    plt.title('LSTM-FinBERT Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(true_prices, label='Actual')
    plt.plot(hbert_pred_prices, label='H-BERT')
    plt.title('H-BERT Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('Data/Figures/stock_predictions.png')
    plt.close()

def main():
    # Load the data
    lstm_finbert_predictions = pd.read_csv('lstm_finbert_predictions_XLV.csv')
    hbert_predictions = pd.read_csv('lstm_hbert_predictions_XLV.csv')
    
    # Convert sentiment probabilities to sentiment labels (0: Negative, 1: Neutral, 2: Positive)
    def get_sentiment_label(row):
        probs = [row['Negative_Sentiment'], row['Neutral_Sentiment'], row['Positive_Sentiment']]
        return np.argmax(probs)
    
    # Calculate sentiment labels
    lstm_sentiment = lstm_finbert_predictions.apply(get_sentiment_label, axis=1)
    hbert_sentiment = hbert_predictions.apply(get_sentiment_label, axis=1)
    
    # Use LSTM-FinBERT's actual values as ground truth since both models use the same data
    true_sentiment = lstm_finbert_predictions.apply(get_sentiment_label, axis=1)
    true_prices = lstm_finbert_predictions['Actual'].values
    
    # Get price predictions
    lstm_pred_prices = lstm_finbert_predictions['Predicted'].values
    hbert_pred_prices = hbert_predictions['Predicted'].values
    
    # Evaluate sentiment predictions
    evaluate_sentiment(true_sentiment, lstm_sentiment, hbert_sentiment)
    
    # Evaluate stock price predictions
    evaluate_stock_predictions(true_prices, lstm_pred_prices, hbert_pred_prices)

if __name__ == "__main__":
    main() 