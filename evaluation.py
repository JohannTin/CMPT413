import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_sentiment(true_sentiment, lstm_pred, hbert_pred, lstm_only_pred):
    """
    Evaluate sentiment prediction metrics for all models
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
    
    # Calculate metrics for LSTM-only
    lstm_only_accuracy = accuracy_score(true_sentiment, lstm_only_pred)
    lstm_only_precision = precision_score(true_sentiment, lstm_only_pred, average='weighted')
    lstm_only_recall = recall_score(true_sentiment, lstm_only_pred, average='weighted')
    lstm_only_f1 = f1_score(true_sentiment, lstm_only_pred, average='weighted')
    lstm_only_cm = confusion_matrix(true_sentiment, lstm_only_pred)
    
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
    
    print("\nLSTM-only Sentiment Metrics:")
    print(f"Accuracy: {lstm_only_accuracy:.4f}")
    print(f"Precision: {lstm_only_precision:.4f}")
    print(f"Recall: {lstm_only_recall:.4f}")
    print(f"F1 Score: {lstm_only_f1:.4f}")
    print("\nConfusion Matrix:")
    print(lstm_only_cm)
    
    # Plot confusion matrices
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(lstm_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('LSTM-FinBERT Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.subplot(1, 3, 2)
    sns.heatmap(hbert_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('H-BERT Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.subplot(1, 3, 3)
    sns.heatmap(lstm_only_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('LSTM-only Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig('Data/Figures/confusion_matrices.png')
    plt.close()

def evaluate_stock_predictions(true_prices, lstm_pred_prices, hbert_pred_prices, lstm_only_pred_prices):
    """
    Evaluate stock price prediction metrics for all models
    """
    # Calculate MAE and R2 for LSTM-FinBERT
    lstm_mae = mean_absolute_error(true_prices, lstm_pred_prices)
    lstm_r2 = r2_score(true_prices, lstm_pred_prices)
    
    # Calculate MAE and R2 for H-BERT
    hbert_mae = mean_absolute_error(true_prices, hbert_pred_prices)
    hbert_r2 = r2_score(true_prices, hbert_pred_prices)
    
    # Calculate MAE and R2 for LSTM-only
    lstm_only_mae = mean_absolute_error(true_prices, lstm_only_pred_prices)
    lstm_only_r2 = r2_score(true_prices, lstm_only_pred_prices)
    
    # Print results
    print("\nLSTM-FinBERT Stock Prediction Metrics:")
    print(f"Mean Absolute Error: {lstm_mae:.4f}")
    print(f"R-squared Score: {lstm_r2:.4f}")
    
    print("\nH-BERT Stock Prediction Metrics:")
    print(f"Mean Absolute Error: {hbert_mae:.4f}")
    print(f"R-squared Score: {hbert_r2:.4f}")
    
    print("\nLSTM-only Stock Prediction Metrics:")
    print(f"Mean Absolute Error: {lstm_only_mae:.4f}")
    print(f"R-squared Score: {lstm_only_r2:.4f}")
    
    # Set Seaborn style
    sns.set_style("whitegrid")
    
    # Define consistent colors
    colors = {
        'LSTM-FinBERT': '#2ecc71',  # Green
        'H-BERT': '#3498db',        # Blue
        'LSTM-only': '#e74c3c'      # Red
    }
    
    # Plot predictions vs actual
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(true_prices, label='Actual', color='black', linewidth=2)
    plt.plot(lstm_pred_prices, label='LSTM-FinBERT', color=colors['LSTM-FinBERT'], alpha=0.8)
    plt.title('LSTM-FinBERT Predictions vs Actual', fontsize=12, pad=15)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(true_prices, label='Actual', color='black', linewidth=2)
    plt.plot(hbert_pred_prices, label='H-BERT', color=colors['H-BERT'], alpha=0.8)
    plt.title('H-BERT Predictions vs Actual', fontsize=12, pad=15)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(true_prices, label='Actual', color='black', linewidth=2)
    plt.plot(lstm_only_pred_prices, label='LSTM-only', color=colors['LSTM-only'], alpha=0.8)
    plt.title('LSTM-only Predictions vs Actual', fontsize=12, pad=15)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('Data/Figures/stock_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_performance(true_prices, lstm_pred_prices, hbert_pred_prices, lstm_only_pred_prices):
    """
    Create a combined plot showing predictions from all models together
    """
    # Set Seaborn style
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(12, 6))
    
    # Define consistent colors
    colors = {
        'LSTM-FinBERT': '#2ecc71',  # Green
        'H-BERT': '#3498db',        # Blue
        'LSTM-only': '#e74c3c'      # Red
    }
    
    plt.plot(true_prices, label='Actual', color='black', linewidth=2)
    plt.plot(lstm_pred_prices, label='LSTM-FinBERT', color=colors['LSTM-FinBERT'], alpha=0.8)
    plt.plot(hbert_pred_prices, label='H-BERT', color=colors['H-BERT'], alpha=0.8)
    plt.plot(lstm_only_pred_prices, label='LSTM-only', color=colors['LSTM-only'], alpha=0.8)
    
    plt.title('Model Performance Comparison', fontsize=12, pad=15)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Data/Figures/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_regression_metrics(true_prices, lstm_pred_prices, hbert_pred_prices, lstm_only_pred_prices):
    """
    Plot MSE and R-squared scores for all models
    """
    # Calculate metrics for all models
    models = {
        'LSTM-FinBERT': lstm_pred_prices,
        'H-BERT': hbert_pred_prices,
        'LSTM-only': lstm_only_pred_prices
    }
    
    mse_scores = []
    r2_scores = []
    model_names = []
    
    for name, pred in models.items():
        mse = mean_squared_error(true_prices, pred)
        r2 = r2_score(true_prices, pred)
        mse_scores.append(mse)
        r2_scores.append(r2)
        model_names.append(name)
    
    # Set Seaborn style
    sns.set_style("whitegrid")
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define consistent colors
    colors = {
        'LSTM-FinBERT': '#2ecc71',  # Green
        'H-BERT': '#3498db',        # Blue
        'LSTM-only': '#e74c3c'      # Red
    }
    
    # Plot MSE
    bars1 = ax1.bar(model_names, mse_scores, color=[colors[name] for name in model_names])
    ax1.set_title('Mean Squared Error (MSE)', fontsize=12, pad=15)
    ax1.set_ylabel('MSE Score', fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    # Plot R-squared
    bars2 = ax2.bar(model_names, r2_scores, color=[colors[name] for name in model_names])
    ax2.set_title('R-squared Score', fontsize=12, pad=15)
    ax2.set_ylabel('R-squared Score', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('Data/Figures/regression_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load the data
    lstm_finbert_predictions = pd.read_csv('lstm_finbert_predictions_XLV.csv')
    hbert_predictions = pd.read_csv('lstm_hbert_predictions_XLV.csv')
    lstm_only_predictions = pd.read_csv('lstm_predictions_xlv.csv')
    
    # Convert sentiment probabilities to sentiment labels (0: Negative, 1: Neutral, 2: Positive)
    def get_sentiment_label(row):
        probs = [row['Negative_Sentiment'], row['Neutral_Sentiment'], row['Positive_Sentiment']]
        return np.argmax(probs)
    
    # Calculate sentiment labels
    lstm_sentiment = lstm_finbert_predictions.apply(get_sentiment_label, axis=1)
    hbert_sentiment = hbert_predictions.apply(get_sentiment_label, axis=1)
    lstm_only_sentiment = lstm_only_predictions['Signal'].values  # Using Signal column for sentiment
    
    # Use LSTM-FinBERT's actual values as ground truth since both models use the same data
    true_sentiment = lstm_finbert_predictions.apply(get_sentiment_label, axis=1)
    true_prices = lstm_finbert_predictions['Actual'].values
    
    # Get price predictions
    lstm_pred_prices = lstm_finbert_predictions['Predicted'].values
    hbert_pred_prices = hbert_predictions['Predicted'].values
    lstm_only_pred_prices = lstm_only_predictions['Predicted'].values
    
    # Evaluate sentiment predictions
    evaluate_sentiment(true_sentiment, lstm_sentiment, hbert_sentiment, lstm_only_sentiment)
    
    # Evaluate stock price predictions
    evaluate_stock_predictions(true_prices, lstm_pred_prices, hbert_pred_prices, lstm_only_pred_prices)
    
    # Create combined performance plot
    plot_combined_performance(true_prices, lstm_pred_prices, hbert_pred_prices, lstm_only_pred_prices)
    
    # Plot regression metrics
    plot_regression_metrics(true_prices, lstm_pred_prices, hbert_pred_prices, lstm_only_pred_prices)

if __name__ == "__main__":
    main() 