import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Layer, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import re
from datetime import datetime
import torch

# Configuration
CONFIG = {
    "SYMBOL": "XLV",
    "SEQUENCE_LENGTHS": [30],
    "TRAIN_SIZE_RATIO": 0.85,
    "EPOCHS": 100,
    "BATCH_SIZES": [64],
    "LSTM_UNITS": [64],
    "DROPOUT_RATES": [0.2],
    "MODEL_OPTIMIZER": "adam",
    "MODEL_LOSS": "mse",
    "CONFIDENCE_THRESHOLD": 0.95,
    "START_DATE": pd.to_datetime("2010-01-01"),
    "END_DATE": pd.to_datetime("2025-01-01"),
}

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(AttentionLayer, self).get_config()

def load_and_prepare_data():
    """Load and prepare price data with HBERT sentiment scores."""
    # Load price data
    print("Loading XLV price data...")
    price_df = pd.read_csv("xlv_data.csv")
    price_df["Date"] = pd.to_datetime(price_df["Unnamed: 0"])
    price_df.set_index("Date", inplace=True)
    price_df = price_df.sort_index()
    print(f"Price data shape: {price_df.shape}")

    # Load sentiment data
    print("Loading HBERT sentiment data...")
    sentiment_df = pd.read_csv("Data/Processed_Data/Recent_News_HBERT.csv")
    print(f"Initial sentiment data shape: {sentiment_df.shape}")
    
    # Extract dates from titles using regex
    def extract_date_from_title(title):
        if not isinstance(title, str):
            return None
            
        # Try to find dates in various formats
        patterns = [
            r'Q[1-4] \d{4}',  # Q1 2021
            r'\d{4}-\d{2}-\d{2}',  # 2021-10-25
            r'\d{2}/\d{2}/\d{4}',  # 10/25/2021
            r'\d{4}-\d{2}',  # 2021-10
            r'\d{4}',  # 2021
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title)
            if match:
                date_str = match.group()
                try:
                    if pattern == r'Q[1-4] \d{4}':
                        quarter, year = date_str.split()
                        month = (int(quarter[1]) - 1) * 3 + 1
                        return datetime(int(year), month, 1)
                    elif pattern == r'\d{4}-\d{2}-\d{2}':
                        return datetime.strptime(date_str, '%Y-%m-%d')
                    elif pattern == r'\d{2}/\d{2}/\d{4}':
                        return datetime.strptime(date_str, '%m/%d/%Y')
                    elif pattern == r'\d{4}-\d{2}':
                        return datetime.strptime(date_str, '%Y-%m')
                    elif pattern == r'\d{4}':
                        return datetime(int(date_str), 1, 1)
                except ValueError:
                    continue
        return None
    
    # Extract dates from titles
    sentiment_df['date'] = sentiment_df['title'].apply(extract_date_from_title)
    print(f"Extracted dates from {sentiment_df['date'].notna().sum()} articles")
    
    # For articles without dates, use the price data dates
    available_dates = price_df.index.tolist()
    missing_dates_mask = sentiment_df['date'].isna()
    num_missing = missing_dates_mask.sum()
    
    if num_missing > 0:
        # Distribute missing dates evenly across available dates
        date_indices = np.linspace(0, len(available_dates)-1, num_missing, dtype=int)
        sentiment_df.loc[missing_dates_mask, 'date'] = [available_dates[i] for i in date_indices]
        print(f"Assigned dates to {num_missing} articles without dates")
    
    sentiment_df.set_index('date', inplace=True)
    
    # Aggregate sentiment scores by date
    daily_sentiment = sentiment_df.groupby(sentiment_df.index).agg({
        'positive_score': 'mean',
        'neutral_score': 'mean',
        'negative_score': 'mean'
    })
    print(f"Daily sentiment shape: {daily_sentiment.shape}")
    
    # Merge price and sentiment data
    df = price_df.join(daily_sentiment, how='left')
    print(f"Shape after joining: {df.shape}")
    
    # Forward fill missing sentiment values
    df[['positive_score', 'neutral_score', 'negative_score']] = df[['positive_score', 'neutral_score', 'negative_score']].ffill()
    
    # Backward fill any remaining NaN values
    df[['positive_score', 'neutral_score', 'negative_score']] = df[['positive_score', 'neutral_score', 'negative_score']].bfill()
    
    # Filter date range
    df = df[(df.index >= CONFIG["START_DATE"]) & (df.index <= CONFIG["END_DATE"])]
    print(f"Shape after date filtering: {df.shape}")
    
    # Drop any remaining NaN values
    df = df.dropna()
    print(f"Final shape after dropping NaN: {df.shape}")
    print("\nSample of final data:")
    print(df.head())
    
    return df

def create_sequences(data, seq_length):
    """Create sequences from multivariate data including sentiment scores."""
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
        targets.append(data[i + seq_length, 0])  # Target is still the closing price
    return np.array(sequences), np.array(targets)

def build_lstm_model(seq_length, n_features, lstm_units, dropout_rate):
    """Build Bidirectional LSTM model with attention mechanism."""
    inputs = Input(shape=(seq_length, n_features))
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(lstm_units // 2, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    x = AttentionLayer()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=CONFIG["MODEL_OPTIMIZER"], loss=CONFIG["MODEL_LOSS"])
    return model

def calculate_confidence(predictions, actuals):
    """Calculate confidence based on prediction error."""
    # Calculate percentage error
    percentage_errors = np.abs((predictions.flatten() - actuals.flatten()) / actuals.flatten())
    # Convert to confidence score (0 to 1)
    confidence = np.exp(-percentage_errors)  # exponential decay of error
    return confidence

def generate_trading_signals(predictions, actuals, confidence_threshold):
    confidence = calculate_confidence(predictions.flatten(), actuals)
    signals = np.zeros(len(predictions))
    
    high_confidence = confidence >= confidence_threshold
    price_diff_pct = (predictions.flatten() - actuals.flatten()) / actuals.flatten()
    
    # Add a threshold for minimum price difference
    min_price_diff = 0.01  # 1% minimum difference
    
    signals[high_confidence & (price_diff_pct > min_price_diff)] = 1   # Buy signals
    signals[high_confidence & (price_diff_pct < -min_price_diff)] = -1 # Sell signals
    
    return signals, confidence

def find_best_hyperparameters(data, train_size):
    """Find the best hyperparameters for the LSTM model."""
    best_val_loss = float("inf")
    best_model = None
    best_history = None
    best_params = None

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )

    for seq_length in CONFIG["SEQUENCE_LENGTHS"]:
        X, y = create_sequences(data, seq_length)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        for batch_size in CONFIG["BATCH_SIZES"]:
            for lstm_units in CONFIG["LSTM_UNITS"]:
                for dropout_rate in CONFIG["DROPOUT_RATES"]:
                    print(
                        f"Testing: seq_length={seq_length}, batch_size={batch_size}, "
                        f"lstm_units={lstm_units}, dropout_rate={dropout_rate}"
                    )

                    model = build_lstm_model(
                        seq_length, X.shape[2], lstm_units, dropout_rate
                    )
                    history = model.fit(
                        X_train,
                        y_train,
                        epochs=CONFIG["EPOCHS"],
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=1,
                    )

                    val_loss = min(history.history["val_loss"])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = model
                        best_history = history
                        best_params = {
                            "sequence_length": seq_length,
                            "batch_size": batch_size,
                            "lstm_units": lstm_units,
                            "dropout_rate": dropout_rate,
                        }

                    print(f"Validation loss: {val_loss:.6f}")

    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return best_model, best_history, best_params

def main():
    # Load and prepare data
    df = load_and_prepare_data()

    # Select features for the model
    feature_columns = [
        "Close",
        "8_EMA",
        "200_SMA",
        "Volume",
        "positive_score",
        "neutral_score",
        "negative_score"
    ]

    # Verify which columns are actually available
    available_columns = [col for col in feature_columns if col in df.columns]
    missing_columns = set(feature_columns) - set(available_columns)
    if missing_columns:
        print(
            f"Warning: The following columns are missing and will be excluded: {missing_columns}"
        )
        feature_columns = available_columns

    # Clean and convert
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)

    # Scale the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])

    # Split into training and testing
    train_size = int(len(scaled_data) * CONFIG["TRAIN_SIZE_RATIO"])

    # Tune and get best model
    best_model, history, best_params = find_best_hyperparameters(
        scaled_data, train_size
    )

    # Final training and prediction
    X, y = create_sequences(scaled_data, best_params["sequence_length"])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Predictions
    best_model.fit(
        X_train,
        y_train,
        epochs=CONFIG["EPOCHS"],
        batch_size=best_params["batch_size"],
        verbose=0,
    )
    best_model.save("models/lstm_hbert_model_XLV.h5")
    print("Final model saved to models/lstm_hbert_model_XLV.h5")

    train_predictions = best_model.predict(X_train, verbose=0)
    test_predictions = best_model.predict(X_test, verbose=0)

    # Inverse transform predictions (for closing price only)
    close_price_scaler = MinMaxScaler()
    close_price_scaler.fit_transform(df[["Close"]])

    train_predictions = close_price_scaler.inverse_transform(train_predictions)
    test_predictions = close_price_scaler.inverse_transform(test_predictions)
    y_train_inv = close_price_scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = close_price_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Confidence and Signals
    test_signals, confidence = generate_trading_signals(
        test_predictions, y_test_inv, CONFIG["CONFIDENCE_THRESHOLD"]
    )

    # RMSE
    train_rmse = np.sqrt(np.mean((train_predictions - y_train_inv) ** 2))
    test_rmse = np.sqrt(np.mean((test_predictions - y_test_inv) ** 2))
    print(f"Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # First subplot: Stock price and predictions
    ax1.plot(df.index[-len(y_test_inv):], y_test_inv, label="Actual", color="blue")
    ax1.plot(
        df.index[-len(test_predictions):],
        test_predictions,
        label="Predicted",
        color="orange",
    )

    buy_signals = test_signals == 1
    sell_signals = test_signals == -1
    ax1.scatter(
        df.index[-len(y_test_inv):][buy_signals],
        y_test_inv[buy_signals],
        color="green",
        marker="^",
        s=100,
        label="Buy Signal",
    )
    ax1.scatter(
        df.index[-len(y_test_inv):][sell_signals],
        y_test_inv[sell_signals],
        color="red",
        marker="v",
        s=100,
        label="Sell Signal",
    )

    ax1.set_title("XLV Stock Price Prediction")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(True)

    # Second subplot: Sentiment scores
    ax2.plot(
        df.index[-len(y_test_inv):],
        df["positive_score"].iloc[-len(y_test_inv):] * 100,
        label="Positive Sentiment",
        color="green",
        linestyle="--",
        alpha=0.7,
    )
    ax2.plot(
        df.index[-len(y_test_inv):],
        df["negative_score"].iloc[-len(y_test_inv):] * 100,
        label="Negative Sentiment",
        color="red",
        linestyle="--",
        alpha=0.7,
    )
    ax2.plot(
        df.index[-len(y_test_inv):],
        df["neutral_score"].iloc[-len(y_test_inv):] * 100,
        label="Neutral Sentiment",
        color="gray",
        linestyle="--",
        alpha=0.7,
    )

    ax2.set_title("HBERT Sentiment Analysis")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sentiment Score (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("Data/Figures/training_history_XLV_HBERT.png")

    # Next day prediction
    last_sequence = scaled_data[-best_params["sequence_length"]:]
    last_sequence = last_sequence.reshape(
        (1, best_params["sequence_length"], scaled_data.shape[1])
    )
    next_day_pred = best_model.predict(last_sequence, verbose=0)
    next_day_pred = close_price_scaler.inverse_transform(next_day_pred)

    last_actual = df["Close"].iloc[-1]
    confidence_next = calculate_confidence(
        next_day_pred.flatten(), np.array([last_actual])
    )

    print(f"\nPredicted price for next day: ${next_day_pred[0][0]:.2f}")
    print(f"Confidence level: {confidence_next[0]:.2%}")

    if confidence_next[0] >= CONFIG["CONFIDENCE_THRESHOLD"]:
        if next_day_pred[0][0] > last_actual:
            print("High confidence BUY signal")
        else:
            print("High confidence SELL signal")
    else:
        print("No trading signal - confidence below threshold")

    # Save predictions with additional features
    predictions_df = pd.DataFrame(
        {
            "Date": df.index[-len(test_predictions):],
            "Actual": y_test_inv.flatten(),
            "Predicted": test_predictions.flatten(),
            "Signal": test_signals.flatten(),
            "Confidence": confidence.flatten(),
            "Positive_Sentiment": df["positive_score"].iloc[-len(test_predictions):].values,
            "Negative_Sentiment": df["negative_score"].iloc[-len(test_predictions):].values,
            "Neutral_Sentiment": df["neutral_score"].iloc[-len(test_predictions):].values,
        }
    )
    predictions_df.to_csv("lstm_hbert_predictions_XLV.csv", index=False)
    print("Predictions saved to lstm_hbert_predictions_XLV.csv")

if __name__ == "__main__":
    main() 