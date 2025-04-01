import pandas as pd
import numpy as np

# Trading simulation configuration
INITIAL_BANKROLL = 10000
TRADE_PERCENTAGE = 0.3  # Default trade percentage if not using confidence-based sizing
CONFIDENCE_THRESHOLD = 0.95  # Minimum confidence level required to execute trades
STOP_LOSS_PERCENTAGE = 0.03  # % stop loss
PREDICTIONS_FILE = 'lstm_predictions_jnj.csv'

# Confidence-based trade sizing configuration
CONFIDENCE_TRADE_SIZES = {
    0.99: 0.4,  # 40% of bankroll for very high confidence trades
    0.97: 0.3,  # 30% of bankroll for high confidence trades
    0.95: 0.2,  # 20% of bankroll for moderate confidence trades
}

def simulate_trading(predictions_file=PREDICTIONS_FILE, 
                    initial_bankroll=INITIAL_BANKROLL, 
                    trade_percentage=TRADE_PERCENTAGE, 
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    stop_loss_percentage=STOP_LOSS_PERCENTAGE):
    # Load predictions
    df = pd.read_csv(predictions_file)
    
    # Initialize variables
    bankroll = initial_bankroll
    position = 0  # 0 = no position, 1 = long position
    shares = 0
    trades = []
    entry_price = 0  # Track entry price for calculating profit/loss
    profitable_trades = 0
    total_trades = 0
    stop_loss_triggered = 0  # Track number of stop loss triggers
    
    # Iterate through predictions
    for i in range(len(df)):
        current_price = df['Actual'].iloc[i]
        signal = df['Signal'].iloc[i]
        date = df['Date'].iloc[i]
        confidence = df['Confidence'].iloc[i] if 'Confidence' in df.columns else 1.0
        
        # Determine trade percentage based on confidence using configuration
        current_trade_percentage = 0  # Default to no trade
        for conf_threshold, trade_size in sorted(CONFIDENCE_TRADE_SIZES.items(), reverse=True):
            if confidence >= conf_threshold:
                current_trade_percentage = trade_size
                break
        
        # Calculate trade amount based on confidence-adjusted percentage
        trade_amount = bankroll * current_trade_percentage
        
        # Check stop loss if we have a position
        if position == 1:
            loss_percentage = (current_price - entry_price) / entry_price
            if loss_percentage <= -stop_loss_percentage:
                # Stop loss triggered - sell position
                trade_value = shares * current_price
                bankroll += trade_value
                position = 0
                
                # Calculate loss
                profit = trade_value - (shares * entry_price)
                total_trades += 1
                stop_loss_triggered += 1
                
                trades.append({
                    'Date': date,
                    'Action': 'STOP_LOSS',
                    'Price': current_price,
                    'Shares': shares,
                    'Trade Amount': trade_value,
                    'Bankroll': bankroll,
                    'Profit': profit,
                    'Confidence': confidence,
                    'Loss Percentage': loss_percentage * 100
                })
                shares = 0
                continue
        
        # Process buy signal with confidence check
        if signal == 1 and position == 0 and confidence >= confidence_threshold:
            shares = trade_amount / current_price
            bankroll -= trade_amount
            position = 1
            entry_price = current_price
            trades.append({
                'Date': date,
                'Action': 'BUY',
                'Price': current_price,
                'Shares': shares,
                'Trade Amount': trade_amount,
                'Bankroll': bankroll,
                'Confidence': confidence
            })
        
        # Process sell signal
        elif signal == -1 and position == 1:
            trade_value = shares * current_price
            bankroll += trade_value
            position = 0
            
            # Calculate if trade was profitable
            profit = trade_value - (shares * entry_price)
            if profit > 0:
                profitable_trades += 1
            total_trades += 1
            
            trades.append({
                'Date': date,
                'Action': 'SELL',
                'Price': current_price,
                'Shares': shares,
                'Trade Amount': trade_value,
                'Bankroll': bankroll,
                'Profit': profit,
                'Confidence': confidence
            })
            shares = 0
    
    # Close any remaining position using the last price
    if position == 1:
        trade_value = shares * df['Actual'].iloc[-1]
        bankroll += trade_value
        
        # Calculate if final trade was profitable
        profit = trade_value - (shares * entry_price)
        if profit > 0:
            profitable_trades += 1
        total_trades += 1
        
        trades.append({
            'Date': df['Date'].iloc[-1],
            'Action': 'SELL',
            'Price': df['Actual'].iloc[-1],
            'Shares': shares,
            'Trade Amount': trade_value,
            'Bankroll': bankroll,
            'Profit': profit,
            'Confidence': df['Confidence'].iloc[-1] if 'Confidence' in df.columns else 1.0
        })
    
    # Calculate performance metrics
    total_profit_loss = bankroll - initial_bankroll
    roi_percentage = (total_profit_loss / initial_bankroll) * 100
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(trades)
    
    return {
        'final_bankroll': bankroll,
        'total_profit_loss': total_profit_loss,
        'roi_percentage': roi_percentage,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'stop_loss_triggered': stop_loss_triggered,
        'trades': trades_df
    }

def main():
    # Run simulation
    results = simulate_trading()
    
    # Print results
    print("\nTrading Simulation Results")
    print("=" * 50)
    print(f"Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD:.2f}")
    print(f"Stop Loss Percentage: {STOP_LOSS_PERCENTAGE:.1%}")
    print(f"Final Bankroll: ${results['final_bankroll']:,.2f}")
    print(f"Total Profit/Loss: ${results['total_profit_loss']:,.2f}")
    print(f"ROI: {results['roi_percentage']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Profitable Trades: {results['profitable_trades']}")
    print(f"Stop Loss Triggers: {results['stop_loss_triggered']}")
    
    # Print trade history
    print("\nTrade History:")
    print("=" * 50)
    if len(results['trades']) > 0:
        print(results['trades'].to_string(index=False))
    else:
        print("No trades were executed")

if __name__ == "__main__":
    main()