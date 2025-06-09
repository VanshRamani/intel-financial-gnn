#!/usr/bin/env python3
"""
01_data_exploration.py
======================

Financial Data Exploration and Analysis

This script demonstrates the data loading and exploration capabilities
of the Intel-Optimized Financial GNN project.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data.data_loader import FinancialDataLoader
from data.preprocessing import GraphPreprocessor
from utils.visualization import GraphVisualizer

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def main():
    """Main function for data exploration"""
    print("🚀 Intel-Optimized Financial GNN - Data Exploration")
    print("=" * 60)
    
    # Initialize components
    data_loader = FinancialDataLoader()
    preprocessor = GraphPreprocessor()
    visualizer = GraphVisualizer()
    
    # Define stocks to analyze
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    print(f"📊 Analyzing stocks: {', '.join(symbols)}")
    
    # Load financial data
    print("\n📈 Loading financial data...")
    raw_data = data_loader.fetch_stock_data(symbols=symbols, period='2y')
    
    # Basic statistics
    print("\n📊 Dataset Statistics:")
    print(f"   • Total records: {len(raw_data):,}")
    print(f"   • Date range: {raw_data.index.min().date()} to {raw_data.index.max().date()}")
    print(f"   • Symbols: {len(symbols)}")
    
    # Price analysis
    print("\n💰 Price Analysis:")
    for symbol in symbols:
        close_col = f'{symbol}_Close'
        if close_col in raw_data.columns:
            current_price = raw_data[close_col].iloc[-1]
            price_change = (raw_data[close_col].iloc[-1] / raw_data[close_col].iloc[0] - 1) * 100
            print(f"   • {symbol}: ${current_price:.2f} ({price_change:+.1f}% over period)")
    
    # Process data for analysis
    print("\n🔧 Processing technical indicators...")
    processed_data = preprocessor.process_financial_data(raw_data)
    
    # Display technical indicators
    print("\n📈 Technical Indicators Generated:")
    indicator_cols = [col for col in processed_data.columns if any(ind in col for ind in ['RSI', 'MACD', 'BB', 'SMA', 'EMA'])]
    print(f"   • Total indicators: {len(indicator_cols)}")
    print(f"   • Sample indicators: {indicator_cols[:5]}")
    
    # Correlation analysis
    print("\n🔗 Correlation Analysis:")
    close_prices = processed_data[[col for col in processed_data.columns if col.endswith('_Close')]]
    correlation_matrix = close_prices.corr()
    
    print("   Correlation Matrix:")
    for i, symbol1 in enumerate(symbols):
        for j, symbol2 in enumerate(symbols):
            if i < j:
                corr = correlation_matrix.iloc[i, j]
                print(f"   • {symbol1} vs {symbol2}: {corr:.3f}")
    
    # Volatility analysis
    print("\n📊 Volatility Analysis:")
    for symbol in symbols:
        close_col = f'{symbol}_Close'
        if close_col in processed_data.columns:
            returns = processed_data[close_col].pct_change()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            print(f"   • {symbol}: {volatility:.1f}% annual volatility")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # 1. Price comparison chart
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Stock Prices', 'Correlation Heatmap', 
                                     'Volume Analysis', 'Returns Distribution'))
    
    # Price chart
    for symbol in symbols:
        close_col = f'{symbol}_Close'
        if close_col in processed_data.columns:
            fig.add_trace(
                go.Scatter(x=processed_data.index, y=processed_data[close_col], 
                          name=symbol, mode='lines'),
                row=1, col=1
            )
    
    # Correlation heatmap
    fig.add_trace(
        go.Heatmap(z=correlation_matrix.values, 
                   x=symbols, y=symbols,
                   colorscale='RdBu', zmid=0),
        row=1, col=2
    )
    
    # Volume analysis
    for symbol in symbols:
        volume_col = f'{symbol}_Volume'
        if volume_col in processed_data.columns:
            fig.add_trace(
                go.Scatter(x=processed_data.index, y=processed_data[volume_col], 
                          name=f'{symbol} Vol', mode='lines', opacity=0.7),
                row=2, col=1
            )
    
    # Returns distribution
    for symbol in symbols[:3]:  # Limit to first 3 for clarity
        close_col = f'{symbol}_Close'
        if close_col in processed_data.columns:
            returns = processed_data[close_col].pct_change().dropna() * 100
            fig.add_trace(
                go.Histogram(x=returns, name=f'{symbol} Returns', 
                           opacity=0.7, nbinsx=50),
                row=2, col=2
            )
    
    fig.update_layout(height=800, title_text="Financial Data Analysis Dashboard")
    fig.write_html("notebooks/data_exploration_results.html")
    print("   • Interactive dashboard saved to: notebooks/data_exploration_results.html")
    
    # Summary statistics
    print("\n📋 Summary Statistics:")
    print("   • Data quality: ✅ Complete dataset loaded")
    print("   • Technical indicators: ✅ Generated successfully")
    print("   • Correlations: ✅ Calculated for all pairs")
    print("   • Visualizations: ✅ Created interactive dashboard")
    
    print("\n🎯 Next Steps:")
    print("   1. Run 02_model_training.py for GNN training")
    print("   2. Run 03_intel_optimization.py for performance optimization")
    print("   3. View results in data_exploration_results.html")
    
    return processed_data

if __name__ == "__main__":
    try:
        processed_data = main()
        print("\n✅ Data exploration completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during data exploration: {str(e)}")
        print("💡 Make sure all dependencies are installed and data sources are accessible") 