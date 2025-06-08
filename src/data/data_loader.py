"""
Financial Data Loader

Fetches financial data from multiple sources including Yahoo Finance,
Alpha Vantage, and other financial APIs.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class FinancialDataLoader:
    """
    Comprehensive financial data loader supporting multiple data sources
    """
    
    def __init__(self, cache_enabled=True):
        self.cache_enabled = cache_enabled
        self.cached_data = {}
        
    def fetch_stock_data(self, symbols: List[str], period: str = "2y", 
                        interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        logger.info(f"Fetching data for {len(symbols)} symbols over {period}")
        
        stock_data = {}
        
        for symbol in symbols:
            try:
                # Check cache first
                cache_key = f"{symbol}_{period}_{interval}"
                if self.cache_enabled and cache_key in self.cached_data:
                    stock_data[symbol] = self.cached_data[cache_key]
                    continue
                
                # Fetch from Yahoo Finance
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Add technical indicators
                data = self._add_technical_indicators(data)
                
                # Cache the data
                if self.cache_enabled:
                    self.cached_data[cache_key] = data
                
                stock_data[symbol] = data
                logger.info(f"Loaded {len(data)} records for {symbol}")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded data for {len(stock_data)} symbols")
        return stock_data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to stock data
        
        Args:
            data: Stock price DataFrame
            
        Returns:
            DataFrame with additional technical indicators
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / df['BB_width']
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['Price_change'] = df['Close'].pct_change()
        df['Price_momentum_5'] = df['Close'].pct_change(periods=5)
        df['Price_momentum_10'] = df['Close'].pct_change(periods=10)
        
        # Volatility
        df['Volatility_5'] = df['Price_change'].rolling(window=5).std()
        df['Volatility_20'] = df['Price_change'].rolling(window=20).std()
        
        # Support and Resistance levels
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Low_20'] = df['Low'].rolling(window=20).min()
        
        return df
    
    def get_market_data(self, symbols: List[str]) -> Dict[str, any]:
        """
        Get additional market data including sector information, market cap, etc.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with market data
        """
        market_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                market_data[symbol] = {
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', None),
                    'beta': info.get('beta', None),
                    'dividend_yield': info.get('dividendYield', 0),
                    'country': info.get('country', 'Unknown')
                }
                
            except Exception as e:
                logger.error(f"Error fetching market data for {symbol}: {str(e)}")
                market_data[symbol] = {
                    'sector': 'Unknown',
                    'industry': 'Unknown',
                    'market_cap': 0,
                    'pe_ratio': None,
                    'beta': None,
                    'dividend_yield': 0,
                    'country': 'Unknown'
                }
        
        return market_data
    
    def calculate_returns(self, stock_data: Dict[str, pd.DataFrame], 
                         periods: List[int] = [1, 5, 10, 20]) -> Dict[str, pd.DataFrame]:
        """
        Calculate various return periods for stocks
        
        Args:
            stock_data: Dictionary of stock DataFrames
            periods: List of periods to calculate returns for
            
        Returns:
            Dictionary with return data
        """
        returns_data = {}
        
        for symbol, data in stock_data.items():
            returns_df = pd.DataFrame(index=data.index)
            
            for period in periods:
                returns_df[f'return_{period}d'] = data['Close'].pct_change(periods=period)
                returns_df[f'volatility_{period}d'] = returns_df[f'return_{period}d'].rolling(window=period).std()
            
            returns_data[symbol] = returns_df
        
        return returns_data
    
    def get_correlation_matrix(self, stock_data: Dict[str, pd.DataFrame], 
                              column: str = 'Close') -> pd.DataFrame:
        """
        Calculate correlation matrix between stocks
        
        Args:
            stock_data: Dictionary of stock DataFrames
            column: Column to calculate correlations for
            
        Returns:
            Correlation matrix DataFrame
        """
        # Create DataFrame with all stock prices
        price_data = pd.DataFrame()
        
        for symbol, data in stock_data.items():
            price_data[symbol] = data[column]
        
        # Calculate correlation matrix
        correlation_matrix = price_data.corr()
        
        return correlation_matrix 