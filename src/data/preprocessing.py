"""
Graph Data Preprocessing

Handles preprocessing of financial data for graph neural networks,
including feature engineering, normalization, and target creation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

logger = logging.getLogger(__name__)


class GraphPreprocessor:
    """
    Preprocessor for financial data to create graph-ready features
    """
    
    def __init__(self, scaler_type='standard', feature_selection=True):
        self.scaler_type = scaler_type
        self.feature_selection = feature_selection
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_columns = []
        
    def process_financial_data(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process raw financial data for graph neural network training
        
        Args:
            stock_data: Dictionary of stock DataFrames
            
        Returns:
            Dictionary of processed DataFrames
        """
        logger.info("Processing financial data for graph neural networks...")
        
        processed_data = {}
        
        for symbol, data in stock_data.items():
            logger.info(f"Processing {symbol}...")
            
            # Create features and targets
            processed_df = self._create_features(data)
            processed_df = self._create_targets(processed_df)
            
            # Handle missing values
            processed_df = self._handle_missing_values(processed_df)
            
            # Feature engineering
            processed_df = self._engineer_features(processed_df)
            
            processed_data[symbol] = processed_df
        
        # Apply scaling and feature selection across all stocks
        processed_data = self._apply_scaling_and_selection(processed_data)
        
        logger.info("Financial data processing completed")
        return processed_data
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create base features from stock data"""
        df = data.copy()
        
        # Price-based features
        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        df['upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df['lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
        
        # Volume-based features
        df['volume_price_trend'] = df['Volume'] * ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1))
        df['ease_of_movement'] = ((df['High'] + df['Low'])/2 - (df['High'].shift(1) + df['Low'].shift(1))/2) * df['Volume'] / (df['High'] - df['Low'])
        
        # Momentum features
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'volatility_momentum_{period}'] = df['Volatility_5'].rolling(window=period).mean()
        
        # Mean reversion features
        df['distance_from_sma20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['distance_from_sma50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        
        # Trend strength
        df['trend_strength'] = np.abs(df['Close'] - df['SMA_20']) / df['Volatility_20']
        
        return df
    
    def _create_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create prediction targets"""
        df = data.copy()
        
        # Future price movements (classification targets)
        df['future_return_1d'] = df['Close'].shift(-1) / df['Close'] - 1
        df['future_return_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        df['future_return_10d'] = df['Close'].shift(-10) / df['Close'] - 1
        
        # Binary targets for direction prediction
        df['price_up_1d'] = (df['future_return_1d'] > 0).astype(int)
        df['price_up_5d'] = (df['future_return_5d'] > 0).astype(int)
        
        # Volatility targets
        df['future_volatility_5d'] = df['Close'].pct_change().shift(-5).rolling(window=5).std()
        
        return df
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = data.copy()
        
        # Forward fill first, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # For remaining NaN values, use interpolation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].interpolate()
        
        # Drop rows with any remaining NaN values
        df = df.dropna()
        
        return df
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering"""
        df = data.copy()
        
        # Technical pattern features
        df['doji'] = ((np.abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])) < 0.1).astype(int)
        df['hammer'] = ((df['lower_shadow'] > 2 * np.abs(df['Close'] - df['Open'])) & 
                       (df['upper_shadow'] < 0.1 * np.abs(df['Close'] - df['Open']))).astype(int)
        
        # Market regime features
        df['high_volatility_regime'] = (df['Volatility_20'] > df['Volatility_20'].rolling(window=100).quantile(0.75)).astype(int)
        df['trending_market'] = (np.abs(df['SMA_5'] - df['SMA_20']) > df['Volatility_20']).astype(int)
        
        # Interaction features
        df['rsi_bb_interaction'] = df['RSI'] * df['BB_position']
        df['volume_momentum_interaction'] = df['Volume_ratio'] * df['momentum_5']
        
        # Lag features
        for col in ['RSI', 'MACD', 'BB_position', 'Volume_ratio']:
            if col in df.columns:
                for lag in [1, 2, 3]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def _apply_scaling_and_selection(self, processed_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply scaling and feature selection across all stocks"""
        
        # Identify feature and target columns
        sample_df = list(processed_data.values())[0]
        feature_cols = [col for col in sample_df.columns if not col.startswith('future_') and col not in ['price_up_1d', 'price_up_5d']]
        target_cols = [col for col in sample_df.columns if col.startswith('future_') or col.startswith('price_up')]
        
        # Combine all data for fitting scalers
        all_features = []
        for symbol, data in processed_data.items():
            features = data[feature_cols].values
            all_features.append(features)
        
        all_features = np.vstack(all_features)
        
        # Fit scaler
        if self.scaler_type == 'standard':
            scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = None
        
        if scaler:
            scaler.fit(all_features)
            self.scalers['features'] = scaler
        
        # Apply scaling and feature selection to each stock
        scaled_data = {}
        for symbol, data in processed_data.items():
            scaled_df = data.copy()
            
            # Scale features
            if scaler:
                scaled_features = scaler.transform(data[feature_cols].values)
                scaled_df[feature_cols] = scaled_features
            
            # Feature selection (if enabled)
            if self.feature_selection and 'future_return_1d' in scaled_df.columns:
                # Remove rows with NaN targets for feature selection
                selection_data = scaled_df.dropna(subset=['future_return_1d'])
                
                if len(selection_data) > 50:  # Minimum samples for feature selection
                    selector = SelectKBest(score_func=f_regression, k=min(50, len(feature_cols)))
                    selector.fit(selection_data[feature_cols], selection_data['future_return_1d'])
                    
                    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
                    self.feature_selectors[symbol] = selector
                    self.feature_columns = selected_features
                    
                    logger.info(f"Selected {len(selected_features)} features for {symbol}")
                else:
                    self.feature_columns = feature_cols
            else:
                self.feature_columns = feature_cols
            
            scaled_data[symbol] = scaled_df
        
        return scaled_data
    
    def get_feature_importance(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get feature importance scores for a specific symbol"""
        if symbol in self.feature_selectors:
            selector = self.feature_selectors[symbol]
            feature_names = self.feature_columns
            scores = selector.scores_
            
            importance_dict = dict(zip(feature_names, scores))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return None
    
    def transform_new_data(self, new_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Transform new data using fitted scalers and feature selectors"""
        transformed_data = {}
        
        for symbol, data in new_data.items():
            # Apply same preprocessing steps
            processed_df = self._create_features(data)
            processed_df = self._create_targets(processed_df)
            processed_df = self._handle_missing_values(processed_df)
            processed_df = self._engineer_features(processed_df)
            
            # Apply scaling
            if 'features' in self.scalers:
                feature_cols = self.feature_columns
                scaled_features = self.scalers['features'].transform(processed_df[feature_cols].values)
                processed_df[feature_cols] = scaled_features
            
            transformed_data[symbol] = processed_df
        
        return transformed_data 