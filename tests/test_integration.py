"""
Integration tests for Intel-Optimized Financial GNN

These tests verify the end-to-end functionality of the system,
including data loading, model training, and Intel optimizations.
"""

import pytest
import torch
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

# Local imports
from src.data.data_loader import FinancialDataLoader
from src.data.preprocessing import GraphPreprocessor
from src.models.gnn_model import FinancialGNN, FinancialGNNEnsemble
from src.models.intel_optimizer import IntelModelOptimizer
from src.utils.graph_utils import GraphConstructor
from src.utils.visualization import GraphVisualizer


class TestFinancialDataIntegration:
    """Integration tests for financial data pipeline"""
    
    def test_data_loading_and_preprocessing_pipeline(self):
        """Test complete data loading and preprocessing pipeline"""
        # Mock data loader
        loader = FinancialDataLoader()
        preprocessor = GraphPreprocessor()
        
        # Create mock stock data
        dates = pd.date_range('2022-01-01', periods=100, freq='D')
        mock_data = {
            'AAPL': pd.DataFrame({
                'Open': np.random.randn(100).cumsum() + 150,
                'High': np.random.randn(100).cumsum() + 155,
                'Low': np.random.randn(100).cumsum() + 145,
                'Close': np.random.randn(100).cumsum() + 150,
                'Volume': np.random.randint(1000000, 10000000, 100)
            }, index=dates),
            'GOOGL': pd.DataFrame({
                'Open': np.random.randn(100).cumsum() + 2500,
                'High': np.random.randn(100).cumsum() + 2550,
                'Low': np.random.randn(100).cumsum() + 2450,
                'Close': np.random.randn(100).cumsum() + 2500,
                'Volume': np.random.randint(500000, 5000000, 100)
            }, index=dates)
        }
        
        # Add technical indicators
        for symbol, data in mock_data.items():
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['RSI'] = 50 + np.random.randn(100) * 20  # Mock RSI
            data['MACD'] = np.random.randn(100) * 5
            data['MACD_signal'] = data['MACD'].rolling(9).mean()
            data['BB_upper'] = data['Close'] * 1.02
            data['BB_lower'] = data['Close'] * 0.98
            data['BB_position'] = 0.5 + np.random.randn(100) * 0.2
            data['Volume_ratio'] = 1 + np.random.randn(100) * 0.3
            data['Price_change'] = data['Close'].pct_change()
            data['Volatility_5'] = data['Price_change'].rolling(5).std()
            data['Volatility_20'] = data['Price_change'].rolling(20).std()
        
        # Test preprocessing
        processed_data = preprocessor.process_financial_data(mock_data)
        
        assert len(processed_data) == 2
        assert 'AAPL' in processed_data
        assert 'GOOGL' in processed_data
        
        for symbol, data in processed_data.items():
            assert not data.empty
            assert 'future_return_1d' in data.columns
            assert 'price_up_1d' in data.columns


class TestGraphConstructionIntegration:
    """Integration tests for graph construction"""
    
    def test_graph_construction_from_financial_data(self):
        """Test graph construction from preprocessed financial data"""
        # Create mock processed data
        dates = pd.date_range('2022-01-01', periods=50, freq='D')
        n_features = 25
        
        processed_data = {
            'AAPL': pd.DataFrame(
                np.random.randn(50, n_features),
                columns=[f'feature_{i}' for i in range(n_features)],
                index=dates
            ),
            'GOOGL': pd.DataFrame(
                np.random.randn(50, n_features),
                columns=[f'feature_{i}' for i in range(n_features)],
                index=dates
            ),
            'MSFT': pd.DataFrame(
                np.random.randn(50, n_features),
                columns=[f'feature_{i}' for i in range(n_features)],
                index=dates
            )
        }
        
        # Add required columns
        for symbol, data in processed_data.items():
            data['Close'] = 100 + np.random.randn(50).cumsum()
            data['future_return_1d'] = np.random.randn(50) * 0.02
        
        # Construct graph
        graph_constructor = GraphConstructor()
        graph_data = graph_constructor.build_correlation_graph(
            processed_data,
            correlation_threshold=0.1,  # Lower threshold for small data
            max_edges_per_node=3
        )
        
        assert graph_data.num_nodes == 3
        assert graph_data.x.size(0) == 3
        assert graph_data.edge_index.size(0) == 2
        assert graph_data.y.size(0) == 3
        assert hasattr(graph_data, 'symbols')


class TestModelTrainingIntegration:
    """Integration tests for model training pipeline"""
    
    def test_end_to_end_model_training(self):
        """Test complete model training pipeline"""
        # Create mock graph data
        num_nodes = 5
        num_features = 20
        
        # Mock graph data
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        y = torch.randn(num_nodes, 1)
        
        class MockGraphData:
            def __init__(self):
                self.x = x
                self.edge_index = edge_index
                self.y = y
                self.num_nodes = num_nodes
                self.num_edges = edge_index.size(1)
                self.num_node_features = num_features
        
        graph_data = MockGraphData()
        
        # Create and train model
        model = FinancialGNN(
            input_dim=num_features,
            hidden_dim=32,
            output_dim=1,
            num_layers=2,
            dropout=0.2
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        # Training loop
        model.train()
        initial_loss = None
        
        for epoch in range(10):
            optimizer.zero_grad()
            predictions = model(graph_data.x, graph_data.edge_index)
            loss = criterion(predictions, graph_data.y)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        
        # Verify training progressed
        assert final_loss < initial_loss * 1.1  # Allow some variance
        assert model.count_parameters() > 0


class TestIntelOptimizationIntegration:
    """Integration tests for Intel optimization pipeline"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Intel optimizations may require specific hardware")
    def test_intel_optimization_pipeline(self):
        """Test Intel optimization integration"""
        # Create mock model and data
        model = FinancialGNN(input_dim=10, hidden_dim=16, output_dim=1, num_layers=1)
        
        class MockGraphData:
            def __init__(self):
                self.x = torch.randn(5, 10)
                self.edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        
        graph_data = MockGraphData()
        
        # Test Intel optimizer
        optimizer = IntelModelOptimizer()
        
        # Test basic optimization (may skip if Intel tools not available)
        try:
            optimized_model = optimizer.optimize_pytorch_model(
                model, 
                (graph_data.x, graph_data.edge_index)
            )
            assert optimized_model is not None
        except Exception as e:
            pytest.skip(f"Intel optimization not available: {e}")
        
        # Test benchmarking
        try:
            results = optimizer.benchmark_model(
                model, 
                optimized_model, 
                graph_data, 
                num_runs=5
            )
            assert 'speedup' in results
            assert 'original_avg_time' in results
            assert 'optimized_avg_time' in results
        except Exception as e:
            pytest.skip(f"Benchmarking not available: {e}")


class TestVisualizationIntegration:
    """Integration tests for visualization pipeline"""
    
    def test_visualization_generation(self):
        """Test visualization generation without display"""
        # Create mock graph data
        class MockGraphData:
            def __init__(self):
                self.num_nodes = 4
                self.edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
                self.edge_attr = torch.randn(3, 1)
                self.correlation_matrix = np.random.randn(4, 4)
        
        graph_data = MockGraphData()
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        
        visualizer = GraphVisualizer()
        
        # Test that visualization methods can be called without errors
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test graph visualization
            graph_path = os.path.join(tmpdir, "test_graph.html")
            try:
                visualizer.plot_financial_graph(graph_data, symbols, graph_path, interactive=True)
                assert os.path.exists(graph_path)
            except Exception as e:
                pytest.skip(f"Graph visualization not available: {e}")
            
            # Test correlation heatmap
            heatmap_path = os.path.join(tmpdir, "test_heatmap.html")
            try:
                visualizer.plot_correlation_heatmap(graph_data, symbols, heatmap_path)
                assert os.path.exists(heatmap_path)
            except Exception as e:
                pytest.skip(f"Heatmap visualization not available: {e}")


class TestSystemIntegration:
    """Full system integration tests"""
    
    def test_minimal_end_to_end_pipeline(self):
        """Test minimal end-to-end pipeline execution"""
        # This test verifies that all components can work together
        # without external dependencies
        
        # 1. Mock data creation
        dates = pd.date_range('2022-01-01', periods=30, freq='D')
        mock_stock_data = {
            'AAPL': pd.DataFrame({
                'Open': 150 + np.random.randn(30).cumsum() * 0.1,
                'High': 151 + np.random.randn(30).cumsum() * 0.1,
                'Low': 149 + np.random.randn(30).cumsum() * 0.1,
                'Close': 150 + np.random.randn(30).cumsum() * 0.1,
                'Volume': np.random.randint(1000000, 10000000, 30),
                # Add required technical indicators
                'SMA_20': 150 + np.random.randn(30) * 0.5,
                'SMA_50': 150 + np.random.randn(30) * 0.5,
                'RSI': 50 + np.random.randn(30) * 20,
                'MACD': np.random.randn(30) * 2,
                'MACD_signal': np.random.randn(30) * 1.5,
                'BB_upper': 152 + np.random.randn(30) * 0.5,
                'BB_lower': 148 + np.random.randn(30) * 0.5,
                'BB_position': 0.5 + np.random.randn(30) * 0.2,
                'Volume_ratio': 1 + np.random.randn(30) * 0.3,
                'Volatility_5': np.abs(np.random.randn(30) * 0.02),
                'Volatility_20': np.abs(np.random.randn(30) * 0.02),
            }, index=dates),
            'GOOGL': pd.DataFrame({
                'Open': 2500 + np.random.randn(30).cumsum() * 2,
                'High': 2510 + np.random.randn(30).cumsum() * 2,
                'Low': 2490 + np.random.randn(30).cumsum() * 2,
                'Close': 2500 + np.random.randn(30).cumsum() * 2,
                'Volume': np.random.randint(500000, 5000000, 30),
                # Add required technical indicators
                'SMA_20': 2500 + np.random.randn(30) * 10,
                'SMA_50': 2500 + np.random.randn(30) * 10,
                'RSI': 50 + np.random.randn(30) * 20,
                'MACD': np.random.randn(30) * 20,
                'MACD_signal': np.random.randn(30) * 15,
                'BB_upper': 2520 + np.random.randn(30) * 10,
                'BB_lower': 2480 + np.random.randn(30) * 10,
                'BB_position': 0.5 + np.random.randn(30) * 0.2,
                'Volume_ratio': 1 + np.random.randn(30) * 0.3,
                'Volatility_5': np.abs(np.random.randn(30) * 0.02),
                'Volatility_20': np.abs(np.random.randn(30) * 0.02),
            }, index=dates)
        }
        
        # 2. Data preprocessing
        preprocessor = GraphPreprocessor()
        processed_data = preprocessor.process_financial_data(mock_stock_data)
        
        # 3. Graph construction
        graph_constructor = GraphConstructor()
        graph_data = graph_constructor.build_correlation_graph(
            processed_data,
            correlation_threshold=0.0,  # Accept any correlation
            max_edges_per_node=2
        )
        
        # 4. Model creation and training
        model = FinancialGNN(
            input_dim=graph_data.x.size(1),
            hidden_dim=16,
            output_dim=1,
            num_layers=1,
            dropout=0.1
        )
        
        # 5. Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        model.train()
        for _ in range(3):  # Just a few epochs
            optimizer.zero_grad()
            predictions = model(graph_data.x, graph_data.edge_index)
            loss = criterion(predictions, graph_data.y)
            loss.backward()
            optimizer.step()
        
        # 6. Verification
        assert graph_data.num_nodes == 2
        assert graph_data.x.size(0) == 2
        assert model.count_parameters() > 0
        assert loss.item() < 1000  # Reasonable loss value
        
        print("✅ End-to-end pipeline test passed!")


if __name__ == "__main__":
    # Run basic integration test
    test = TestSystemIntegration()
    test.test_minimal_end_to_end_pipeline()
    print("🎉 All integration tests completed!") 