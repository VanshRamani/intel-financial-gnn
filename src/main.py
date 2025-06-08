#!/usr/bin/env python3
"""
Main entry point for Intel-Optimized Financial Graph Neural Network

This script orchestrates the entire pipeline:
1. Data collection and preprocessing
2. Graph construction
3. Model training with Intel optimizations
4. Inference and visualization
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import warnings

# Intel optimizations
import intel_extension_for_pytorch as ipex
import torch

# Local imports
from data.data_loader import FinancialDataLoader
from data.preprocessing import GraphPreprocessor
from models.gnn_model import FinancialGNN
from models.intel_optimizer import IntelModelOptimizer
from utils.visualization import GraphVisualizer
from utils.graph_utils import GraphConstructor

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_gnn.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class IntelFinancialGNN:
    """Main class for Intel-optimized Financial Graph Neural Network"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.data_loader = FinancialDataLoader()
        self.preprocessor = GraphPreprocessor()
        self.graph_constructor = GraphConstructor()
        self.model = None
        self.optimizer = IntelModelOptimizer()
        self.visualizer = GraphVisualizer()
        
        # Enable Intel optimizations
        torch.jit.enable_onednn_fusion(True)
        logger.info("Intel optimizations enabled")
    
    def _default_config(self):
        """Default configuration for the model"""
        return {
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'],
            'period': '2y',
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'device': 'cpu'  # Intel optimizations work best on CPU
        }
    
    def load_and_preprocess_data(self):
        """Load financial data and create graph structure"""
        logger.info("Loading financial data...")
        
        # Load stock data
        raw_data = self.data_loader.fetch_stock_data(
            symbols=self.config['symbols'],
            period=self.config['period']
        )
        
        # Preprocess data
        logger.info("Preprocessing data...")
        processed_data = self.preprocessor.process_financial_data(raw_data)
        
        # Construct graph
        logger.info("Constructing financial correlation graph...")
        self.graph_data = self.graph_constructor.build_correlation_graph(
            processed_data,
            correlation_threshold=0.3
        )
        
        logger.info(f"Graph created with {self.graph_data.num_nodes} nodes and {self.graph_data.num_edges} edges")
        return self.graph_data
    
    def build_model(self):
        """Build and optimize the GNN model"""
        logger.info("Building Graph Neural Network...")
        
        self.model = FinancialGNN(
            input_dim=self.graph_data.num_node_features,
            hidden_dim=self.config['hidden_dim'],
            output_dim=1,  # Predict price movement
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        # Apply Intel optimizations
        logger.info("Applying Intel optimizations...")
        self.model = ipex.optimize(self.model)
        
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def train_model(self):
        """Train the model with Intel optimizations"""
        logger.info("Starting model training...")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        criterion = torch.nn.MSELoss()
        
        train_losses = []
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(self.graph_data.x, self.graph_data.edge_index)
            loss = criterion(predictions, self.graph_data.y)
            
            # Backward pass with Intel optimizations
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        logger.info("Training completed!")
        return train_losses
    
    def optimize_for_inference(self):
        """Optimize model for deployment using Intel OpenVINO"""
        logger.info("Optimizing model for inference with Intel OpenVINO...")
        
        # Convert to OpenVINO format
        optimized_model = self.optimizer.convert_to_openvino(
            self.model,
            self.graph_data,
            model_name="financial_gnn"
        )
        
        # Benchmark performance
        performance_results = self.optimizer.benchmark_model(
            original_model=self.model,
            optimized_model=optimized_model,
            test_data=self.graph_data
        )
        
        logger.info("Intel optimization completed!")
        logger.info(f"Speedup: {performance_results['speedup']:.2f}x")
        logger.info(f"Memory reduction: {performance_results['memory_reduction']:.1f}%")
        
        return optimized_model, performance_results
    
    def visualize_results(self):
        """Create beautiful visualizations"""
        logger.info("Creating visualizations...")
        
        # Graph structure visualization
        self.visualizer.plot_financial_graph(
            self.graph_data,
            symbols=self.config['symbols'],
            save_path="results/financial_graph.html"
        )
        
        # Correlation heatmap
        self.visualizer.plot_correlation_heatmap(
            self.graph_data,
            symbols=self.config['symbols'],
            save_path="results/correlation_heatmap.png"
        )
        
        # Performance comparison
        if hasattr(self, 'performance_results'):
            self.visualizer.plot_performance_comparison(
                self.performance_results,
                save_path="results/performance_comparison.png"
            )
        
        logger.info("Visualizations saved to results/ directory")
    
    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        try:
            logger.info("🚀 Starting Intel-Optimized Financial GNN Pipeline")
            
            # Create results directory
            os.makedirs("results", exist_ok=True)
            
            # Step 1: Data loading and preprocessing
            self.load_and_preprocess_data()
            
            # Step 2: Model building
            self.build_model()
            
            # Step 3: Training
            train_losses = self.train_model()
            
            # Step 4: Intel optimization
            optimized_model, self.performance_results = self.optimize_for_inference()
            
            # Step 5: Visualization
            self.visualize_results()
            
            logger.info("✅ Pipeline completed successfully!")
            logger.info("📊 Check the results/ directory for outputs")
            
            return {
                'model': self.model,
                'optimized_model': optimized_model,
                'performance': self.performance_results,
                'losses': train_losses
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Intel-Optimized Financial Graph Neural Network")
    parser.add_argument("--symbols", nargs="+", default=['AAPL', 'GOOGL', 'MSFT', 'AMZN'], 
                       help="Stock symbols to analyze")
    parser.add_argument("--period", default="2y", help="Data period (1y, 2y, 5y)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension size")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'symbols': args.symbols,
        'period': args.period,
        'epochs': args.epochs,
        'hidden_dim': args.hidden_dim,
        'num_layers': 3,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'device': 'cpu'
    }
    
    # Run pipeline
    gnn_system = IntelFinancialGNN(config)
    results = gnn_system.run_full_pipeline()
    
    print("\n🎉 Intel-Optimized Financial GNN completed successfully!")
    print(f"📈 Final training loss: {results['losses'][-1]:.4f}")
    print(f"⚡ Intel speedup: {results['performance']['speedup']:.2f}x")
    print(f"💾 Memory reduction: {results['performance']['memory_reduction']:.1f}%")


if __name__ == "__main__":
    main() 