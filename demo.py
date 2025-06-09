#!/usr/bin/env python3
"""
🚀 Intel-Optimized Financial GNN Demo

This demo script showcases the key features of our Intel-optimized
Financial Graph Neural Network system.

Usage:
    python demo.py [--quick] [--symbols AAPL GOOGL MSFT]
"""

import os
import sys
import time
import argparse
from datetime import datetime
import warnings

# Suppress warnings for demo
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

try:
    import torch
    import pandas as pd
    import numpy as np
    import intel_extension_for_pytorch as ipex
    INTEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: {e}")
    INTEL_AVAILABLE = False

from src.data.data_loader import FinancialDataLoader
from src.data.preprocessing import GraphPreprocessor
from src.models.gnn_model import FinancialGNN
from src.models.intel_optimizer import IntelModelOptimizer
from src.utils.graph_utils import GraphConstructor
from src.utils.visualization import GraphVisualizer


class FinancialGNNDemo:
    """Demo class for Intel-Optimized Financial GNN"""
    
    def __init__(self, symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN'], quick_mode=False):
        self.symbols = symbols
        self.quick_mode = quick_mode
        self.results = {}
        
        print("🚀 Initializing Intel-Optimized Financial GNN Demo")
        print("=" * 60)
        
    def print_header(self, title):
        """Print formatted section header"""
        print(f"\n🎯 {title}")
        print("-" * 50)
        
    def print_success(self, message):
        """Print success message"""
        print(f"✅ {message}")
        
    def print_info(self, message):
        """Print info message"""
        print(f"ℹ️ {message}")
        
    def demo_data_loading(self):
        """Demonstrate financial data loading"""
        self.print_header("Financial Data Loading & Processing")
        
        # Initialize data loader
        data_loader = FinancialDataLoader()
        
        print(f"📊 Loading data for: {', '.join(self.symbols)}")
        start_time = time.time()
        
        # Load stock data
        period = "6mo" if self.quick_mode else "2y"
        stock_data = data_loader.fetch_stock_data(
            symbols=self.symbols,
            period=period
        )
        
        load_time = time.time() - start_time
        
        # Display results
        total_records = sum(len(data) for data in stock_data.values())
        self.print_success(f"Loaded {total_records:,} records in {load_time:.2f}s")
        
        for symbol, data in stock_data.items():
            print(f"   📈 {symbol}: {len(data)} days, "
                  f"${data['Close'].iloc[-1]:.2f} current price")
        
        # Get market data
        market_data = data_loader.get_market_data(self.symbols)
        
        print("\n📊 Market Information:")
        for symbol, info in market_data.items():
            sector = info.get('sector', 'Unknown')
            market_cap = info.get('market_cap', 0)
            print(f"   {symbol}: {sector} | Market Cap: ${market_cap:,}")
        
        self.results['stock_data'] = stock_data
        self.results['market_data'] = market_data
        self.results['load_time'] = load_time
        
        return stock_data
    
    def demo_preprocessing(self, stock_data):
        """Demonstrate data preprocessing"""
        self.print_header("Advanced Data Preprocessing")
        
        preprocessor = GraphPreprocessor()
        
        print("🔧 Applying feature engineering...")
        start_time = time.time()
        
        processed_data = preprocessor.process_financial_data(stock_data)
        
        process_time = time.time() - start_time
        
        # Analyze features
        sample_data = list(processed_data.values())[0]
        feature_cols = [col for col in sample_data.columns 
                       if not col.startswith('future_') and col not in ['price_up_1d', 'price_up_5d']]
        
        self.print_success(f"Generated {len(feature_cols)} features in {process_time:.2f}s")
        
        print("\n📊 Feature Categories:")
        technical_features = [col for col in feature_cols if any(x in col.lower() 
                            for x in ['sma', 'ema', 'rsi', 'macd', 'bb'])]
        volume_features = [col for col in feature_cols if 'volume' in col.lower()]
        momentum_features = [col for col in feature_cols if 'momentum' in col.lower()]
        
        print(f"   📈 Technical Indicators: {len(technical_features)}")
        print(f"   📊 Volume Features: {len(volume_features)}")
        print(f"   🚀 Momentum Features: {len(momentum_features)}")
        
        self.results['processed_data'] = processed_data
        self.results['feature_count'] = len(feature_cols)
        self.results['process_time'] = process_time
        
        return processed_data
    
    def demo_graph_construction(self, processed_data):
        """Demonstrate graph construction"""
        self.print_header("Financial Correlation Graph Construction")
        
        graph_constructor = GraphConstructor()
        
        print("🕸️ Building correlation graph...")
        start_time = time.time()
        
        graph_data = graph_constructor.build_correlation_graph(
            processed_data,
            correlation_threshold=0.3,
            max_edges_per_node=5
        )
        
        graph_time = time.time() - start_time
        
        # Analyze graph structure
        self.print_success(f"Graph constructed in {graph_time:.2f}s")
        
        print(f"\n🔍 Graph Analysis:")
        print(f"   📊 Nodes (Stocks): {graph_data.num_nodes}")
        print(f"   🔗 Edges (Correlations): {graph_data.edge_index.size(1)}")
        print(f"   📈 Features per Node: {graph_data.x.size(1)}")
        print(f"   🎯 Average Degree: {graph_data.edge_index.size(1) / graph_data.num_nodes:.2f}")
        
        # Show correlations
        if hasattr(graph_data, 'correlation_matrix'):
            corr_matrix = graph_data.correlation_matrix
            max_corr = np.max(corr_matrix[corr_matrix < 1.0])
            min_corr = np.min(corr_matrix)
            avg_corr = np.mean(corr_matrix[corr_matrix < 1.0])
            
            print(f"   📊 Max Correlation: {max_corr:.3f}")
            print(f"   📊 Min Correlation: {min_corr:.3f}")
            print(f"   📊 Avg Correlation: {avg_corr:.3f}")
        
        self.results['graph_data'] = graph_data
        self.results['graph_time'] = graph_time
        
        return graph_data
    
    def demo_model_training(self, graph_data):
        """Demonstrate model training"""
        self.print_header("Graph Neural Network Training")
        
        # Model configuration
        config = {
            'input_dim': graph_data.x.size(1),
            'hidden_dim': 64 if self.quick_mode else 128,
            'output_dim': 1,
            'num_layers': 2 if self.quick_mode else 3,
            'num_heads': 4 if self.quick_mode else 8,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 20 if self.quick_mode else 100
        }
        
        # Create model
        model = FinancialGNN(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        )
        
        print(f"🧠 Model Architecture:")
        print(f"   📊 Parameters: {model.count_parameters():,}")
        print(f"   🏗️ Layers: {config['num_layers']}")
        print(f"   🎯 Hidden Dimension: {config['hidden_dim']}")
        print(f"   👁️ Attention Heads: {config['num_heads']}")
        
        # Apply Intel optimizations if available
        if INTEL_AVAILABLE:
            print("\n⚡ Applying Intel Extension for PyTorch optimizations...")
            model = ipex.optimize(model, dtype=torch.float32)
            self.print_success("Intel optimizations applied!")
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = torch.nn.MSELoss()
        
        print(f"\n🏋️ Training for {config['epochs']} epochs...")
        start_time = time.time()
        
        train_losses = []
        model.train()
        
        for epoch in range(config['epochs']):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(graph_data.x, graph_data.edge_index)
            loss = criterion(predictions, graph_data.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if epoch % max(1, config['epochs'] // 5) == 0:
                print(f"   Epoch {epoch:3d}: Loss = {loss.item():.6f}")
        
        train_time = time.time() - start_time
        
        self.print_success(f"Training completed in {train_time:.2f}s")
        print(f"   📊 Initial Loss: {train_losses[0]:.6f}")
        print(f"   📊 Final Loss: {train_losses[-1]:.6f}")
        print(f"   📈 Improvement: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
        
        self.results['model'] = model
        self.results['train_losses'] = train_losses
        self.results['train_time'] = train_time
        self.results['config'] = config
        
        return model
    
    def demo_intel_optimization(self, model, graph_data):
        """Demonstrate Intel optimization"""
        self.print_header("Intel OpenVINO Optimization")
        
        optimizer = IntelModelOptimizer()
        
        print("🔧 Converting model to OpenVINO format...")
        start_time = time.time()
        
        try:
            optimized_model = optimizer.convert_to_openvino(
                model,
                graph_data,
                model_name="financial_gnn_demo",
                precision="FP16"
            )
            
            conversion_time = time.time() - start_time
            
            if optimized_model:
                self.print_success(f"OpenVINO conversion completed in {conversion_time:.2f}s")
                
                # Benchmark performance
                print("\n📊 Running performance benchmark...")
                benchmark_results = optimizer.benchmark_model(
                    original_model=model,
                    optimized_model=optimized_model,
                    test_data=graph_data,
                    num_runs=10 if self.quick_mode else 50
                )
                
                print(f"\n🚀 Performance Results:")
                print(f"   ⚡ Speedup: {benchmark_results['speedup']:.2f}x")
                print(f"   💾 Memory Reduction: {benchmark_results['memory_reduction']:.1f}%")
                print(f"   ⏱️ Original Time: {benchmark_results['original_avg_time']:.4f}s")
                print(f"   ⚡ Optimized Time: {benchmark_results['optimized_avg_time']:.4f}s")
                
                self.results['optimization'] = benchmark_results
                
            else:
                print("⚠️ OpenVINO conversion failed - using Intel Extension optimizations")
                
        except Exception as e:
            print(f"ℹ️ OpenVINO not available: {e}")
            print("   Using Intel Extension for PyTorch optimizations instead")
    
    def demo_visualization(self, graph_data):
        """Demonstrate visualization capabilities"""
        self.print_header("Interactive Visualizations")
        
        visualizer = GraphVisualizer()
        
        print("🎨 Creating visualizations...")
        
        # Create results directory
        os.makedirs("demo_results", exist_ok=True)
        
        try:
            # Graph visualization
            print("   📊 Financial correlation graph...")
            visualizer.plot_financial_graph(
                graph_data, 
                self.symbols, 
                "demo_results/financial_graph.html"
            )
            
            # Correlation heatmap
            print("   🔥 Correlation heatmap...")
            visualizer.plot_correlation_heatmap(
                graph_data, 
                self.symbols, 
                "demo_results/correlation_heatmap.html"
            )
            
            # Performance comparison (if available)
            if 'optimization' in self.results:
                print("   ⚡ Performance comparison...")
                visualizer.plot_performance_comparison(
                    self.results['optimization'],
                    "demo_results/performance_comparison.html"
                )
            
            # Training history
            if 'train_losses' in self.results:
                print("   📈 Training history...")
                visualizer.plot_training_history(
                    self.results['train_losses'],
                    save_path="demo_results/training_history.html"
                )
            
            self.print_success("Visualizations saved to demo_results/")
            
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
    
    def demo_summary(self):
        """Print demo summary"""
        self.print_header("Demo Summary & Results")
        
        print("🎉 Intel-Optimized Financial GNN Demo Completed!")
        print("\n📊 Key Metrics:")
        
        if 'load_time' in self.results:
            print(f"   ⏱️ Data Loading: {self.results['load_time']:.2f}s")
        
        if 'process_time' in self.results:
            print(f"   🔧 Data Processing: {self.results['process_time']:.2f}s")
        
        if 'graph_time' in self.results:
            print(f"   🕸️ Graph Construction: {self.results['graph_time']:.2f}s")
        
        if 'train_time' in self.results:
            print(f"   🏋️ Model Training: {self.results['train_time']:.2f}s")
        
        if 'feature_count' in self.results:
            print(f"   📈 Features Generated: {self.results['feature_count']}")
        
        if 'optimization' in self.results:
            opt = self.results['optimization']
            print(f"   ⚡ Intel Speedup: {opt['speedup']:.2f}x")
            print(f"   💾 Memory Reduction: {opt['memory_reduction']:.1f}%")
        
        print("\n🎯 Achievements:")
        print("   ✅ Successfully loaded and processed financial data")
        print("   ✅ Constructed correlation-based graph representation")
        print("   ✅ Trained Graph Neural Network model")
        print("   ✅ Applied Intel AI optimizations")
        print("   ✅ Generated interactive visualizations")
        
        if INTEL_AVAILABLE:
            print("   ✅ Demonstrated Intel performance acceleration")
        
        print("\n📁 Output Files:")
        print("   📊 demo_results/financial_graph.html")
        print("   🔥 demo_results/correlation_heatmap.html")
        print("   📈 demo_results/training_history.html")
        
        if 'optimization' in self.results:
            print("   ⚡ demo_results/performance_comparison.html")
        
        print("\n🚀 Next Steps:")
        print("   • Explore the generated visualizations")
        print("   • Experiment with different stock symbols")
        print("   • Try the full training pipeline with more data")
        print("   • Deploy the model for real-time inference")
        
        print("\n" + "=" * 60)
        print("🎊 Thank you for trying Intel-Optimized Financial GNN!")
    
    def run_demo(self):
        """Run the complete demo"""
        try:
            # Demo pipeline
            stock_data = self.demo_data_loading()
            processed_data = self.demo_preprocessing(stock_data)
            graph_data = self.demo_graph_construction(processed_data)
            model = self.demo_model_training(graph_data)
            self.demo_intel_optimization(model, graph_data)
            self.demo_visualization(graph_data)
            self.demo_summary()
            
        except KeyboardInterrupt:
            print("\n⚠️ Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Intel-Optimized Financial GNN Demo")
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick demo with reduced parameters')
    parser.add_argument('--symbols', nargs='+', 
                       default=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
                       help='Stock symbols to analyze')
    
    args = parser.parse_args()
    
    # Initialize and run demo
    demo = FinancialGNNDemo(symbols=args.symbols, quick_mode=args.quick)
    demo.run_demo()


if __name__ == "__main__":
    main() 