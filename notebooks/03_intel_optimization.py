#!/usr/bin/env python3
"""
03_intel_optimization.py
========================

Intel OpenVINO Optimization and Performance Benchmarking

This script demonstrates converting the trained Graph Neural Network
to Intel OpenVINO format for optimal inference performance.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import intel_extension_for_pytorch as ipex
import numpy as np
import time
import matplotlib.pyplot as plt
import psutil
from datetime import datetime
from pathlib import Path

try:
    import openvino as ov
    from openvino.tools import mo
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("⚠️ OpenVINO not available - will show alternative optimization techniques")

from data.data_loader import FinancialDataLoader
from data.preprocessing import GraphPreprocessor
from models.gnn_model import FinancialGNN
from models.intel_optimizer import IntelModelOptimizer
from utils.graph_utils import GraphConstructor

class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_model(self, model, data, model_name, num_runs=100):
        """Benchmark model inference performance"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(data.x, data.edge_index)
        
        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Benchmark inference time
        inference_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(data.x, data.edge_index)
                end_time = time.perf_counter()
                inference_times.append(end_time - start_time)
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate statistics
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        self.results[model_name] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'memory_usage': memory_after - memory_before,
            'throughput': 1.0 / avg_time  # inferences per second
        }
        
        return self.results[model_name]
    
    def compare_models(self, baseline_name, optimized_name):
        """Compare performance between two models"""
        baseline = self.results[baseline_name]
        optimized = self.results[optimized_name]
        
        speedup = baseline['avg_time'] / optimized['avg_time']
        memory_reduction = (baseline['memory_usage'] - optimized['memory_usage']) / baseline['memory_usage'] * 100
        throughput_improvement = (optimized['throughput'] - baseline['throughput']) / baseline['throughput'] * 100
        
        return {
            'speedup': speedup,
            'memory_reduction': memory_reduction,
            'throughput_improvement': throughput_improvement
        }

def create_sample_input(graph_data):
    """Create sample input for model conversion"""
    return (graph_data.x, graph_data.edge_index)

def convert_to_openvino(model, sample_input, output_path):
    """Convert PyTorch model to OpenVINO format"""
    if not OPENVINO_AVAILABLE:
        print("⚠️ OpenVINO not available - skipping conversion")
        return None
    
    try:
        # Trace the model
        print("🔧 Tracing PyTorch model...")
        traced_model = torch.jit.trace(model, sample_input)
        
        # Convert to OpenVINO
        print("⚡ Converting to OpenVINO IR format...")
        ov_model = mo.convert_model(
            traced_model,
            input_shape=[list(sample_input[0].shape), list(sample_input[1].shape)],
            compress_to_fp16=True
        )
        
        # Save the model
        print(f"💾 Saving OpenVINO model to {output_path}")
        ov.save_model(ov_model, output_path)
        
        return ov_model
        
    except Exception as e:
        print(f"❌ OpenVINO conversion failed: {str(e)}")
        return None

def main():
    """Main optimization function"""
    print("🚀 Intel-Optimized Financial GNN - Performance Optimization")
    print("=" * 70)
    
    # Check Intel extensions
    print("🔍 Checking Intel optimizations...")
    print(f"   • Intel Extension for PyTorch: {'✅' if ipex.__version__ else '❌'}")
    print(f"   • OpenVINO Runtime: {'✅' if OPENVINO_AVAILABLE else '❌'}")
    print(f"   • CPU Info: {torch.get_num_threads()} threads available")
    
    # Initialize components
    data_loader = FinancialDataLoader()
    preprocessor = GraphPreprocessor()
    graph_constructor = GraphConstructor()
    optimizer = IntelModelOptimizer()
    benchmark = PerformanceBenchmark()
    
    # Load data for testing
    print("\n📊 Loading test data...")
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    raw_data = data_loader.fetch_stock_data(symbols=symbols, period='1y')
    processed_data = preprocessor.process_financial_data(raw_data)
    graph_data = graph_constructor.build_correlation_graph(processed_data, correlation_threshold=0.3)
    
    print(f"   • Test data: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    print(f"   • Features: {graph_data.num_node_features} per node")
    
    # Load or create model
    print("\n🧠 Setting up models for comparison...")
    
    # 1. Baseline PyTorch model
    print("   1️⃣ Baseline PyTorch model")
    baseline_model = FinancialGNN(
        input_dim=graph_data.num_node_features,
        hidden_dim=128,
        output_dim=1,
        num_layers=3,
        num_heads=8,
        dropout=0.0  # Disable dropout for inference
    )
    baseline_model.eval()
    
    # Try to load pretrained weights
    model_path = Path('notebooks/best_model.pth')
    if model_path.exists():
        baseline_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("     • Loaded pretrained weights")
    else:
        print("     • Using random weights (run 02_model_training.py first for best results)")
    
    # 2. Intel Extension optimized model
    print("   2️⃣ Intel Extension optimized model")
    intel_model = FinancialGNN(
        input_dim=graph_data.num_node_features,
        hidden_dim=128,
        output_dim=1,
        num_layers=3,
        num_heads=8,
        dropout=0.0
    )
    if model_path.exists():
        intel_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # Apply Intel optimizations
    intel_model = ipex.optimize(intel_model)
    intel_model.eval()
    print("     • Intel Extension for PyTorch applied")
    
    # 3. JIT compiled model
    print("   3️⃣ JIT compiled model")
    sample_input = (graph_data.x, graph_data.edge_index)
    jit_model = torch.jit.trace(intel_model, sample_input)
    jit_model.eval()
    print("     • JIT compilation completed")
    
    # Performance benchmarking
    print("\n⏱️ Running performance benchmarks...")
    print("   (This may take a few minutes for accurate measurements)")
    
    # Benchmark each model
    print("\n   🔬 Benchmarking baseline PyTorch model...")
    baseline_results = benchmark.benchmark_model(baseline_model, graph_data, 'baseline', num_runs=50)
    
    print("   🔬 Benchmarking Intel Extension model...")
    intel_results = benchmark.benchmark_model(intel_model, graph_data, 'intel_ext', num_runs=50)
    
    print("   🔬 Benchmarking JIT compiled model...")
    jit_results = benchmark.benchmark_model(jit_model, graph_data, 'jit', num_runs=50)
    
    # OpenVINO optimization (if available)
    if OPENVINO_AVAILABLE:
        print("   🔬 Converting to OpenVINO and benchmarking...")
        try:
            # Convert to OpenVINO
            ov_model_path = "notebooks/financial_gnn_openvino.xml"
            ov_model = convert_to_openvino(intel_model, sample_input, ov_model_path)
            
            if ov_model is not None:
                # Create OpenVINO runtime
                core = ov.Core()
                compiled_model = core.compile_model(ov_model, device_name="CPU")
                
                # Benchmark OpenVINO model
                def openvino_inference(x, edge_index):
                    input_data = {
                        compiled_model.inputs[0].get_any_name(): x.numpy(),
                        compiled_model.inputs[1].get_any_name(): edge_index.numpy()
                    }
                    result = compiled_model(input_data)
                    return torch.from_numpy(result[compiled_model.outputs[0]])
                
                # Custom benchmark for OpenVINO
                inference_times = []
                for _ in range(50):
                    start_time = time.perf_counter()
                    _ = openvino_inference(graph_data.x, graph_data.edge_index)
                    end_time = time.perf_counter()
                    inference_times.append(end_time - start_time)
                
                ov_avg_time = np.mean(inference_times)
                benchmark.results['openvino'] = {
                    'avg_time': ov_avg_time,
                    'throughput': 1.0 / ov_avg_time
                }
                print("     • OpenVINO benchmarking completed")
        except Exception as e:
            print(f"     • OpenVINO optimization failed: {str(e)}")
    
    # Results analysis
    print("\n📊 Performance Analysis Results")
    print("=" * 50)
    
    # Display individual results
    for name, results in benchmark.results.items():
        print(f"\n{name.upper()} MODEL:")
        print(f"   • Average inference time: {results['avg_time']*1000:.2f} ms")
        print(f"   • Throughput: {results['throughput']:.1f} inferences/sec")
        if 'memory_usage' in results:
            print(f"   • Memory usage: {results['memory_usage']:.1f} MB")
    
    # Comparative analysis
    print(f"\n🚀 OPTIMIZATION IMPROVEMENTS:")
    
    if 'intel_ext' in benchmark.results:
        intel_comparison = benchmark.compare_models('baseline', 'intel_ext')
        print(f"   Intel Extension vs Baseline:")
        print(f"     • Speedup: {intel_comparison['speedup']:.2f}x")
        print(f"     • Throughput improvement: {intel_comparison['throughput_improvement']:.1f}%")
    
    if 'jit' in benchmark.results:
        jit_comparison = benchmark.compare_models('baseline', 'jit')
        print(f"   JIT Compiled vs Baseline:")
        print(f"     • Speedup: {jit_comparison['speedup']:.2f}x")
        print(f"     • Throughput improvement: {jit_comparison['throughput_improvement']:.1f}%")
    
    if 'openvino' in benchmark.results:
        ov_speedup = baseline_results['avg_time'] / benchmark.results['openvino']['avg_time']
        ov_throughput_imp = (benchmark.results['openvino']['throughput'] - baseline_results['throughput']) / baseline_results['throughput'] * 100
        print(f"   OpenVINO vs Baseline:")
        print(f"     • Speedup: {ov_speedup:.2f}x")
        print(f"     • Throughput improvement: {ov_throughput_imp:.1f}%")
    
    # Create performance visualization
    print("\n🎨 Creating performance visualizations...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Inference time comparison
    models = list(benchmark.results.keys())
    times = [benchmark.results[model]['avg_time'] * 1000 for model in models]  # Convert to ms
    
    bars1 = ax1.bar(models, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
    ax1.set_title('Inference Time Comparison')
    ax1.set_ylabel('Average Inference Time (ms)')
    ax1.set_xlabel('Model Type')
    
    # Add value labels on bars
    for bar, time_val in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{time_val:.2f}ms', ha='center', va='bottom')
    
    # Throughput comparison
    throughputs = [benchmark.results[model]['throughput'] for model in models]
    bars2 = ax2.bar(models, throughputs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
    ax2.set_title('Throughput Comparison')
    ax2.set_ylabel('Inferences per Second')
    ax2.set_xlabel('Model Type')
    
    # Add value labels on bars
    for bar, throughput in zip(bars2, throughputs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{throughput:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('notebooks/optimization_results.png', dpi=300, bbox_inches='tight')
    print("   • Performance plots saved to: notebooks/optimization_results.png")
    
    # Summary recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    
    best_model = min(benchmark.results.keys(), key=lambda x: benchmark.results[x]['avg_time'])
    best_speedup = baseline_results['avg_time'] / benchmark.results[best_model]['avg_time']
    
    print(f"   • Best performing model: {best_model.upper()}")
    print(f"   • Maximum speedup achieved: {best_speedup:.2f}x")
    print(f"   • Recommended for production: {best_model}")
    
    if OPENVINO_AVAILABLE and 'openvino' in benchmark.results:
        print(f"   • OpenVINO model saved to: notebooks/financial_gnn_openvino.xml")
    else:
        print(f"   • Install OpenVINO for additional 2-4x speedup potential")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Deploy {best_model} model for production inference")
    print(f"   2. Use Intel DevCloud for additional optimization testing")
    print(f"   3. Consider model quantization for edge deployment")
    print(f"   4. Implement batch processing for higher throughput")
    
    return benchmark.results

if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Intel optimization analysis completed successfully!")
        
        # Find best performance
        best_model = min(results.keys(), key=lambda x: results[x]['avg_time'])
        best_time = results[best_model]['avg_time'] * 1000
        baseline_time = results['baseline']['avg_time'] * 1000
        improvement = baseline_time / best_time
        
        print(f"🏆 Best model: {best_model} ({best_time:.2f}ms, {improvement:.2f}x faster)")
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {str(e)}")
        print("💡 Make sure model training completed successfully first") 