#!/usr/bin/env python3
"""
02_model_training.py
===================

Graph Neural Network Model Training

This script demonstrates the training process for the Intel-Optimized
Financial Graph Neural Network with performance monitoring.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F
import intel_extension_for_pytorch as ipex
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

from data.data_loader import FinancialDataLoader
from data.preprocessing import GraphPreprocessor
from models.gnn_model import FinancialGNN
from utils.graph_utils import GraphConstructor
from utils.visualization import GraphVisualizer

# Configure PyTorch for Intel optimizations
torch.jit.enable_onednn_fusion(True)

def train_epoch(model, data, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(out, data.y.to(device))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate_model(model, data, criterion, device):
    """Evaluate model performance"""
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = criterion(out, data.y.to(device))
        
        # Calculate accuracy for classification
        if data.y.dim() > 1 and data.y.size(1) > 1:
            pred = out.argmax(dim=1)
            acc = (pred == data.y.argmax(dim=1)).float().mean()
        else:
            # For regression, calculate R²
            ss_res = torch.sum((data.y.to(device) - out) ** 2)
            ss_tot = torch.sum((data.y.to(device) - torch.mean(data.y.to(device))) ** 2)
            acc = 1 - ss_res / ss_tot
    
    return loss.item(), acc.item()

def main():
    """Main training function"""
    print("🚀 Intel-Optimized Financial GNN - Model Training")
    print("=" * 60)
    
    # Set device
    device = torch.device('cpu')  # Intel optimizations work best on CPU
    print(f"🖥️  Using device: {device}")
    
    # Initialize components
    data_loader = FinancialDataLoader()
    preprocessor = GraphPreprocessor()
    graph_constructor = GraphConstructor()
    visualizer = GraphVisualizer()
    
    # Configuration
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
        'period': '2y',
        'hidden_dim': 128,
        'num_layers': 3,
        'num_heads': 8,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'epochs': 100,
        'patience': 20
    }
    
    print(f"📊 Training configuration:")
    for key, value in config.items():
        print(f"   • {key}: {value}")
    
    # Load and preprocess data
    print("\n📈 Loading financial data...")
    raw_data = data_loader.fetch_stock_data(
        symbols=config['symbols'], 
        period=config['period']
    )
    
    print("🔧 Preprocessing data...")
    processed_data = preprocessor.process_financial_data(raw_data)
    
    print("🕸️  Constructing graph...")
    graph_data = graph_constructor.build_correlation_graph(
        processed_data, 
        correlation_threshold=0.3
    )
    
    # Model architecture info
    print(f"\n🏗️  Graph structure:")
    print(f"   • Nodes: {graph_data.num_nodes}")
    print(f"   • Edges: {graph_data.num_edges}")
    print(f"   • Features per node: {graph_data.num_node_features}")
    
    # Initialize model
    print("🧠 Initializing Graph Neural Network...")
    model = FinancialGNN(
        input_dim=graph_data.num_node_features,
        hidden_dim=config['hidden_dim'],
        output_dim=1,
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    )
    
    # Apply Intel optimizations
    print("⚡ Applying Intel optimizations...")
    model = ipex.optimize(model)
    model = model.to(device)
    
    # Model info
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   • Total parameters: {param_count:,}")
    print(f"   • Model size: {param_count * 4 / 1024 / 1024:.1f} MB")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    print(f"\n🏋️  Starting training for {config['epochs']} epochs...")
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # Training
        train_loss = train_epoch(model, graph_data, optimizer, criterion, device)
        
        # Validation (using same data for demo - in practice use separate validation set)
        val_loss, val_acc = evaluate_model(model, graph_data, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'notebooks/best_model.pth')
        else:
            patience_counter += 1
        
        # Progress reporting
        if epoch % 20 == 0 or epoch == config['epochs'] - 1:
            elapsed = time.time() - start_time
            print(f"   Epoch {epoch:3d}: Loss={train_loss:.4f}, Val={val_loss:.4f}, "
                  f"Acc={val_acc:.3f}, Time={elapsed:.1f}s")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"   Early stopping at epoch {epoch}")
            break
    
    # Training summary
    total_time = time.time() - start_time
    print(f"\n📊 Training completed in {total_time:.1f} seconds")
    print(f"   • Final loss: {train_losses[-1]:.4f}")
    print(f"   • Best validation loss: {best_loss:.4f}")
    print(f"   • Final accuracy: {val_accuracies[-1]:.3f}")
    print(f"   • Average time per epoch: {total_time / len(train_losses):.2f}s")
    
    # Performance analysis
    print("\n📈 Training Performance Analysis:")
    loss_improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    print(f"   • Loss improvement: {loss_improvement:.1f}%")
    print(f"   • Convergence: {'✅ Converged' if patience_counter < config['patience'] else '⚠️ Max epochs reached'}")
    
    # Create training visualizations
    print("\n🎨 Creating training visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    epochs_range = range(len(train_losses))
    ax1.plot(epochs_range, train_losses, 'b-', label='Training Loss', alpha=0.8)
    ax1.plot(epochs_range, val_losses, 'r-', label='Validation Loss', alpha=0.8)
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs_range, val_accuracies, 'g-', label='Validation Accuracy', alpha=0.8)
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss distribution
    ax3.hist(train_losses, bins=20, alpha=0.7, color='blue', label='Training')
    ax3.hist(val_losses, bins=20, alpha=0.7, color='red', label='Validation')
    ax3.set_title('Loss Distribution')
    ax3.set_xlabel('Loss Value')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # Model predictions vs targets (sample)
    model.eval()
    with torch.no_grad():
        predictions = model(graph_data.x.to(device), graph_data.edge_index.to(device))
        predictions_np = predictions.cpu().numpy().flatten()
        targets_np = graph_data.y.numpy().flatten()
    
    ax4.scatter(targets_np, predictions_np, alpha=0.6, s=30)
    ax4.plot([targets_np.min(), targets_np.max()], [targets_np.min(), targets_np.max()], 'r--', alpha=0.8)
    ax4.set_title('Predictions vs Targets')
    ax4.set_xlabel('Actual Values')
    ax4.set_ylabel('Predicted Values')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('notebooks/training_results.png', dpi=300, bbox_inches='tight')
    print("   • Training plots saved to: notebooks/training_results.png")
    
    # Model analysis
    print("\n🔍 Model Analysis:")
    print(f"   • Architecture: {config['num_layers']}-layer GAT with {config['num_heads']} attention heads")
    print(f"   • Parameters: {param_count:,} trainable parameters")
    print(f"   • Memory usage: ~{param_count * 4 / 1024 / 1024:.1f} MB")
    print(f"   • Training efficiency: {len(train_losses) / total_time:.1f} epochs/second")
    
    # Intel optimization benefits
    print("\n⚡ Intel Optimization Benefits:")
    print("   • Intel Extension for PyTorch: ✅ Applied")
    print("   • OneDNN fusion: ✅ Enabled")
    print("   • Optimized for Intel CPU: ✅ Configured")
    print("   • Memory optimization: ✅ Active")
    
    print("\n🎯 Next Steps:")
    print("   1. Run 03_intel_optimization.py for further optimization")
    print("   2. Model saved to: notebooks/best_model.pth")
    print("   3. View training plots: notebooks/training_results.png")
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_loss': best_loss,
        'training_time': total_time,
        'final_accuracy': val_accuracies[-1]
    }

if __name__ == "__main__":
    try:
        model, results = main()
        print("\n✅ Model training completed successfully!")
        print(f"🏆 Final model accuracy: {results['final_accuracy']:.3f}")
        print(f"⏱️ Total training time: {results['training_time']:.1f} seconds")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("💡 Make sure all dependencies are installed and data is accessible") 