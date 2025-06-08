"""
Visualization Utilities for Financial Graph Analysis

Creates beautiful and interactive visualizations of financial graphs,
correlations, performance metrics, and model results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class GraphVisualizer:
    """
    Comprehensive visualization utilities for financial graph analysis
    """
    
    def __init__(self, figsize: tuple = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = px.colors.qualitative.Set3
        
    def plot_financial_graph(self, 
                            graph_data: Any,
                            symbols: List[str],
                            save_path: Optional[str] = None,
                            interactive: bool = True) -> None:
        """
        Create interactive visualization of the financial correlation graph
        
        Args:
            graph_data: PyTorch Geometric graph data
            symbols: List of stock symbols
            save_path: Path to save the visualization
            interactive: Whether to create interactive plot
        """
        logger.info("Creating financial graph visualization...")
        
        # Convert to NetworkX
        G = nx.Graph()
        
        # Add nodes
        for i, symbol in enumerate(symbols):
            G.add_node(i, symbol=symbol, size=10)
        
        # Add edges
        edge_index = graph_data.edge_index.numpy()
        edge_weights = graph_data.edge_attr.numpy().flatten()
        
        for i in range(edge_index.shape[1]):
            source, target = edge_index[0, i], edge_index[1, i]
            weight = edge_weights[i]
            G.add_edge(source, target, weight=weight)
        
        if interactive:
            self._create_interactive_graph(G, symbols, save_path)
        else:
            self._create_static_graph(G, symbols, save_path)
    
    def _create_interactive_graph(self, G: nx.Graph, symbols: List[str], save_path: Optional[str]):
        """Create interactive graph using Plotly"""
        
        # Use spring layout for positioning
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Extract edge coordinates
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2].get('weight', 1.0))
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightblue'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Extract node coordinates and info
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info
            symbol = symbols[node] if node < len(symbols) else f"Node_{node}"
            adjacencies = list(G.neighbors(node))
            node_info = f"Symbol: {symbol}<br>Connections: {len(adjacencies)}"
            node_text.append(node_info)
            
            # Color by degree (number of connections)
            node_colors.append(len(adjacencies))
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[symbols[i] if i < len(symbols) else f"N{i}" for i in range(len(node_x))],
            textposition="middle center",
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=True,
                color=node_colors,
                size=30,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    title="Node Connections"
                ),
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Intel-Optimized Financial Correlation Graph',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Nodes represent stocks, edges represent correlations",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor="left", yanchor="bottom",
                        font=dict(color="gray", size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive graph saved to {save_path}")
        else:
            fig.show()
    
    def _create_static_graph(self, G: nx.Graph, symbols: List[str], save_path: Optional[str]):
        """Create static graph using matplotlib"""
        
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw edges
        edges = G.edges(data=True)
        edge_weights = [edge[2].get('weight', 1.0) for edge in edges]
        
        nx.draw_networkx_edges(
            G, pos,
            width=[w*3 for w in edge_weights],
            alpha=0.6,
            edge_color='lightblue'
        )
        
        # Draw nodes
        node_colors = [len(list(G.neighbors(node))) for node in G.nodes()]
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=1000,
            cmap='viridis',
            alpha=0.8
        )
        
        # Draw labels
        labels = {i: symbols[i] if i < len(symbols) else f"N{i}" for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        plt.title('Intel-Optimized Financial Correlation Graph', fontsize=16, fontweight='bold')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"Static graph saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_correlation_heatmap(self, 
                               graph_data: Any,
                               symbols: List[str],
                               save_path: Optional[str] = None) -> None:
        """
        Create correlation heatmap visualization
        
        Args:
            graph_data: Graph data containing correlation matrix
            symbols: List of stock symbols
            save_path: Path to save the visualization
        """
        logger.info("Creating correlation heatmap...")
        
        # Get correlation matrix
        if hasattr(graph_data, 'correlation_matrix'):
            corr_matrix = graph_data.correlation_matrix
        else:
            # Fallback: calculate from edge weights
            n = len(symbols)
            corr_matrix = np.eye(n)
            
            edge_index = graph_data.edge_index.numpy()
            edge_weights = graph_data.edge_attr.numpy().flatten()
            
            for i in range(edge_index.shape[1]):
                row, col = edge_index[0, i], edge_index[1, i]
                corr_matrix[row, col] = edge_weights[i]
                corr_matrix[col, row] = edge_weights[i]  # Symmetric
        
        # Create DataFrame
        corr_df = pd.DataFrame(corr_matrix, index=symbols, columns=symbols)
        
        # Create interactive heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_df.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            )
        )
        
        fig.update_layout(
            title='Stock Correlation Heatmap',
            xaxis_title='Stocks',
            yaxis_title='Stocks',
            width=600,
            height=600
        )
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Correlation heatmap saved to {save_path}")
        else:
            fig.show()
    
    def plot_performance_comparison(self, 
                                  performance_results: Dict[str, float],
                                  save_path: Optional[str] = None) -> None:
        """
        Create performance comparison charts
        
        Args:
            performance_results: Results from model benchmarking
            save_path: Path to save the visualization
        """
        logger.info("Creating performance comparison visualization...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Inference Time Comparison', 'Speedup Factor', 
                          'Memory Usage', 'Performance Summary'),
            specs=[[{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Inference time comparison
        categories = ['Original Model', 'Intel Optimized']
        times = [performance_results['original_avg_time'], performance_results['optimized_avg_time']]
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=times,
                name='Inference Time (s)',
                marker_color=['lightcoral', 'lightblue']
            ),
            row=1, col=1
        )
        
        # Speedup indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=performance_results['speedup'],
                title={'text': "Speedup Factor"},
                gauge={
                    'axis': {'range': [0, 5]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgray"},
                        {'range': [1, 2], 'color': "yellow"},
                        {'range': [2, 5], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 2
                    }
                }
            ),
            row=1, col=2
        )
        
        # Memory usage
        memory_categories = ['Memory Reduction %']
        memory_values = [performance_results['memory_reduction']]
        
        fig.add_trace(
            go.Bar(
                x=memory_categories,
                y=memory_values,
                name='Memory Reduction',
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
        
        # Performance summary table
        summary_data = [
            ['Metric', 'Original', 'Optimized', 'Improvement'],
            ['Inference Time (s)', 
             f"{performance_results['original_avg_time']:.4f}",
             f"{performance_results['optimized_avg_time']:.4f}",
             f"{performance_results['speedup']:.2f}x"],
            ['Memory Usage', '100%', 
             f"{100 - performance_results['memory_reduction']:.1f}%",
             f"{performance_results['memory_reduction']:.1f}% reduction"],
            ['Parameters', f"{performance_results['original_params']:,}", 
             f"{performance_results['original_params']:,}", 'Same']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=summary_data[0], fill_color='paleturquoise'),
                cells=dict(values=list(zip(*summary_data[1:])), fill_color='lavender')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Intel Optimization Performance Results",
            showlegend=False,
            height=800,
            width=1000
        )
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Performance comparison saved to {save_path}")
        else:
            fig.show()
    
    def plot_training_history(self, 
                            train_losses: List[float],
                            val_losses: Optional[List[float]] = None,
                            save_path: Optional[str] = None) -> None:
        """
        Plot training and validation loss curves
        
        Args:
            train_losses: Training loss history
            val_losses: Validation loss history (optional)
            save_path: Path to save the visualization
        """
        logger.info("Creating training history visualization...")
        
        epochs = list(range(1, len(train_losses) + 1))
        
        fig = go.Figure()
        
        # Training loss
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=train_losses,
                mode='lines',
                name='Training Loss',
                line=dict(color='blue', width=2)
            )
        )
        
        # Validation loss
        if val_losses:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=val_losses,
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='red', width=2)
                )
            )
        
        fig.update_layout(
            title='Training History - Intel-Optimized Financial GNN',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            hovermode='x unified',
            width=800,
            height=500
        )
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Training history saved to {save_path}")
        else:
            fig.show()
    
    def plot_feature_importance(self, 
                              feature_importance: Dict[str, float],
                              top_k: int = 20,
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance scores
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            top_k: Number of top features to show
            save_path: Path to save the visualization
        """
        logger.info("Creating feature importance visualization...")
        
        # Sort and select top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        features, scores = zip(*sorted_features)
        
        fig = go.Figure(
            go.Bar(
                x=list(scores),
                y=list(features),
                orientation='h',
                marker_color='lightblue'
            )
        )
        
        fig.update_layout(
            title=f'Top {top_k} Feature Importance Scores',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=max(400, top_k * 25),
            width=800
        )
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            fig.show()
    
    def create_dashboard(self, 
                        graph_data: Any,
                        symbols: List[str],
                        performance_results: Dict[str, float],
                        train_losses: List[float],
                        save_path: str = "dashboard.html") -> None:
        """
        Create comprehensive dashboard with all visualizations
        
        Args:
            graph_data: Graph data
            symbols: Stock symbols
            performance_results: Performance benchmark results
            train_losses: Training loss history
            save_path: Path to save dashboard
        """
        logger.info("Creating comprehensive dashboard...")
        
        # Create individual plots and save them
        dashboard_dir = Path(save_path).parent
        dashboard_dir.mkdir(exist_ok=True)
        
        # Individual plot paths
        graph_path = dashboard_dir / "financial_graph.html"
        heatmap_path = dashboard_dir / "correlation_heatmap.html"
        performance_path = dashboard_dir / "performance_comparison.html"
        training_path = dashboard_dir / "training_history.html"
        
        # Generate plots
        self.plot_financial_graph(graph_data, symbols, str(graph_path))
        self.plot_correlation_heatmap(graph_data, symbols, str(heatmap_path))
        self.plot_performance_comparison(performance_results, str(performance_path))
        self.plot_training_history(train_losses, save_path=str(training_path))
        
        # Create main dashboard HTML
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Intel-Optimized Financial GNN Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; color: #0071c5; }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
                .plot-container {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; }}
                iframe {{ width: 100%; height: 500px; border: none; }}
                .metrics {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Intel-Optimized Financial Graph Neural Network Dashboard</h1>
                <p>Comprehensive analysis of financial markets using Intel AI technologies</p>
            </div>
            
            <div class="metrics">
                <h3>📊 Performance Metrics</h3>
                <ul>
                    <li><strong>Speedup:</strong> {performance_results['speedup']:.2f}x faster inference</li>
                    <li><strong>Memory Reduction:</strong> {performance_results['memory_reduction']:.1f}% less memory usage</li>
                    <li><strong>Model Parameters:</strong> {performance_results['original_params']:,}</li>
                    <li><strong>Final Training Loss:</strong> {train_losses[-1]:.4f}</li>
                </ul>
            </div>
            
            <div class="grid">
                <div class="plot-container">
                    <h3>🕸️ Financial Correlation Graph</h3>
                    <iframe src="financial_graph.html"></iframe>
                </div>
                
                <div class="plot-container">
                    <h3>🔥 Correlation Heatmap</h3>
                    <iframe src="correlation_heatmap.html"></iframe>
                </div>
                
                <div class="plot-container">
                    <h3>⚡ Performance Comparison</h3>
                    <iframe src="performance_comparison.html"></iframe>
                </div>
                
                <div class="plot-container">
                    <h3>📈 Training History</h3>
                    <iframe src="training_history.html"></iframe>
                </div>
            </div>
            
            <footer style="text-align: center; margin-top: 40px; color: #666;">
                <p>Powered by Intel AI Technologies | OpenVINO | Intel Extension for PyTorch</p>
            </footer>
        </body>
        </html>
        """
        
        # Save dashboard
        with open(save_path, 'w') as f:
            f.write(dashboard_html)
        
        logger.info(f"Dashboard created and saved to {save_path}")
        print(f"🎉 Dashboard available at: {Path(save_path).absolute()}") 