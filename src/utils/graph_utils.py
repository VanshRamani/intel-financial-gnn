"""
Graph Utilities for Financial Data

Constructs graph representations of financial markets where stocks are nodes
and correlations/relationships are edges.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score
import networkx as nx

logger = logging.getLogger(__name__)


class GraphConstructor:
    """
    Constructs graph representations of financial markets
    """
    
    def __init__(self, correlation_method='pearson'):
        self.correlation_method = correlation_method
        self.node_features = []
        self.edge_weights = []
        self.symbols = []
        
    def build_correlation_graph(self, 
                               processed_data: Dict[str, pd.DataFrame],
                               correlation_threshold: float = 0.3,
                               max_edges_per_node: int = 5) -> Data:
        """
        Build a correlation graph from financial data
        
        Args:
            processed_data: Dictionary of processed financial DataFrames
            correlation_threshold: Minimum correlation to create edge
            max_edges_per_node: Maximum edges per node
            
        Returns:
            PyTorch Geometric Data object
        """
        logger.info(f"Building correlation graph with {len(processed_data)} stocks...")
        
        self.symbols = list(processed_data.keys())
        num_nodes = len(self.symbols)
        
        # Extract features for each stock
        node_features = self._extract_node_features(processed_data)
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(processed_data)
        
        # Build edges based on correlations
        edge_index, edge_weights = self._build_edges(
            correlation_matrix, 
            correlation_threshold, 
            max_edges_per_node
        )
        
        # Create targets (using first stock's future returns as example)
        targets = self._create_targets(processed_data)
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        
        # Create graph data object
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )
        
        # Add metadata
        graph_data.symbols = self.symbols
        graph_data.num_nodes = num_nodes
        graph_data.correlation_matrix = correlation_matrix
        
        logger.info(f"Graph created: {num_nodes} nodes, {edge_index.size(1)} edges")
        return graph_data
    
    def _extract_node_features(self, processed_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Extract node features for each stock
        
        Args:
            processed_data: Dictionary of processed DataFrames
            
        Returns:
            Node feature matrix [num_nodes, num_features]
        """
        logger.info("Extracting node features...")
        
        node_features = []
        
        for symbol in self.symbols:
            data = processed_data[symbol]
            
            # Select latest values for features (excluding targets)
            feature_cols = [col for col in data.columns 
                          if not col.startswith('future_') and col not in ['price_up_1d', 'price_up_5d']]
            
            # Get latest non-NaN values
            latest_features = data[feature_cols].dropna().iloc[-1].values
            
            # Handle any remaining NaN values
            latest_features = np.nan_to_num(latest_features, nan=0.0)
            
            node_features.append(latest_features)
        
        node_features = np.array(node_features)
        logger.info(f"Extracted {node_features.shape[1]} features for {node_features.shape[0]} nodes")
        
        return node_features
    
    def _calculate_correlation_matrix(self, processed_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Calculate correlation matrix between stocks
        
        Args:
            processed_data: Dictionary of processed DataFrames
            
        Returns:
            Correlation matrix [num_stocks, num_stocks]
        """
        logger.info(f"Calculating {self.correlation_method} correlation matrix...")
        
        # Extract price data
        price_data = {}
        for symbol, data in processed_data.items():
            if 'Close' in data.columns:
                price_data[symbol] = data['Close']
        
        # Create price DataFrame
        price_df = pd.DataFrame(price_data)
        
        # Calculate correlations
        if self.correlation_method == 'pearson':
            correlation_matrix = price_df.corr().values
        elif self.correlation_method == 'spearman':
            correlation_matrix = price_df.corr(method='spearman').values
        else:
            # Fallback to Pearson
            correlation_matrix = price_df.corr().values
        
        # Handle NaN values
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        logger.info("Correlation matrix calculated")
        return correlation_matrix
    
    def _build_edges(self, 
                    correlation_matrix: np.ndarray,
                    correlation_threshold: float,
                    max_edges_per_node: int) -> Tuple[List[List[int]], List[float]]:
        """
        Build edges based on correlation matrix
        
        Args:
            correlation_matrix: Correlation matrix
            correlation_threshold: Minimum correlation for edge
            max_edges_per_node: Maximum edges per node
            
        Returns:
            Edge index and edge weights
        """
        logger.info(f"Building edges with threshold {correlation_threshold}...")
        
        num_nodes = correlation_matrix.shape[0]
        edge_list = []
        edge_weights = []
        
        for i in range(num_nodes):
            # Get correlations for node i
            correlations = correlation_matrix[i]
            
            # Find nodes with correlation above threshold (excluding self)
            candidates = []
            for j in range(num_nodes):
                if i != j and abs(correlations[j]) >= correlation_threshold:
                    candidates.append((j, abs(correlations[j])))
            
            # Sort by correlation strength and take top connections
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[:max_edges_per_node]
            
            # Add edges
            for j, weight in top_candidates:
                edge_list.append([i, j])
                edge_weights.append(weight)
        
        # Convert to edge_index format [2, num_edges]
        if edge_list:
            edge_index = list(zip(*edge_list))
        else:
            edge_index = [[], []]
        
        logger.info(f"Created {len(edge_weights)} edges")
        return edge_index, edge_weights
    
    def _create_targets(self, processed_data: Dict[str, pd.DataFrame]) -> List[float]:
        """
        Create prediction targets for each node
        
        Args:
            processed_data: Dictionary of processed DataFrames
            
        Returns:
            Target values for each node
        """
        targets = []
        
        for symbol in self.symbols:
            data = processed_data[symbol]
            
            # Use future 1-day return as target
            if 'future_return_1d' in data.columns:
                target = data['future_return_1d'].dropna().iloc[-1] if not data['future_return_1d'].dropna().empty else 0.0
            else:
                target = 0.0  # Default target
            
            targets.append(target)
        
        return targets
    
    def create_dynamic_graph(self, 
                           processed_data: Dict[str, pd.DataFrame],
                           window_size: int = 60,
                           stride: int = 1) -> List[Data]:
        """
        Create a sequence of dynamic graphs over time
        
        Args:
            processed_data: Dictionary of processed DataFrames
            window_size: Size of rolling window
            stride: Stride between windows
            
        Returns:
            List of graph snapshots
        """
        logger.info(f"Creating dynamic graphs with window size {window_size}...")
        
        # Find common date range
        min_len = min(len(data) for data in processed_data.values())
        max_windows = (min_len - window_size) // stride + 1
        
        graphs = []
        
        for i in range(0, max_windows * stride, stride):
            window_data = {}
            
            # Extract window data for each stock
            for symbol, data in processed_data.items():
                window_data[symbol] = data.iloc[i:i+window_size]
            
            # Build graph for this window
            graph = self.build_correlation_graph(window_data)
            graph.time_step = i // stride
            
            graphs.append(graph)
        
        logger.info(f"Created {len(graphs)} dynamic graph snapshots")
        return graphs
    
    def add_sector_edges(self, graph_data: Data, market_data: Dict[str, Dict]) -> Data:
        """
        Add edges between stocks in the same sector
        
        Args:
            graph_data: Original graph data
            market_data: Market metadata including sectors
            
        Returns:
            Graph with sector edges added
        """
        logger.info("Adding sector-based edges...")
        
        # Group stocks by sector
        sectors = {}
        for i, symbol in enumerate(self.symbols):
            if symbol in market_data:
                sector = market_data[symbol].get('sector', 'Unknown')
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(i)
        
        # Add edges within sectors
        additional_edges = []
        additional_weights = []
        
        for sector, nodes in sectors.items():
            if len(nodes) > 1:
                # Add edges between all pairs in the sector
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        # Check if edge already exists
                        existing_edge = False
                        for k in range(graph_data.edge_index.size(1)):
                            if ((graph_data.edge_index[0, k] == nodes[i] and graph_data.edge_index[1, k] == nodes[j]) or
                                (graph_data.edge_index[0, k] == nodes[j] and graph_data.edge_index[1, k] == nodes[i])):
                                existing_edge = True
                                break
                        
                        if not existing_edge:
                            additional_edges.extend([[nodes[i], nodes[j]], [nodes[j], nodes[i]]])
                            additional_weights.extend([0.5, 0.5])  # Moderate sector connection weight
        
        if additional_edges:
            # Combine with existing edges
            new_edge_index = torch.cat([
                graph_data.edge_index,
                torch.tensor(additional_edges, dtype=torch.long).t()
            ], dim=1)
            
            new_edge_attr = torch.cat([
                graph_data.edge_attr,
                torch.tensor(additional_weights, dtype=torch.float32).unsqueeze(1)
            ], dim=0)
            
            graph_data.edge_index = new_edge_index
            graph_data.edge_attr = new_edge_attr
            
            logger.info(f"Added {len(additional_edges)} sector-based edges")
        
        return graph_data
    
    def visualize_graph_structure(self, graph_data: Data) -> nx.Graph:
        """
        Convert to NetworkX graph for visualization
        
        Args:
            graph_data: PyTorch Geometric graph
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        # Add nodes with features
        for i in range(graph_data.num_nodes):
            G.add_node(i, symbol=self.symbols[i] if i < len(self.symbols) else f"Node_{i}")
        
        # Add edges with weights
        edge_index = graph_data.edge_index.numpy()
        edge_weights = graph_data.edge_attr.numpy().flatten()
        
        for i in range(edge_index.shape[1]):
            source, target = edge_index[0, i], edge_index[1, i]
            weight = edge_weights[i]
            G.add_edge(source, target, weight=weight)
        
        return G 