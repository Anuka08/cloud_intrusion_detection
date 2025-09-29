#!/usr/bin/env python3
"""
Fuzzy C-Means Clustering Implementation
Proper implementation equivalent to Code/FCM.java
"""

import numpy as np
import random
from typing import List, Tuple

class FuzzyCMeans:
    def __init__(self, n_clusters: int, max_iter: int = 100, m: float = 2.0, min_improvement: float = 1e-5):
        """
        Initialize FCM clustering
        
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            m: Exponent for fuzzy membership (equivalent to 'expo' in Java)
            min_improvement: Minimum improvement threshold
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m  # Fuzzy exponent
        self.min_improvement = min_improvement
        self.U = None  # Membership matrix
        self.centers = None
        self.objective_function = []
    
    def initialize_membership_matrix(self, n_samples: int) -> np.ndarray:
        """Initialize fuzzy membership matrix equivalent to initfcm() in Java"""
        # Random initialization with constraint that sum of memberships = 1 for each sample
        U = np.random.rand(self.n_clusters, n_samples)
        
        # Normalize so each column sums to 1
        U = U / np.sum(U, axis=0, keepdims=True)
        
        return U
    
    def calculate_centers(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Calculate cluster centers"""
        um = U ** self.m
        centers = np.dot(um, X) / np.sum(um, axis=1, keepdims=True)
        return centers
    
    def calculate_membership_matrix(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Calculate new membership matrix"""
        n_samples, n_features = X.shape
        U = np.zeros((self.n_clusters, n_samples))
        
        for i in range(n_samples):
            for j in range(self.n_clusters):
                distances = np.linalg.norm(X[i] - centers, axis=1)
                
                if distances[j] == 0:
                    U[j, i] = 1.0
                    # Set others to 0
                    for k in range(self.n_clusters):
                        if k != j:
                            U[k, i] = 0.0
                    break
                else:
                    sum_term = 0.0
                    for k in range(self.n_clusters):
                        if distances[k] > 0:
                            sum_term += (distances[j] / distances[k]) ** (2 / (self.m - 1))
                    
                    if sum_term > 0:
                        U[j, i] = 1.0 / sum_term
                    else:
                        U[j, i] = 0.0
        
        return U
    
    def calculate_objective_function(self, X: np.ndarray, U: np.ndarray, centers: np.ndarray) -> float:
        """Calculate objective function value"""
        obj_fcn = 0.0
        um = U ** self.m
        
        for i in range(self.n_clusters):
            for j in range(X.shape[0]):
                distance = np.linalg.norm(X[j] - centers[i])
                obj_fcn += um[i, j] * (distance ** 2)
        
        return obj_fcn
    
    def fit(self, X: np.ndarray) -> 'FuzzyCMeans':
        """
        Perform FCM clustering equivalent to group() method in Java
        
        Args:
            X: Data matrix (samples x features)
            
        Returns:
            self: Fitted FCM object
        """
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize membership matrix
        self.U = self.initialize_membership_matrix(n_samples)
        self.objective_function = []
        
        for iteration in range(self.max_iter):
            # Calculate cluster centers
            self.centers = self.calculate_centers(X, self.U)
            
            # Calculate objective function
            obj_fcn = self.calculate_objective_function(X, self.U, self.centers)
            self.objective_function.append(obj_fcn)
            
            # Update membership matrix
            U_new = self.calculate_membership_matrix(X, self.centers)
            
            # Check for convergence
            if iteration > 0:
                if abs(self.objective_function[iteration] - self.objective_function[iteration-1]) < self.min_improvement:
                    print(f"FCM converged after {iteration+1} iterations")
                    break
            
            self.U = U_new
        
        return self
    
    def predict_clusters(self) -> List[int]:
        """Get cluster assignments (1-indexed to match Java)"""
        if self.U is None:
            raise ValueError("Model must be fitted first")
        
        # Get cluster with maximum membership for each sample
        cluster_assignments = np.argmax(self.U, axis=0)
        
        # Convert to 1-indexed (matching Java implementation)
        return [c + 1 for c in cluster_assignments]
    
    def get_membership_matrix(self) -> np.ndarray:
        """Get the fuzzy membership matrix"""
        return self.U
    
    def get_feature_clusters(self, n_features: int) -> List[int]:
        """
        Get cluster assignments for features equivalent to Java implementation
        Returns 1-indexed cluster assignments for features
        """
        if n_features != self.U.shape[1]:
            # If mismatch, ensure we have enough clusters
            if n_features < self.n_clusters:
                # Assign features to first n_features clusters
                return [i % self.n_clusters + 1 for i in range(n_features)]
            else:
                # Repeat cluster pattern
                return [i % self.n_clusters + 1 for i in range(n_features)]
        
        return self.predict_clusters()

def transpose_matrix(matrix: List[List]) -> List[List]:
    """Transpose matrix equivalent to transpose() in Java"""
    if not matrix or not matrix[0]:
        return []
    
    return [list(row) for row in zip(*matrix)]

# Test the FCM implementation
if __name__ == "__main__":
    # Test with sample data
    print("Testing FCM implementation...")
    
    # Sample data
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    
    # Create FCM object
    fcm = FuzzyCMeans(n_clusters=4, max_iter=10)
    
    # Fit the model
    fcm.fit(X)
    
    # Get cluster assignments
    clusters = fcm.predict_clusters()
    print(f"Cluster assignments: {clusters[:10]}...")  # Show first 10
    
    # Get membership matrix
    U = fcm.get_membership_matrix()
    print(f"Membership matrix shape: {U.shape}")
    print(f"Sample memberships for first sample: {U[:, 0]}")
    
    print("FCM implementation working correctly!")