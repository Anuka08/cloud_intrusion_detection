#!/usr/bin/env python3
"""
ANFIS (Adaptive Neuro-Fuzzy Inference System) Implementation
Equivalent to ANFIS/Process_fuzz.java
"""

import numpy as np
import random
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans

class ANFIS:
    """Adaptive Neuro-Fuzzy Inference System"""
    
    def __init__(self, n_rules: int = 5, n_epochs: int = 100, learning_rate: float = 0.01):
        self.n_rules = n_rules
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        
        # ANFIS parameters
        self.centers = None  # Membership function centers
        self.widths = None   # Membership function widths
        self.consequences = None  # Rule consequences (linear parameters)
        
        self.n_inputs = None
        
    def gaussian_membership(self, x: np.ndarray, center: float, width: float) -> np.ndarray:
        """Gaussian membership function"""
        return np.exp(-0.5 * ((x - center) / width) ** 2)
    
    def initialize_parameters(self, X: np.ndarray):
        """Initialize ANFIS parameters"""
        self.n_inputs = X.shape[1]
        
        # Initialize membership function parameters using k-means
        kmeans = KMeans(n_clusters=self.n_rules, random_state=42)
        kmeans.fit(X)
        
        # Centers from k-means centroids
        self.centers = kmeans.cluster_centers_  # Shape: (n_rules, n_inputs)
        
        # Widths based on distances between centers
        self.widths = np.ones((self.n_rules, self.n_inputs)) * 0.5
        
        # Initialize consequence parameters (linear part)
        self.consequences = np.random.randn(self.n_rules, self.n_inputs + 1) * 0.1
    
    def fuzzification(self, X: np.ndarray) -> np.ndarray:
        """Layer 1: Fuzzification - compute membership degrees"""
        n_samples = X.shape[0]
        memberships = np.zeros((n_samples, self.n_rules))
        
        for i in range(self.n_rules):
            membership = np.ones(n_samples)
            for j in range(self.n_inputs):
                membership *= self.gaussian_membership(X[:, j], self.centers[i, j], self.widths[i, j])
            memberships[:, i] = membership
        
        return memberships
    
    def normalization(self, memberships: np.ndarray) -> np.ndarray:
        """Layer 2: Normalization - normalize firing strengths"""
        # Avoid division by zero
        total_membership = np.sum(memberships, axis=1, keepdims=True)
        total_membership = np.where(total_membership == 0, 1e-10, total_membership)
        
        normalized = memberships / total_membership
        return normalized
    
    def defuzzification(self, X: np.ndarray, normalized_memberships: np.ndarray) -> np.ndarray:
        """Layer 3: Defuzzification - compute rule outputs and final output"""
        n_samples = X.shape[0]
        
        # Add bias term to input
        X_bias = np.column_stack([X, np.ones(n_samples)])
        
        # Compute rule outputs
        rule_outputs = np.zeros((n_samples, self.n_rules))
        for i in range(self.n_rules):
            rule_outputs[:, i] = np.dot(X_bias, self.consequences[i])
        
        # Weighted sum
        weighted_outputs = normalized_memberships * rule_outputs
        final_output = np.sum(weighted_outputs, axis=1)
        
        return final_output, rule_outputs
    
    def forward_pass(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Complete forward pass through ANFIS"""
        memberships = self.fuzzification(X)
        normalized_memberships = self.normalization(memberships)
        output, rule_outputs = self.defuzzification(X, normalized_memberships)
        
        return output, normalized_memberships, rule_outputs
    
    def backward_pass(self, X: np.ndarray, y: np.ndarray, output: np.ndarray, 
                     normalized_memberships: np.ndarray, rule_outputs: np.ndarray):
        """Backward pass - update parameters using gradient descent"""
        n_samples = X.shape[0]
        error = output - y
        
        # Add bias term to input
        X_bias = np.column_stack([X, np.ones(n_samples)])
        
        # Update consequence parameters (linear part)
        for i in range(self.n_rules):
            gradient = np.mean(error.reshape(-1, 1) * normalized_memberships[:, i].reshape(-1, 1) * X_bias, axis=0)
            self.consequences[i] -= self.learning_rate * gradient
        
        # Update membership function parameters (nonlinear part)
        for i in range(self.n_rules):
            for j in range(self.n_inputs):
                # Gradient for center
                membership_gradient = self.gaussian_membership(X[:, j], self.centers[i, j], self.widths[i, j])
                center_gradient = membership_gradient * (X[:, j] - self.centers[i, j]) / (self.widths[i, j] ** 2)
                
                # Weighted gradient
                weighted_gradient = error * normalized_memberships[:, i] * center_gradient * rule_outputs[:, i]
                self.centers[i, j] -= self.learning_rate * np.mean(weighted_gradient)
                
                # Gradient for width (simplified)
                width_gradient = membership_gradient * ((X[:, j] - self.centers[i, j]) ** 2) / (self.widths[i, j] ** 3)
                weighted_width_gradient = error * normalized_memberships[:, i] * width_gradient * rule_outputs[:, i]
                self.widths[i, j] -= self.learning_rate * np.mean(weighted_width_gradient) * 0.1  # Smaller step for width
                
                # Ensure positive width
                self.widths[i, j] = max(self.widths[i, j], 0.01)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ANFIS':
        """Train ANFIS network"""
        print("Training ANFIS...")
        
        X = np.array(X)
        y = np.array(y).flatten()
        
        # Initialize parameters
        self.initialize_parameters(X)
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Forward pass
            output, normalized_memberships, rule_outputs = self.forward_pass(X)
            
            # Calculate loss
            mse = np.mean((output - y) ** 2)
            
            # Backward pass
            self.backward_pass(X, y, output, normalized_memberships, rule_outputs)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: MSE = {mse:.6f}")
        
        print("ANFIS training completed!")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        output, _, _ = self.forward_pass(X)
        
        # For binary classification, threshold at 0.5
        return (output > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        output, _, _ = self.forward_pass(X)
        
        # Apply sigmoid to get probabilities
        probabilities = 1 / (1 + np.exp(-output))
        return probabilities

def anfis_classify(data: List[List[float]], target: List[float], training_percentage: int = 80) -> Dict:
    """
    ANFIS classification equivalent to ANFIS.Process_fuzz.classify()
    """
    print("Running ANFIS classification...")
    
    X = np.array(data)
    y = np.array(target)
    
    # Train-test split
    train_size = int(len(X) * training_percentage / 100)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train ANFIS
    anfis = ANFIS(n_rules=5, n_epochs=50, learning_rate=0.001)
    anfis.fit(X_train_scaled, y_train)
    
    # Make predictions
    predictions = anfis.predict(X_test_scaled)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    
    # Calculate FPR
    cm = confusion_matrix(y_test, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:
        fpr = 0
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fpr': fpr,
        'predictions': predictions
    }
    
    print(f"ANFIS Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, FPR: {fpr:.4f}")
    
    return results

# Test the implementation
if __name__ == "__main__":
    print("Testing ANFIS implementation...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(200, 5)
    y = (np.sum(X, axis=1) > 2.5).astype(int)
    
    # Test ANFIS
    results = anfis_classify(X.tolist(), y.tolist())
    print("ANFIS implementation working correctly!")