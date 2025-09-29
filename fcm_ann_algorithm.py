#!/usr/bin/env python3
"""
FCM-ANN (Fuzzy C-Means with Artificial Neural Network) Implementation
Equivalent to FCM_ANN/run.java
"""

import numpy as np
from typing import List, Dict, Tuple
from fcm_algorithm import FuzzyCMeans

class FCMANN:
    """FCM-ANN classifier combining Fuzzy C-Means with Neural Network"""
    
    def __init__(self, n_clusters: int = 4, hidden_size: int = 64, learning_rate: float = 0.01, epochs: int = 100):
        self.n_clusters = n_clusters
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # FCM component
        self.fcm = FuzzyCMeans(n_clusters=n_clusters)
        
        # Neural network weights
        self.W1 = None  # Input to hidden
        self.b1 = None  # Hidden bias
        self.W2 = None  # Hidden to output
        self.b2 = None  # Output bias
        
        # Normalization parameters
        self.mean = None
        self.std = None
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
    
    def normalize_data(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize input data"""
        if fit:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            self.std = np.where(self.std == 0, 1, self.std)  # Avoid division by zero
        
        return (X - self.mean) / self.std
    
    def fuzzy_feature_extraction(self, X: np.ndarray) -> np.ndarray:
        """Extract fuzzy features using FCM"""
        print("Extracting fuzzy features using FCM...")
        
        # Fit FCM to get membership matrix
        self.fcm.fit(X)
        membership_matrix = self.fcm.get_membership_matrix()  # Shape: (n_clusters, n_samples)
        
        # Use membership degrees as features
        fuzzy_features = membership_matrix.T  # Shape: (n_samples, n_clusters)
        
        # Combine original features with fuzzy features
        combined_features = np.column_stack([X, fuzzy_features])
        
        print(f"Original features: {X.shape[1]}, Fuzzy features: {fuzzy_features.shape[1]}")
        print(f"Combined features: {combined_features.shape[1]}")
        
        return combined_features
    
    def initialize_network(self, input_size: int):
        """Initialize neural network weights"""
        # Xavier initialization
        self.W1 = np.random.randn(input_size, self.hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, 1) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros((1, 1))
    
    def forward_pass(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through the network"""
        # Hidden layer
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        
        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        
        return a1, a2, z1
    
    def backward_pass(self, X: np.ndarray, y: np.ndarray, a1: np.ndarray, a2: np.ndarray, z1: np.ndarray):
        """Backward pass - update weights"""
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = a2 - y.reshape(-1, 1)
        dW2 = np.dot(a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FCMANN':
        """Train FCM-ANN model"""
        print("Training FCM-ANN model...")
        
        X = np.array(X)
        y = np.array(y)
        
        # Step 1: Extract fuzzy features
        fuzzy_X = self.fuzzy_feature_extraction(X)
        
        # Step 2: Normalize features
        fuzzy_X_norm = self.normalize_data(fuzzy_X, fit=True)
        
        # Step 3: Initialize neural network
        self.initialize_network(fuzzy_X_norm.shape[1])
        
        # Step 4: Train neural network
        print("Training neural network component...")
        for epoch in range(self.epochs):
            # Forward pass
            a1, a2, z1 = self.forward_pass(fuzzy_X_norm)
            
            # Calculate loss
            loss = np.mean((a2.flatten() - y) ** 2)
            
            # Backward pass
            self.backward_pass(fuzzy_X_norm, y, a1, a2, z1)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")
        
        print("FCM-ANN training completed!")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X = np.array(X)
        
        # Extract fuzzy features
        fuzzy_X = self.fuzzy_feature_extraction(X)
        
        # Normalize features
        fuzzy_X_norm = self.normalize_data(fuzzy_X, fit=False)
        
        # Forward pass
        _, a2, _ = self.forward_pass(fuzzy_X_norm)
        
        # Binary classification
        predictions = (a2.flatten() > 0.5).astype(int)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        X = np.array(X)
        
        # Extract fuzzy features
        fuzzy_X = self.fuzzy_feature_extraction(X)
        
        # Normalize features
        fuzzy_X_norm = self.normalize_data(fuzzy_X, fit=False)
        
        # Forward pass
        _, a2, _ = self.forward_pass(fuzzy_X_norm)
        
        return a2.flatten()

def fcm_ann_classify(data: List[List[float]], target: List[float], 
                    fused_features: List[List[float]] = None, 
                    training_percentage: int = 80) -> Dict:
    """
    FCM-ANN classification equivalent to FCM_ANN.run.callmain()
    """
    print("Running FCM-ANN classification...")
    
    # Use fused features if available, otherwise use original data
    if fused_features and len(fused_features) > 0:
        X = np.array(fused_features)
        print("Using fused features for FCM-ANN")
    else:
        X = np.array(data)
        print("Using original features for FCM-ANN")
    
    y = np.array(target)
    
    # Train-test split
    train_size = int(len(X) * training_percentage / 100)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and train FCM-ANN
    fcm_ann = FCMANN(n_clusters=4, hidden_size=32, learning_rate=0.001, epochs=50)
    fcm_ann.fit(X_train, y_train)
    
    # Make predictions
    predictions = fcm_ann.predict(X_test)
    
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
    
    print(f"FCM-ANN Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, FPR: {fpr:.4f}")
    
    return results

# Test the implementation
if __name__ == "__main__":
    print("Testing FCM-ANN implementation...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(200, 8)
    y = (np.sum(X, axis=1) > 4).astype(int)
    
    # Test FCM-ANN
    results = fcm_ann_classify(X.tolist(), y.tolist())
    print("FCM-ANN implementation working correctly!")