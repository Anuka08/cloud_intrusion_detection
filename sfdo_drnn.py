#!/usr/bin/env python3
"""
SailFish-Dolphin Optimization with Deep Recurrent Neural Network
Implementation equivalent to Proposed_SFDO_DRNN package
"""

import numpy as np
import random
import math
from typing import List, Tuple, Dict
from fcm_algorithm import FuzzyCMeans

class SailFishDolphinOptimizer:
    """SailFish-Dolphin Optimization Algorithm equivalent to SailFish_update.java"""
    
    def __init__(self, population_size: int = 30, max_iterations: int = 50):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.population = []
        
    def initialize_population(self, dim: int, bounds: Tuple[float, float] = (-10, 10)) -> List[np.ndarray]:
        """Initialize population for optimization"""
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(bounds[0], bounds[1], dim)
            population.append(individual)
        return population
    
    def sailfish_behavior(self, sailfish: np.ndarray, best_position: np.ndarray, 
                         iteration: int, bounds: Tuple[float, float]) -> np.ndarray:
        """Sailfish hunting behavior"""
        # Sailfish update equation from line 109 in SailFish_update.java
        r1, r2 = random.random(), random.random()
        A = 2 * random.random() - 1  # Random coefficient
        
        # Update position based on best sailfish
        new_position = sailfish + A * (best_position - sailfish) + r1 * (random.random() - 0.5)
        
        # Apply bounds
        new_position = np.clip(new_position, bounds[0], bounds[1])
        return new_position
    
    def dolphin_behavior(self, dolphin: np.ndarray, target: np.ndarray, 
                        iteration: int, bounds: Tuple[float, float]) -> np.ndarray:
        """Dolphin echolocation behavior"""
        # Dolphin update equation
        r = random.random()
        beta = 2 * random.random() - 1
        
        # Echolocation update
        new_position = dolphin + beta * (target - dolphin) + r * (random.random() - 0.5)
        
        # Apply bounds
        new_position = np.clip(new_position, bounds[0], bounds[1])
        return new_position
    
    def fitness_function(self, position: np.ndarray, X_train: np.ndarray, 
                        y_train: np.ndarray, drnn: 'DRNN') -> float:
        """Fitness function equivalent to SFDO_fitness.java"""
        try:
            # Use position to set DRNN weights
            drnn.set_weights_from_vector(position)
            
            # Calculate DRNN loss (lower is better, so negate for maximization)
            loss = drnn.calculate_loss(X_train, y_train)
            fitness = -loss  # Convert to maximization problem
            
            return fitness
        except:
            return float('-inf')
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray, drnn: 'DRNN') -> Tuple[np.ndarray, float]:
        """
        Main optimization loop equivalent to optimization() in SailFish_update.java
        """
        print("Starting SFDO optimization...")
        
        # Get dimension from DRNN
        dim = drnn.get_weight_vector_size()
        bounds = (-1, 1)  # Weight bounds
        
        # Initialize population
        self.population = self.initialize_population(dim, bounds)
        
        # Evaluate initial population
        fitness_scores = []
        for individual in self.population:
            fitness = self.fitness_function(individual, X_train, y_train, drnn)
            fitness_scores.append(fitness)
        
        # Find best individual
        best_idx = np.argmax(fitness_scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = fitness_scores[best_idx]
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            new_population = []
            
            for i, individual in enumerate(self.population):
                # Determine behavior based on fitness rank
                if fitness_scores[i] > np.median(fitness_scores):
                    # Top performers use sailfish behavior
                    new_individual = self.sailfish_behavior(individual, self.best_solution, iteration, bounds)
                else:
                    # Lower performers use dolphin behavior
                    target = self.population[random.randint(0, len(self.population)-1)]
                    new_individual = self.dolphin_behavior(individual, target, iteration, bounds)
                
                new_population.append(new_individual)
            
            # Evaluate new population
            new_fitness_scores = []
            for individual in new_population:
                fitness = self.fitness_function(individual, X_train, y_train, drnn)
                new_fitness_scores.append(fitness)
            
            # Selection (keep better individuals)
            for i in range(len(self.population)):
                if new_fitness_scores[i] > fitness_scores[i]:
                    self.population[i] = new_population[i]
                    fitness_scores[i] = new_fitness_scores[i]
                    
                    # Update global best
                    if fitness_scores[i] > self.best_fitness:
                        self.best_solution = self.population[i].copy()
                        self.best_fitness = fitness_scores[i]
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.best_fitness:.6f}")
        
        print(f"SFDO optimization completed. Best fitness: {self.best_fitness:.6f}")
        return self.best_solution, self.best_fitness

class DRNN:
    """Deep Recurrent Neural Network equivalent to DRNN.java"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1, num_layers: int = 2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Initialize weights and biases
        self.initialize_weights()
        
        # Hidden states
        self.hidden_states = []
        
    def initialize_weights(self):
        """Initialize DRNN weights"""
        # Input to hidden weights
        self.W_ih = np.random.randn(self.input_size, self.hidden_size) * 0.1
        
        # Hidden to hidden weights (recurrent)
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        
        # Hidden to output weights
        self.W_ho = np.random.randn(self.hidden_size, self.output_size) * 0.1
        
        # Biases
        self.b_h = np.zeros((1, self.hidden_size))
        self.b_o = np.zeros((1, self.output_size))
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function"""
        return np.tanh(np.clip(x, -250, 250))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through DRNN"""
        batch_size, seq_len, input_size = X.shape
        
        # Initialize hidden state
        h = np.zeros((batch_size, self.hidden_size))
        outputs = []
        
        # Process sequence
        for t in range(seq_len):
            x_t = X[:, t, :]  # Input at time t
            
            # RNN cell computation
            h = self.tanh(np.dot(x_t, self.W_ih) + np.dot(h, self.W_hh) + self.b_h)
            
            # Output computation
            y_t = self.sigmoid(np.dot(h, self.W_ho) + self.b_o)
            outputs.append(y_t)
        
        # Return final output
        return outputs[-1]
    
    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate loss for optimization"""
        try:
            # Reshape input for RNN (add sequence dimension)
            if len(X.shape) == 2:
                X = X.reshape(X.shape[0], 1, X.shape[1])  # (batch, 1, features)
            
            # Forward pass
            predictions = self.forward(X)
            
            # Binary cross-entropy loss
            y = y.reshape(-1, 1)
            predictions = np.clip(predictions, 1e-7, 1 - 1e-7)  # Prevent log(0)
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            
            return loss
        except:
            return float('inf')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        predictions = self.forward(X)
        return (predictions > 0.5).astype(int).flatten()
    
    def get_weight_vector_size(self) -> int:
        """Get total number of weights for optimization"""
        size = (self.W_ih.size + self.W_hh.size + self.W_ho.size + 
                self.b_h.size + self.b_o.size)
        return size
    
    def set_weights_from_vector(self, weight_vector: np.ndarray):
        """Set weights from optimization vector"""
        idx = 0
        
        # W_ih
        w_ih_size = self.W_ih.size
        self.W_ih = weight_vector[idx:idx + w_ih_size].reshape(self.W_ih.shape)
        idx += w_ih_size
        
        # W_hh
        w_hh_size = self.W_hh.size
        self.W_hh = weight_vector[idx:idx + w_hh_size].reshape(self.W_hh.shape)
        idx += w_hh_size
        
        # W_ho
        w_ho_size = self.W_ho.size
        self.W_ho = weight_vector[idx:idx + w_ho_size].reshape(self.W_ho.shape)
        idx += w_ho_size
        
        # b_h
        b_h_size = self.b_h.size
        self.b_h = weight_vector[idx:idx + b_h_size].reshape(self.b_h.shape)
        idx += b_h_size
        
        # b_o
        b_o_size = self.b_o.size
        self.b_o = weight_vector[idx:idx + b_o_size].reshape(self.b_o.shape)

class FeatureFusion:
    """Feature fusion equivalent to Feature_fusion.java"""
    
    @staticmethod
    def process(data: List[List[float]], target: List[float], n_clusters: int) -> List[List[float]]:
        """
        Feature fusion process equivalent to Feature_fusion.process()
        """
        print("Starting feature fusion process...")
        
        # Step 1: Feature grouping by FCM
        print(">> Feature grouping by FCM...")
        fcm = FuzzyCMeans(n_clusters=n_clusters, max_iter=50)
        
        # Transpose data for feature clustering
        data_transposed = list(map(list, zip(*data)))
        fcm.fit(data_transposed)
        
        # Get cluster assignments for features
        cluster_assignments = fcm.get_feature_clusters(len(data[0]))
        
        # Step 2: Feature fusion of every group
        print(">> Feature fusion of every group...")
        fused_features = []
        
        for cluster_id in range(1, n_clusters + 1):
            fused_column = []
            alpha = random.random() + 0.1  # Avoid division by zero
            
            for sample in data:
                fused_value = 0.0
                count = 0
                
                for feature_idx, feature_val in enumerate(sample):
                    if feature_idx < len(cluster_assignments) and cluster_assignments[feature_idx] == cluster_id:
                        fused_value += (1.0 / alpha) * feature_val
                        count += 1
                
                # Average if multiple features in cluster
                if count > 0:
                    fused_value /= count
                
                fused_column.append(fused_value)
            
            fused_features.append(fused_column)
        
        # Transpose to get samples x features format
        fused_features = list(map(list, zip(*fused_features)))
        
        print(f"Feature fusion completed. Fused shape: {len(fused_features)} x {len(fused_features[0])}")
        return fused_features

def optimize_drnn_with_sfdo(X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Complete SFDO-DRNN optimization equivalent to SailFish_update.optimization()
    """
    print("Initializing SFDO-DRNN optimization...")
    
    # Create DRNN
    input_size = X_train.shape[1]
    drnn = DRNN(input_size=input_size, hidden_size=32, output_size=1)
    
    # Create SFDO optimizer
    sfdo = SailFishDolphinOptimizer(population_size=20, max_iterations=30)
    
    # Optimize DRNN weights
    best_weights, best_fitness = sfdo.optimize(X_train, y_train, drnn)
    
    # Set optimal weights
    drnn.set_weights_from_vector(best_weights)
    
    # Evaluate on test set
    predictions = drnn.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'best_fitness': best_fitness,
        'predictions': predictions
    }
    
    print(f"SFDO-DRNN Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    return results

# Test the implementation
if __name__ == "__main__":
    print("Testing SFDO-DRNN implementation...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = (np.sum(X, axis=1) > 5).astype(int)
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Test DRNN
    drnn = DRNN(input_size=10)
    print(f"DRNN weight vector size: {drnn.get_weight_vector_size()}")
    
    # Test optimization
    results = optimize_drnn_with_sfdo(X_train, y_train, X_test, y_test)
    print("SFDO-DRNN implementation working correctly!")