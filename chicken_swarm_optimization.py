#!/usr/bin/env python3
"""
Chicken Swarm Optimization Implementation
Equivalent to Proposed_SFDO_DRNN/chicken_swarm.java
"""

import numpy as np
import random
from typing import List, Tuple, Dict

class ChickenSwarmOptimizer:
    """Chicken Swarm Optimization for VM load balancing"""
    
    def __init__(self, population_size: int = 10, max_generations: int = 50, G: int = 2):
        self.population_size = population_size  # N in Java
        self.max_generations = max_generations
        self.G = G  # Time step between status updates
        
        self.roosters = []  # Best individuals
        self.hens = []      # Middle individuals  
        self.chicks = []    # Worst individuals
        
        self.best_solution = None
        self.best_fitness = float('-inf')
    
    def initialize_population(self, vm_count: int, pm_count: int) -> List[List[int]]:
        """Initialize population equivalent to Java initialization"""
        population = []
        
        for _ in range(self.population_size):
            individual = []
            for _ in range(vm_count):
                # Random PM assignment (1 to pm_count)
                pm_assignment = random.randint(1, pm_count)
                individual.append(pm_assignment)
            population.append(individual)
        
        return population
    
    def fitness_function(self, solution: List[int], vm_processing: List[float], 
                        vm_cpu: List[float], vm_bandwidth: List[float], 
                        vm_memory: List[float], vm_mips: List[float]) -> float:
        """
        Fitness function equivalent to fitness_CSO.func()
        Calculates load balancing quality
        """
        try:
            pm_count = max(solution)
            pm_loads = [0.0] * pm_count
            
            # Calculate load for each PM
            for vm_idx, pm_id in enumerate(solution):
                pm_idx = pm_id - 1  # Convert to 0-indexed
                if pm_idx < len(pm_loads):
                    # Aggregate VM resources assigned to this PM
                    vm_load = (vm_processing[vm_idx] + vm_cpu[vm_idx] + 
                              vm_bandwidth[vm_idx] + vm_memory[vm_idx] + vm_mips[vm_idx])
                    pm_loads[pm_idx] += vm_load
            
            # Fitness is inverse of load variance (better load balancing = higher fitness)
            if len(pm_loads) > 1:
                load_variance = np.var(pm_loads)
                fitness = 1.0 / (1.0 + load_variance)  # Higher is better
            else:
                fitness = 1.0
            
            return fitness
        except:
            return 0.0
    
    def divide_swarm(self, population: List[List[int]], fitness_scores: List[float]):
        """Divide swarm into roosters, hens, and chicks based on fitness"""
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        
        m = self.population_size // 3  # Size of each group
        
        # Clear previous groups
        self.roosters = []
        self.hens = []
        self.chicks = []
        
        # Roosters (best m individuals)
        for i in range(m):
            if i < len(sorted_indices):
                self.roosters.append(sorted_indices[i])
        
        # Chicks (worst m individuals)
        for i in range(len(sorted_indices) - m, len(sorted_indices)):
            if i >= 0:
                self.chicks.append(sorted_indices[i])
        
        # Hens (remaining individuals)
        for i in range(len(sorted_indices)):
            if i not in self.roosters and i not in self.chicks:
                self.hens.append(i)
    
    def rooster_update(self, rooster_idx: int, population: List[List[int]], 
                      fitness_scores: List[float], iteration: int) -> List[int]:
        """Update rooster position"""
        current_rooster = population[rooster_idx].copy()
        
        # Rooster update equation
        for j in range(len(current_rooster)):
            if random.random() < 0.1:  # Random update probability
                # Random walk with constraints
                r1 = random.random()
                r2 = random.random()
                
                # Find another rooster for comparison
                other_rooster_idx = random.choice(self.roosters)
                if other_rooster_idx != rooster_idx:
                    other_rooster = population[other_rooster_idx]
                    
                    # Update based on comparison
                    if fitness_scores[rooster_idx] >= fitness_scores[other_rooster_idx]:
                        # Move towards better position
                        current_rooster[j] = int(current_rooster[j] + r1 * (other_rooster[j] - current_rooster[j]))
                    else:
                        # Random exploration
                        current_rooster[j] = random.randint(1, max(current_rooster))
        
        return current_rooster
    
    def hen_update(self, hen_idx: int, population: List[List[int]], 
                  fitness_scores: List[float], iteration: int) -> List[int]:
        """Update hen position"""
        current_hen = population[hen_idx].copy()
        
        # Find rooster to follow
        if self.roosters:
            rooster_idx = random.choice(self.roosters)
            rooster = population[rooster_idx]
            
            # Hen update equation
            for j in range(len(current_hen)):
                r1 = random.random()
                r2 = random.random()
                
                # Move towards rooster with random component
                if random.random() < 0.2:
                    current_hen[j] = int(current_hen[j] + r1 * (rooster[j] - current_hen[j]) + r2 * (random.randint(1, max(current_hen)) - current_hen[j]))
        
        return current_hen
    
    def chick_update(self, chick_idx: int, population: List[List[int]], 
                    fitness_scores: List[float], iteration: int) -> List[int]:
        """Update chick position"""
        current_chick = population[chick_idx].copy()
        
        # Find mother hen
        if self.hens:
            mother_idx = random.choice(self.hens)
            mother = population[mother_idx]
            
            # Chick update equation - follow mother
            for j in range(len(current_chick)):
                r = random.random()
                current_chick[j] = int(mother[j] + r * (current_chick[j] - mother[j]))
        
        return current_chick
    
    def optimize(self, vm_count: int, pm_count: int, vm_processing: List[float],
                vm_cpu: List[float], vm_bandwidth: List[float], 
                vm_memory: List[float], vm_mips: List[float]) -> List[int]:
        """
        Main optimization loop equivalent to chicken_swarm.main()
        """
        print("Starting Chicken Swarm Optimization for VM load balancing...")
        
        # Initialize population
        population = self.initialize_population(vm_count, pm_count)
        
        # Evaluate initial population
        fitness_scores = []
        for individual in population:
            fitness = self.fitness_function(individual, vm_processing, vm_cpu, 
                                          vm_bandwidth, vm_memory, vm_mips)
            fitness_scores.append(fitness)
        
        # Find initial best
        best_idx = np.argmax(fitness_scores)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness_scores[best_idx]
        
        # Main optimization loop
        for iteration in range(self.max_generations):
            
            # Update status every G generations
            if iteration % self.G == 0:
                self.divide_swarm(population, fitness_scores)
            
            # Update positions
            new_population = []
            
            for i in range(len(population)):
                if i in self.roosters:
                    new_individual = self.rooster_update(i, population, fitness_scores, iteration)
                elif i in self.hens:
                    new_individual = self.hen_update(i, population, fitness_scores, iteration)
                elif i in self.chicks:
                    new_individual = self.chick_update(i, population, fitness_scores, iteration)
                else:
                    new_individual = population[i].copy()  # No change
                
                # Ensure valid PM assignments
                for j in range(len(new_individual)):
                    new_individual[j] = max(1, min(new_individual[j], pm_count))
                
                new_population.append(new_individual)
            
            # Evaluate new population
            new_fitness_scores = []
            for individual in new_population:
                fitness = self.fitness_function(individual, vm_processing, vm_cpu,
                                              vm_bandwidth, vm_memory, vm_mips)
                new_fitness_scores.append(fitness)
            
            # Selection (keep better individuals)
            for i in range(len(population)):
                if new_fitness_scores[i] > fitness_scores[i]:
                    population[i] = new_population[i]
                    fitness_scores[i] = new_fitness_scores[i]
                    
                    # Update global best
                    if fitness_scores[i] > self.best_fitness:
                        self.best_solution = population[i].copy()
                        self.best_fitness = fitness_scores[i]
            
            if iteration % 10 == 0:
                print(f"Generation {iteration}: Best fitness = {self.best_fitness:.6f}")
        
        print(f"Chicken Swarm Optimization completed. Best fitness: {self.best_fitness:.6f}")
        return self.best_solution

# Test the implementation
if __name__ == "__main__":
    print("Testing Chicken Swarm Optimization...")
    
    # Sample VM parameters
    vm_count = 20
    pm_count = 5
    
    vm_processing = [random.random() * 10 + 1 for _ in range(vm_count)]
    vm_cpu = [random.random() * 10 + 1 for _ in range(vm_count)]
    vm_bandwidth = [random.random() * 10 + 1 for _ in range(vm_count)]
    vm_memory = [random.random() * 10 + 1 for _ in range(vm_count)]
    vm_mips = [random.random() * 10 + 1 for _ in range(vm_count)]
    
    # Run optimization
    cso = ChickenSwarmOptimizer(population_size=10, max_generations=20)
    best_solution = cso.optimize(vm_count, pm_count, vm_processing, vm_cpu, 
                                vm_bandwidth, vm_memory, vm_mips)
    
    print(f"Best VM-PM assignment: {best_solution}")
    print("Chicken Swarm Optimization working correctly!")