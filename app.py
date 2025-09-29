#!/usr/bin/env python3
"""
Web interface for Cloud Simulation with SFDO-DRNN
This serves as a frontend to the Python-based cloud simulation application.
"""
import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
import threading
import time
import traceback
from cloud_simulation import CloudSimulation

app = Flask(__name__)

class SimulationRunner:
    def __init__(self):
        self.is_running = False
        self.results = {}
        self.logs = []
    
    def add_log(self, message):
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        if len(self.logs) > 100:  # Keep only last 100 logs
            self.logs.pop(0)
    
    def run_simulation(self, group_size=4, training_percentage=80):
        """Run actual cloud simulation with Python algorithms"""
        self.is_running = True
        self.add_log(f"Starting simulation with group_size={group_size}, training_percentage={training_percentage}")
        
        try:
            # Create and run the actual cloud simulation
            simulation = CloudSimulation(group_size=group_size, training_percentage=training_percentage)
            
            self.add_log("Initializing cloud simulation environment...")
            self.add_log("Setting up VM parameters (10 PMs, 50 VMs)...")
            
            self.add_log("Loading Bot-IoT dataset...")
            self.add_log("Preprocessing network traffic data...")
            
            self.add_log("Performing feature fusion with FCM clustering...")
            self.add_log("Grouping features into clusters...")
            
            self.add_log("Running cloud load balancing optimization...")
            self.add_log("Applying Chicken Swarm Optimization...")
            
            self.add_log("Starting machine learning algorithms...")
            self.add_log("1. Training FCM-ANN classifier...")
            
            self.add_log("2. Training ANFIS classifier...")
            
            self.add_log("3. Training SVM classifier...")
            
            self.add_log("4. Training SFDO-DRNN classifier...")
            self.add_log("   - SailFish-Dolphin optimization...")
            self.add_log("   - Deep Recurrent Neural Network...")
            
            self.add_log("Running full simulation...")
            
            # Run the actual simulation
            results = simulation.run_simulation()
            
            # Extract results in the format expected by frontend
            algorithms = results['algorithms']
            best_accuracy = max(results['accuracy']) * 100
            best_precision = max(results['precision']) * 100
            best_recall = max(results['recall']) * 100
            best_f1 = max(results['f1_score']) * 100
            avg_fpr = sum(results['fpr']) / len(results['fpr']) * 100
            
            self.results = {
                'accuracy': round(best_accuracy, 2),
                'precision': round(best_precision, 2),
                'recall': round(best_recall, 2),
                'f1_score': round(best_f1, 2),
                'fpr': round(avg_fpr, 2),
                'execution_time': round(results['execution_time'], 2),
                'group_size': group_size,
                'training_percentage': training_percentage,
                'total_samples': results['total_samples'],
                'training_samples': results['training_samples'],
                'testing_samples': results['testing_samples'],
                'algorithms': algorithms,
                'detailed_results': {
                    'fcm_ann': {
                        'accuracy': round(results['accuracy'][0] * 100, 2),
                        'precision': round(results['precision'][0] * 100, 2),
                        'recall': round(results['recall'][0] * 100, 2),
                        'fpr': round(results['fpr'][0] * 100, 2)
                    },
                    'anfis': {
                        'accuracy': round(results['accuracy'][1] * 100, 2),
                        'precision': round(results['precision'][1] * 100, 2),
                        'recall': round(results['recall'][1] * 100, 2),
                        'fpr': round(results['fpr'][1] * 100, 2)
                    },
                    'svm': {
                        'accuracy': round(results['accuracy'][2] * 100, 2),
                        'precision': round(results['precision'][2] * 100, 2),
                        'recall': round(results['recall'][2] * 100, 2),
                        'fpr': round(results['fpr'][2] * 100, 2)
                    },
                    'sfdo_drnn': {
                        'accuracy': round(results['accuracy'][3] * 100, 2),
                        'precision': round(results['precision'][3] * 100, 2),
                        'recall': round(results['recall'][3] * 100, 2),
                        'fpr': round(results['fpr'][3] * 100, 2)
                    }
                }
            }
            
            self.add_log("Simulation completed successfully!")
            self.add_log(f"Best accuracy achieved: {best_accuracy:.2f}%")
            self.add_log(f"Processing completed in {results['execution_time']:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Error during simulation: {str(e)}"
            self.add_log(error_msg)
            self.add_log(f"Traceback: {traceback.format_exc()}")
            print(f"Simulation error: {e}")
            print(traceback.format_exc())
        finally:
            self.is_running = False
    
    def get_status(self):
        return {
            'is_running': self.is_running,
            'results': self.results,
            'logs': self.logs[-10:]  # Return last 10 logs
        }

# Global simulation runner
sim_runner = SimulationRunner()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/start_simulation', methods=['POST'])
def start_simulation():
    """Start a new simulation"""
    if sim_runner.is_running:
        return jsonify({'error': 'Simulation already running'}), 400
    
    data = request.get_json()
    group_size = data.get('group_size', 4)
    training_percentage = data.get('training_percentage', 80)
    
    # Start simulation in background thread
    thread = threading.Thread(target=sim_runner.run_simulation, args=(group_size, training_percentage))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Simulation started'})

@app.route('/api/status')
def get_status():
    """Get simulation status"""
    return jsonify(sim_runner.get_status())

@app.route('/api/results')
def get_results():
    """Get simulation results"""
    return jsonify(sim_runner.results)

@app.route('/dataset')
def view_dataset():
    """View dataset information"""
    dataset_info = {
        'name': 'Bot-IoT Dataset',
        'description': 'Network traffic dataset for IoT botnet detection',
        'samples': 5000,
        'features': 42,
        'classes': ['Normal', 'Attack'],
        'file_path': '74216/dataset/Bot-Iot.csv'
    }
    return jsonify(dataset_info)

if __name__ == '__main__':
    # Ensure we bind to all interfaces on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)