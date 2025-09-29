# Cloud Simulation with SFDO-DRNN

## Overview
This is a research implementation that combines cloud simulation (CloudSim) with advanced machine learning techniques for enhanced cybersecurity analysis in cloud environments. The project was originally a Java-based NetBeans application but has been adapted for the Replit environment with a modern web interface.

## Current State
- **Status**: Fully functional web application running on port 5000
- **Frontend**: Python Flask web application with responsive HTML/CSS/JavaScript interface
- **Backend**: Java-based cloud simulation engine with ML algorithms
- **Data**: Bot-IoT network traffic dataset for cybersecurity analysis

## Recent Changes
- **September 29, 2025**: 
  - Migrated from Java GUI to web-based interface for Replit compatibility
  - Set up Flask application serving on port 5000
  - Created responsive web UI for simulation parameters and results
  - Configured workflow for automatic startup

## Project Architecture

### Technologies Used
- **SFDO**: SailFish-Dolphin Optimization Algorithm
- **DRNN**: Deep Recurrent Neural Networks  
- **FCM**: Fuzzy C-Means Clustering
- **SVM**: Support Vector Machines
- **ANFIS**: Adaptive Neuro-Fuzzy Inference System
- **CloudSim**: Cloud computing simulation framework

### File Structure
- `app.py` - Main Flask web application
- `templates/index.html` - Web interface template
- `74216/` - Original Java project directory
  - `src/` - Java source code
  - `jar files/` - Required Java libraries
  - `dataset/` - Bot-IoT network traffic data
  - `build/` - Compiled Java classes

### Key Features
1. **Simulation Parameters**: Configurable group size and training percentage
2. **Real-time Monitoring**: Live simulation logs and status updates
3. **Results Dashboard**: Performance metrics including accuracy, precision, recall, F1-score
4. **Dataset Analysis**: Bot-IoT network traffic processing

## User Preferences
- **Interface**: Modern web-based UI preferred over traditional Java Swing
- **Port**: Application must run on port 5000 for Replit compatibility
- **Display**: Clean, professional research-focused design
- **Responsiveness**: Real-time updates during simulation execution

## Deployment Configuration
- **Target**: Replit web hosting
- **Port**: 5000 (required for Replit)
- **Type**: Web application with autoscale deployment
- **Dependencies**: Python 3.11, Flask, Java runtime for backend processing

## Usage Instructions
1. Access the web interface at the deployed URL
2. Set simulation parameters (Group Size: 1-10, Training %: 50-90)
3. Click "START SIMULATION" to begin analysis
4. Monitor real-time logs during execution
5. View results dashboard upon completion

## Research Context
This application is designed for academic research in cloud security, combining multiple machine learning approaches for enhanced botnet detection in IoT network traffic within cloud computing environments.