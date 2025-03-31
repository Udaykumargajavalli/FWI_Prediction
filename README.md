# FWI Prediction

This project focuses on predicting the Fire Weather Index (FWI) and serves as a sample training exercise.

## Project Overview

The Fire Weather Index (FWI) is a meteorological index used worldwide to estimate fire danger levels in forested areas. This project aims to develop a predictive model for FWI based on various weather parameters.

## Features

- **Data Preprocessing:** Handling missing values, encoding categorical variables, and normalizing data.
- **Data Cleaning:** Removing inconsistencies and ensuring data quality.
- **Model Training:** Developing and training machine learning models to predict FWI, with the final model saved as a `.pkl` file.
- **Flask Deployment:** Deploying the trained model using a Flask web application for real-time predictions.

## Installation

To set up the project environment, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Udaykumargajavalli/FWI_Prediction.git
2. Navigate to the project directory:
   ```bash
   cd FWI_Prediction
3. Install the required packages:
   ```bash
   pip install -r requirements.txt

## Usage
1. Run the Flask application:
   ```bash
   python app.py

2. Access the web interface: Open a web browser and navigate to http://127.0.0.1:5000/ to use the application.

## File Structure
1. app.py: Main script to run the Flask application.

2. requirements.txt: Lists all the Python packages required to run the project.

3. templates/: Directory containing HTML templates for the web interface.
