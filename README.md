# ABC Hotels – Booking Cancellation Prediction

This project predicts the probability of booking cancellations for ABC Hotels using supervised machine learning (dense feedforward neural networks) in Python.
The goal is to identify high-risk bookings (those likely to be cancelled) so hotel management can implement targeted retention strategies, such as special offers or personalized communications, to reduce cancellations and improve occupancy rates.

## Folder Structure

- **data:** contains the provided dataset

- **notebooks:** contains Jupyter notebooks for Preliminary Results, and Final Report.

- **reports:** includes the analytic plan, preliminary results, final report, and final code appendix.

- **src:** contains modular Python scripts supporting the machine learning pipeline:
  - `utils.py`: helper functions for reproducibility and device setup  
  - `feature_engineering.py`: data cleaning and feature engineering steps  
  - `data_preprocessing.py`: data encoding, scaling, and tensor conversion  
  - `model_training.py`: model architecture definitions, dataloaders, and training loop  
  - `model_evaluation.py`: performance evaluation metrics and plots  
  - `model_comparison.py`: ROC/AUC, calibration, and comparative visualizations  

- **main.py:** the central script that runs the **entire end-to-end pipeline**:
  1. Loads and preprocesses the data  
  2. Engineers new features  
  3. Trains both an overfitting and a regularized neural network model  
  4. Evaluates and compares model performance using validation metrics and plots

- **requirements.txt:** lists the Python libraries used in this project.

- **README.md:** project overview and usage instructions.

## Project Overview

**Business Need**

ABC Hotels wants to predict which bookings are most likely to be cancelled. By understanding the cancellation risk for each reservation, the hotel can take proactive actions to improve guest retention and minimize lost revenue.

**Analytic Approach**

This project follows a full machine learning lifecycle, including:

- Analytic planning (business problem definition, data assessment, and feature proposal)

- Model development and preliminary evaluation

- Final model selection, evaluation, and business recommendations

A supervised classification approach was implemented using dense feedforward neural networks. Models were trained and evaluated using ROC curves, AUC, calibration plots, and learning curve analyses.


## Usage Instructions

#### 1. Download or Clone the Project

Before setting up the environment, download the project files to your local machine.

You can Clone using Git. 

#### 2. Set up environment
Make sure you have [Anaconda](https://www.anaconda.com/) installed.  
Create and activate a new Conda environment (you can name it hotel_env):

**conda create -n hotel_env python=3.12**

**conda activate hotel_env**

#### 3. Open the Project in the Environment

Once the environment is created and activated, navigate to the project folder in your terminal:

**cd path/to/your/project/folder**

Then install the required dependencies:

**pip install -r requirements.txt**

#### 2. Run the full pipeline
From the root directory of the project (where main.py is located):

**python main.py**

This will:

- Load and preprocess the dataset

- Train both neural network models

- Generate learning curves, confusion matrices, ROC curves, and calibration plots

#### 3. View reports and notebooks

If you prefer a detailed, documented walkthrough:

Open the Jupyter notebooks in the notebooks/ folder for model development and analysis steps.

Final deliverables and results can be viewed in the reports/ folder.




