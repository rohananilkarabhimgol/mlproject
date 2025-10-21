## End to End Machine Learning Project
# Student Performance Prediction System

The **Student Performance Prediction System** is an intelligent machine learning application designed to predict student math scores based on academic and demographic factors. It can perform various actions such as:
- Processing and transforming educational data
- Training multiple machine learning models automatically  
- Selecting the best performing algorithm
- Providing predictions through a web interface
- Handling both numerical and categorical data
- Generating comprehensive logs and error reports

---

## Prerequisites

To use this project, ensure you have the following:
1. **Python** (version 3.8 or above)
2. A stable internet connection for package installation
3. **Git** for version control and cloning the repository

## Installation

### Step 1: Clone the Repository

Run the following command to clone the project to your local machine:

```bash
git clone https://github.com/your-username/student-performance-predictor.git
cd student-performance-predictor
 ```

### Step 2: Create Virtual Environment 
Set up a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Step 3: Install Required Libraries
Run the following commands to install all necessary Python modules:

```
pip install -r requirements.txt
```

Key Features
Data Ingestion & Processing
Automatically loads and splits student data into training and testing sets
Handles missing values and data validation

Machine Learning Pipeline
Tests 9+ different algorithms automatically
Performs hyperparameter tuning for optimal performance
Selects the best model based on R² score

Web Interface
Beautiful, responsive UI
Real-time predictions through web forms
Professional result display

Error Handling & Logging
Comprehensive custom exception handling
Detailed execution logs for debugging
Robust error recovery mechanisms

Start the Web Application
```bash
python app.py
```
