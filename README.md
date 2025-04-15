# 🧬 Cancer Mortality Predictor

A data science project that analyzes and predicts cancer mortality rates using socio-economic, demographic, and healthcare-related features.

## 📌 Project Overview

This project explores regional cancer mortality rates using statistical modeling and machine learning techniques. It applies:
- Data preprocessing and feature engineering
- Outlier handling
- Correlation and significance analysis
- Linear Regression for prediction
- Data visualization for insights

---

## 📊 Dataset Description

The dataset (`cancer_reg.csv`) contains health-related statistics by US county. The target variable is:

- **TARGET_deathRate**: Average annual deaths due to cancer per 100,000 people.

### Key Features:
- `avgAnnCount`: Average annual cancer cases  
- `medianagefemale`, `medianagemale`: Median age of female/male population  
- `povertypercent`, `pctprivatecoverage`, `pctpubliccoverage`: Socio-economic indicators  
- `popest2015`: Population estimate  
- State and county identifiers  
- Other demographic and insurance-related columns  

---



## 🛠️ Features & Workflow

### ✅ Data Ingestion (`data_ingest.py`)
- Reads the dataset using `IngestData` class
- Handles different file encodings
- Displays available columns for reference

### ✅ Preprocessing & Cleaning (`data_processing.py`)
- Removes constant and near-constant columns
- One-hot encodes categorical features
- Fills missing values appropriately
- Splits the dataset into training/testing sets

### ✅ Outlier Handling
- Uses 3-sigma rule to cap outliers for numeric columns

### ✅ Feature Engineering (`feature_engineering.py`)
- Converts income brackets to numeric values
- One-hot encodes categorical data
- Prepares features for modeling

### ✅ Modeling (`regression_model.py`)
- Removes highly correlated features (correlation > 0.8)
- Fits a Linear Regression model using `statsmodels`
- Identifies important features using p-values
- Prints model performance and diagnostics

### ✅ Visualization (`visualize_data.py`)
- Distribution plots
- Boxplots for outliers
- Correlation heatmaps
- Skewness detection

---

## 🔧 Installation & Setup

### 1. Installation & Project Setup
git clone https://github.com/your-username/cancer-mortality-predictor.git<br>
cd cancer-mortality-predictor

### 2. Create and Activate a Virtual Environment
python -m venv venv
#### Windows:
venv\Scripts\activate
#### macOS/Linux:
source venv/bin/activate

### 2. Create and Activate a Virtual Environment
pip install pandas numpy matplotlib statsmodels scikit-learn<br>
python data_processing_test.py<br>
python regression_model.py<br>
python visualize_data.py