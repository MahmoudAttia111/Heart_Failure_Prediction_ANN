# 💓 Heart Failure Prediction (ANN + Streamlit)
## 📌 Project Overview

This project aims to predict the risk of heart failure using an Artificial Neural Network (ANN) built with TensorFlow/Keras.
The trained model is deployed using Streamlit to create a user-friendly web application.

## 🛠️ Features

Preprocessing of dataset (handling missing values, normalization, encoding).

ANN model built using Keras (TensorFlow backend).

Evaluation metrics: Accuracy, F1-score, ROC Curve.

Interactive Streamlit Web App for predictions.

Deployment via Streamlit Cloud (free & easy).

## 📂 Project Structure

Heart-Failure-Prediction/
│
├── app.py                  # Streamlit web app  
├── heart_failure_model.keras # Trained ANN model  
├── scaler.pkl              # Saved StandardScaler (for normalization)  
├── requirements.txt        # Project dependencies  
├── README.md               # Project documentation  
└── notebook.ipynb          # (Optional) Training Jupyter Notebook  


## ⚙️ Installation & Setup
1. Clone Repository
git clone https://github.com/MahmoudAttia111/Heart_Failure_Prediction_ANN.git
cd heart-failure-prediction

2. Create Virtual Environment (recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Run Locally
streamlit run app.py


The app will open at: https://heartfailurepredictionann-gcl9oye7sddsd9cyc5uo7d.streamlit.app/

## 🧠 Model Details

ANN with:

Dense(64, activation='relu')

Dense(32, activation='relu')

Dense(1, activation='sigmoid')

Optimizer: Adam

Loss: Binary Crossentropy

EarlyStopping to prevent overfitting.

## 📌 Requirements
streamlit
tensorflow
scikit-learn
numpy
pandas
joblib
 
