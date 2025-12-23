# Driver Behaviour Classification using ANN
This project implements an **Artificial Neural Network (ANN)** to classify **driver behaviour** using vehicle telematics data.  
The system detects **unsafe driving behaviour** based on engineered features derived from raw sensor readings.
A **Streamlit web application** is included to demonstrate real-time prediction.

---
## Problem Statement
Unsafe driving behaviours such as harsh acceleration, erratic speed changes, and fatigued driving significantly increase accident risk.
The objective of this project is to:
- Analyse vehicle telematics data
- Engineer meaningful driving features
- Train an ANN model to classify driving behaviour
- Deploy the trained model using a simple web interface
  
---
## Dataset Description
The dataset contains **raw telematics sensor data** collected from vehicles.

### Sensor variables used:
- Vehicle speed  
- Engine RPM  
- Longitudinal acceleration (ACCELERATION X)  
- Throttle position  
- Idling status  

Raw sensor readings are aggregated into **30-second driving windows** to represent short driving behaviour segments.

---

## Feature Engineering
For each 30-second window, the following features are derived:

```
|---------------------|------------------------------------|
| Feature             | Description                        |
|---------------------|------------------------------------|
| avg\_speed          | Mean vehicle speed                 |
| speed\_std          | Speed variability                  |
| max\_speed          | Maximum speed                      |
| avg\_acceleration   | Mean longitudinal acceleration     |
| throttle\_variance  | Variance of throttle input         |
| rpm\_mean           | Average engine RPM                 |
| idle\_time\_ratio   | Ratio of idle time                 |
| hard\_brake\_count  | Count of harsh braking events      |
| hard\_accel\_count  | Count of harsh acceleration events |
|---------------------|------------------------------------|

```

## Label Definition
Driving behaviour is classified into two categories:



- **0 – Normal Driving**

- **1 – Unsafe Driving**

Unsafe driving includes:
- Aggressive behaviour  
- Distracted behaviour  
- Fatigued behaviour  

Due to class imbalance, all unsafe behaviours are merged into a single class.

---

## 🧠 Model Architecture



An **Artificial Neural Network (ANN)** is implemented using Keras.

Input Layer (9 features)

Dense (9 units, ReLU) input

Dense (32 units, ReLU) hidden layer #1

Dense (16 units, ReLU) hidden layer #2

Dense (1 unit, Sigmoid) output layer

---

### Training Configuration
- Optimiser: Adam  
- Loss Function: Binary Cross-Entropy  
- Class imbalance handled using \*\*class weights\*\*  
- Early stopping is used to prevent overfitting  

---

## Class Imbalance Handling

Unsafe driving events are rare.

To address this:
- Class weights are applied during training
- A **custom decision threshold (0.3)** is used instead of the default 0.5
- This improves recall for unsafe driving detection

---

##  Model Performance (Test Set)



Using a decision threshold of **0.3**:

**Confusion Matrix**
```
[ [133 32]
  [ 7 13 ] ]
```
**Key Metrics**
- Accuracy: **79%**
- Unsafe driving recall: **65%**
- Model prioritises risk detection over raw accuracy

---

## Streamlit Web Application
A lightweight Streamlit application demonstrates real-time prediction.
### Features:
- Manual input of driving features
- Probability-based unsafe driving prediction
- Threshold-based decision logic
- Simple and user-friendly interface

---

### directory:

```
DriverBehaviourClassification/
│
├── dataset/
│   └── Telematicsdata.csv
│
├── model/
│   ├── driving\_model.keras
│   └── scaler.pkl
│
├── app.py
├── README.md
└── requirements.txt
```

---

## How to Run the Project


### Clone the repository:
  
  git clone https://github.com/chrisjamesranni/DriverBehaviourClassification.git

  cd DriverBehaviourClassification

### Install dependencies:

  pip install -r requirements.txt

### Launch the Streamlit app:

  streamlit run app.py

---

## Technologies Used
  Python, TensorFlow / Keras, Pandas, NumPy, Scikit-learn, Streamlit

---

## Future Improvements
  Temporal modelling using LSTM/GRU
  
  Real fatigue-labelled datasets
  
  Batch prediction via CSV upload  
  
  Deployment on Streamlit Cloud





