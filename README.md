# ⚡ EV-Lution: EV Charging Demand Prediction & Smart Station Recommendation

<p align="center">
  <img src="assets/Logo.png" alt="EV-Lution Logo" width="180"/>
</p>

<p align="center">
  <b>An intelligent EV charging assistant that predicts charging station utilization using a Temporal Convolutional Network (TCN) and recommends the best charging station based on predicted congestion, distance, and charging cost.</b>
</p>

---

## 🌐 Live Demo

**🚀 Deployed Application:**  
<br>
[https://ev-lution-ml-1.onrender.com/](https://ev-lution-ml-1.onrender.com/)

---

## 📖 Overview

With the rapid adoption of Electric Vehicles (EVs), charging stations often experience congestion and long waiting times. EV-Lution helps EV users make informed charging decisions by:

- Predicting future charging station utilization using a deep learning model.
- Estimating charging demand.
- Recommending the most suitable charging station.
- Visualizing charging stations on an interactive map.
- Providing analytics and occupancy statistics.

The application combines Machine Learning with an interactive Streamlit dashboard to provide an end-to-end intelligent charging recommendation system.

---

## ✨ Features

### 🔮 Demand Prediction
- Predicts future charging station utilization using a trained Temporal Convolutional Network (TCN).
- Supports multi-hour forecasting.
- Displays utilization and demand trends.

### 📍 Interactive Station Map
- Interactive Folium-based map.
- Click anywhere to set your location.
- Automatically identifies the nearest charging station.
- Color-coded stations based on station type.

### ⭐ Smart Recommendation System
Recommends charging stations by considering:

- Predicted congestion
- Distance from user
- Charging cost
- Station type
- Number of chargers

A weighted scoring mechanism helps users select the best charging station.

### 📊 Analytics Dashboard

Visualizes:

- Station utilization
- Vacancy percentage
- Charging demand
- Average utilization by station type
- Charger availability

---

# 🧠 Machine Learning Model

The forecasting model is built using a **Temporal Convolutional Network (TCN)**.

### Why TCN?

Compared to traditional recurrent models such as LSTMs, TCNs provide:

- Parallel computation
- Stable gradients
- Long-range temporal dependency learning
- Faster training
- Better scalability

The trained model predicts future charging station utilization based on historical charging patterns.

---

# 🛠 Tech Stack

## Frontend

- Streamlit
- HTML/CSS
- Plotly
- Folium

## Machine Learning

- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Pandas

## Model Artifacts

- Trained TCN Model (.keras)
- StandardScaler
- Label Encoder

---

# 📂 Project Structure

```text
EV-Lution/
│
├── assets/
│   ├── Background.png
│   └── Logo.png
│
├── data/
│   ├── history.csv
│   └── ev_stations.csv
│
├── models/
│   ├── tcn_model_final.keras
│   ├── scaler.pkl
│   ├── seq_feature_cols.pkl
│   └── station_label_encoder.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Akshatmishra1211/EV-Lution-ML/
```

Move into the project

```bash
cd ev-lution-ml
```

Create a virtual environment

### Windows

```bash
python -m venv .venv
```

Activate it

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📈 Recommendation Score

Each charging station is ranked using a weighted score based on:

- Congestion
- Distance
- Charging Price

Lower score indicates a better recommendation.

---

# 📊 Dashboard Modules

### 🗺 Map

- Interactive charging station map
- User location selection
- Nearest station identification

### 🔮 Prediction

- Future utilization prediction
- Demand estimation
- Cost estimation

### ⭐ Recommendations

- Station ranking
- Congestion analysis
- Distance-aware recommendations

### 📊 Statistics

- Vacancy percentage
- Utilization trends
- Station comparison

---

# 🚀 Future Improvements

- Live traffic integration
- Real-time charging station API
- Dynamic electricity pricing
- Weather-aware prediction
- User authentication
- Reservation system
- Mobile application
- Route optimization using Maps API

---

# 📸 Screenshots

<img width="1892" height="870" alt="ev-1" src="https://github.com/user-attachments/assets/de970280-558d-44cf-8842-cf67d59c732f" />
<br><br>
<img width="1873" height="857" alt="ev-2" src="https://github.com/user-attachments/assets/c2550c6c-6c09-4b92-9a63-d01587231221" />
<br><br>
<img width="1832" height="867" alt="ev-3" src="https://github.com/user-attachments/assets/2e4071bc-a826-483e-a150-74560e65c691" />
<br><br>
<img width="1837" height="812" alt="ev-5" src="https://github.com/user-attachments/assets/c908b8c0-6926-4696-8b59-fda1d572d94a" />
<br><br>
<img width="1845" height="837" alt="ev-6" src="https://github.com/user-attachments/assets/416ce82f-d32d-4738-b805-c384ce725cdd" />

---

# ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub!
