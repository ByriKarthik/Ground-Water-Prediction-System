# 🌍 Groundwater Level Prediction using Deep Learning

An end-to-end Machine Learning web application that predicts groundwater levels using environmental and hydrological parameters. The system includes a trained deep learning model, a Flask-based REST API, and an interactive web interface for real-time predictions.

---

## 🚀 Project Highlights

- Built a **Deep Neural Network** for groundwater level prediction with **R² ≈ 0.98**
- Developed a **production-style Flask API** for real-time predictions
- Implemented **data preprocessing, scaling, and outlier handling**
- Added **dataset-driven realistic input validation**
- Designed an **interactive web dashboard** with Chart.js visualization
- Implemented **status classification** (Low / Normal / High)
- Ensured robust **error handling and validation**

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| R² Score | **0.98** |
| RMSE | ~9.9 |
| MAE | ~7.9 |

The model demonstrates strong predictive performance on environmental and hydrological data.

---

## 🧠 Features Used

- Rainfall (mm)  
- Soil Moisture (%)  
- Evaporation Rate (mm/day)  
- Recharge Rate (mm/year)  
- Well Yield (L/s)  
- Aquifer Thickness (m)  

---

## 🏗️ System Architecture

User Input (Web UI)
↓
Flask REST API (Validation + Scaling)
↓
Deep Learning Model (Prediction)
↓
Groundwater Level Output + Status
↓
Visualization (Chart.js)


---

## 🧰 Tech Stack

### Machine Learning
- Python
- TensorFlow / Keras
- NumPy, Pandas, Scikit-learn

### Backend
- Flask
- Flask-CORS

### Frontend
- HTML, CSS, JavaScript
- Chart.js

---

## 📂 Project Structure
```text
MLPROJECT/
│
├── app.py
├── train_model.py
├── test_model.py
├── requirements.txt
│
├── models/
│ ├── ml_model.keras
│ ├── data_scaler.pkl
│ └── y_scaler.pkl
│
├── data/
│ └── groundwater_dataset.csv
│
├── static/
│ ├── styles.css
│ ├── script.js
│ └── training_loss.png
│
├── templates/
│ └── index.html
```
---

## ▶️ Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```
### 2. Run Flask server
- python app.py
### 3. Open in browser
```bash
http://127.0.0.1:5000
```
### 🔍 How It Works
- User enters environmental parameters

- Inputs are validated against dataset-based realistic ranges

- Features are scaled using trained MinMax scalers

- Deep learning model predicts groundwater level

- Output is classified into Low / Normal / High

- Results are visualized using Chart.js


### 👨‍💻 Author

Karthik Byri

B.Tech CSE 
