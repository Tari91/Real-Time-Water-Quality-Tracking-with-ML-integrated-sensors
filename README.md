Real-Time Water Quality Tracking with ML-Integrated Sensors
Overview

This project demonstrates an end-to-end real-time water quality monitoring system using synthetic data and machine learning. The system simulates ML-integrated sensors that continuously monitor water parameters, predict a Water Quality Index (WQI), and detect contamination events in real time.

It is designed for academic, research, and prototype IoT applications where real sensor data may be limited or expensive.

Features

Synthetic water quality data generation

Real-time sensor data streaming simulation

Anomaly detection using Isolation Forest

Water Quality Index (WQI) prediction using Random Forest

ML-integrated virtual sensors

Edge and IoT friendly architecture

Parameters Monitored

pH

Turbidity (NTU)

Dissolved Oxygen (mg/L)

Temperature (°C)

Electrical Conductivity (µS/cm)

Nitrate (mg/L)

Project Structure
.
├── real_time_water_quality_ml.py   # Main Python application
└── README.md                       # Project documentation

Requirements

Python 3.8 or higher

numpy

pandas

scikit-learn

Install Dependencies
pip install numpy pandas scikit-learn

How to Run

Execute the main script:

python real_time_water_quality_ml.py


To stop real-time monitoring, press CTRL + C.

Output

The system continuously prints:

Live sensor readings

Predicted Water Quality Index (WQI)

Water status (NORMAL or CONTAMINATION ALERT)

Example Output
Sensor Data: {'pH': 7.1, 'turbidity': 2.9, 'dissolved_oxygen': 8.2, ...}
WQI: 86.45 | Status: NORMAL
------------------------------------------------------------
Sensor Data: {'pH': 5.3, 'turbidity': 22.1, 'dissolved_oxygen': 4.8, ...}
WQI: 42.11 | Status: CONTAMINATION ALERT

Machine Learning Models Used
Task	Model
Anomaly Detection	Isolation Forest
WQI Prediction	Random Forest Regressor
Use Cases

Smart water distribution systems

River and lake pollution monitoring

Aquaculture health monitoring

Industrial wastewater compliance

Academic and research projects

Future Enhancements

GAN-based synthetic time-series data generation

LSTM / Transformer-based forecasting

IoT integration using MQTT or Kafka

Real-time dashboards (Streamlit / Grafana)

Edge deployment on Raspberry Pi or ESP32

Explainable AI for regulatory reporting

License

This project is provided for educational and research purposes.

Author

william Tarinabo,  williamtarinabo@gmail.com
