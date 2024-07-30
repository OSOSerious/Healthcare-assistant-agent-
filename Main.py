import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime

class EnvironmentalMonitoringAgent:
    def __init__(self, sensor_data):
        self.sensor_data = sensor_data
        self.model = IsolationForest(contamination=0.1)
    
    def collect_data(self):
        # Assume sensor_data is continuously updated
        self.data = pd.DataFrame(self.sensor_data)
    
    def detect_anomalies(self):
        X = self.data[['Temperature', 'Humidity', 'Air_Quality']]
        self.model.fit(X)
        self.data['Anomaly'] = self.model.predict(X)
        anomalies = self.data[self.data['Anomaly'] == -1]
        return anomalies
    
    def analyze_trends(self):
        self.data['Date'] = pd.to_datetime(self.data['Timestamp'])
        self.data.set_index('Date', inplace=True)
        trends = self.data.resample('D').mean()
        return trends
    
    def send_alerts(self, anomalies):
        alerts = anomalies[['Timestamp', 'Temperature', 'Humidity', 'Air_Quality']]
        # Implement alert sending mechanism (e.g., email, SMS)
        return alerts
    
    def generate_report(self):
        trends = self.analyze_trends()
        anomalies = self.detect_anomalies()
        alerts = self.send_alerts(anomalies)
        report = {
            'Trends': trends,
            'Anomalies': anomalies,
            'Alerts': alerts
        }
        return report
