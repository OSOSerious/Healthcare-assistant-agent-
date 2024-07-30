import pandas as pd
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class HealthcareAssistantAgent:
    def __init__(self, patient_records, appointment_data):
        self.patient_records = patient_records
        self.appointment_data = appointment_data
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()
    
    def manage_records(self, new_record):
        self.patient_records = self.patient_records.append(new_record, ignore_index=True)
    
    def schedule_appointment(self, patient_id, preferred_time):
        available_slots = self.appointment_data[self.appointment_data['Available'] == True]
        for index, slot in available_slots.iterrows():
            if slot['Time'] >= preferred_time:
                self.appointment_data.at[index, 'Available'] = False
                self.appointment_data.at[index, 'Patient_ID'] = patient_id
                return slot['Time']
        return "No available slots"
    
    def send_medication_reminders(self, patient_id):
        patient = self.patient_records[self.patient_records['Patient_ID'] == patient_id]
        medication_schedule = patient['Medication_Schedule'].values[0]
        reminders = []
        for med, time in medication_schedule.items():
            reminder_time = datetime.now() + timedelta(minutes=time)
            reminders.append({'Patient_ID': patient_id, 'Medication': med, 'Reminder_Time': reminder_time})
        return reminders
    
    def symptom_checker(self, symptoms):
        # Dummy symptom checker model for demonstration
        symptoms_vector = self.vectorizer.transform([symptoms])
        diagnosis = self.model.predict(symptoms_vector)
        return diagnosis
    
    def healthcare_analytics(self):
        analytics = self.patient_records.describe()
        return analytics

    def run_healthcare_assistant(self, new_record, patient_id, preferred_time, symptoms):
        self.manage_records(new_record)
        appointment = self.schedule_appointment(patient_id, preferred_time)
        reminders = self.send_medication_reminders(patient_id)
        diagnosis = self.symptom_checker(symptoms)
        analytics = self.healthcare_analytics()
        return {'Appointment': appointment, 'Reminders': reminders, 'Diagnosis': diagnosis, 'Analytics': analytics}
