import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Expanded dataset with 4 flex sensors and 30 gestures
data = pd.DataFrame([
    # ---- Hello ----
    {'flex1': 200, 'flex2': 210, 'flex3': 220, 'flex4': 230, 'accel_x': 0.1, 'accel_y': 0.0, 'accel_z': 1.0, 'gyro_x': 0.01, 'gyro_y': 0.02, 'gyro_z': 0.03, 'label': 'Hello'},
    {'flex1': 202, 'flex2': 208, 'flex3': 215, 'flex4': 225, 'accel_x': 0.11, 'accel_y': -0.01, 'accel_z': 1.01, 'gyro_x': 0.015, 'gyro_y': 0.018, 'gyro_z': 0.025, 'label': 'Hello'},
    {'flex1': 198, 'flex2': 212, 'flex3': 218, 'flex4': 220, 'accel_x': 0.09, 'accel_y': 0.02, 'accel_z': 0.99, 'gyro_x': 0.012, 'gyro_y': 0.021, 'gyro_z': 0.03, 'label': 'Hello'},

    # ---- Goodbye ----
    {'flex1': 210, 'flex2': 220, 'flex3': 230, 'flex4': 240, 'accel_x': 0.15, 'accel_y': 0.05, 'accel_z': 1.2, 'gyro_x': 0.02, 'gyro_y': 0.02, 'gyro_z': 0.04, 'label': 'Goodbye'},
    {'flex1': 213, 'flex2': 223, 'flex3': 233, 'flex4': 243, 'accel_x': 0.14, 'accel_y': 0.06, 'accel_z': 1.15, 'gyro_x': 0.021, 'gyro_y': 0.019, 'gyro_z': 0.03, 'label': 'Goodbye'},

    # ---- Yes ----
    {'flex1': 180, 'flex2': 190, 'flex3': 200, 'flex4': 210, 'accel_x': -0.2, 'accel_y': 0.1, 'accel_z': 0.9, 'gyro_x': 0.04, 'gyro_y': 0.01, 'gyro_z': 0.01, 'label': 'Yes'},
    {'flex1': 178, 'flex2': 192, 'flex3': 202, 'flex4': 212, 'accel_x': -0.22, 'accel_y': 0.12, 'accel_z': 0.88, 'gyro_x': 0.042, 'gyro_y': 0.015, 'gyro_z': 0.02, 'label': 'Yes'},
    
    # ---- No ----
    {'flex1': 220, 'flex2': 230, 'flex3': 240, 'flex4': 250, 'accel_x': 0.0, 'accel_y': -0.1, 'accel_z': 1.1, 'gyro_x': 0.03, 'gyro_y': 0.03, 'gyro_z': 0.02, 'label': 'No'},
    {'flex1': 222, 'flex2': 232, 'flex3': 242, 'flex4': 252, 'accel_x': 0.02, 'accel_y': -0.11, 'accel_z': 1.09, 'gyro_x': 0.028, 'gyro_y': 0.032, 'gyro_z': 0.023, 'label': 'No'},
    
    # ---- Please ----
    {'flex1': 240, 'flex2': 250, 'flex3': 260, 'flex4': 270, 'accel_x': 0.1, 'accel_y': 0.1, 'accel_z': 0.8, 'gyro_x': 0.05, 'gyro_y': 0.03, 'gyro_z': 0.04, 'label': 'Please'},
    {'flex1': 242, 'flex2': 252, 'flex3': 262, 'flex4': 272, 'accel_x': 0.09, 'accel_y': 0.08, 'accel_z': 0.85, 'gyro_x': 0.052, 'gyro_y': 0.029, 'gyro_z': 0.045, 'label': 'Please'},

    # ---- Thank You ----
    {'flex1': 200, 'flex2': 210, 'flex3': 220, 'flex4': 230, 'accel_x': 0.0, 'accel_y': -0.1, 'accel_z': 0.9, 'gyro_x': 0.02, 'gyro_y': 0.03, 'gyro_z': 0.02, 'label': 'Thank You'},
    {'flex1': 198, 'flex2': 208, 'flex3': 218, 'flex4': 228, 'accel_x': -0.02, 'accel_y': -0.1, 'accel_z': 0.92, 'gyro_x': 0.025, 'gyro_y': 0.02, 'gyro_z': 0.04, 'label': 'Thank You'},

    # ---- Sorry ----
    {'flex1': 250, 'flex2': 260, 'flex3': 270, 'flex4': 280, 'accel_x': 0.1, 'accel_y': 0.1, 'accel_z': 0.85, 'gyro_x': 0.03, 'gyro_y': 0.03, 'gyro_z': 0.03, 'label': 'Sorry'},
    {'flex1': 253, 'flex2': 263, 'flex3': 273, 'flex4': 283, 'accel_x': 0.12, 'accel_y': 0.05, 'accel_z': 0.82, 'gyro_x': 0.035, 'gyro_y': 0.031, 'gyro_z': 0.032, 'label': 'Sorry'},

    # ---- Help ----
    {'flex1': 210, 'flex2': 220, 'flex3': 230, 'flex4': 240, 'accel_x': -0.1, 'accel_y': 0.05, 'accel_z': 0.8, 'gyro_x': 0.02, 'gyro_y': 0.02, 'gyro_z': 0.04, 'label': 'Help'},
    {'flex1': 212, 'flex2': 222, 'flex3': 232, 'flex4': 242, 'accel_x': -0.12, 'accel_y': 0.07, 'accel_z': 0.75, 'gyro_x': 0.021, 'gyro_y': 0.023, 'gyro_z': 0.05, 'label': 'Help'},

    # ---- Stop ----
    {'flex1': 250, 'flex2': 260, 'flex3': 270, 'flex4': 280, 'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0, 'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0, 'label': 'Stop'},
    {'flex1': 252, 'flex2': 258, 'flex3': 268, 'flex4': 278, 'accel_x': 0.01, 'accel_y': -0.01, 'accel_z': 0.01, 'gyro_x': 0.001, 'gyro_y': 0.001, 'gyro_z': 0.001, 'label': 'Stop'},
])

# Train model
X = data.drop(columns=['label'])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, 'model.pkl')

# Evaluate
preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
