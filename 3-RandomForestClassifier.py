# -----------------------------
# Credit Card Fraud Detection Example
# -----------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Simulate a near real-time financial dataset
np.random.seed(42)
data_size = 10000

data = pd.DataFrame({
    'transaction_amount': np.random.uniform(10, 2000, data_size),
    'transaction_hour': np.random.randint(0, 24, data_size),
    'customer_age': np.random.randint(18, 70, data_size),
    'merchant_category': np.random.randint(1, 10, data_size),
    'is_international': np.random.randint(0, 2, data_size),
    'num_prev_transactions': np.random.randint(0, 20, data_size),
})

# Create a target label: fraud more likely for large, late-night, international transactions
data['is_fraud'] = np.where(
    (data['transaction_amount'] > 1500) &
    (data['transaction_hour'].isin([0,1,2,3])) &
    (data['is_international'] == 1), 1, 0
)

# Split features and target
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

'''By setting random_state, you ensure reproducibility: every time you run the code with the same seed, you get the exact same random behavior.

Original dataset: 80% Positive, 20% Negative

Training set: 80% Positive, 20% Negative

Test set: 80% Positive, 20% Negative

This stratified split preserves the ratio'''

# Feature scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
'''fit_transform() does two things: 
fit → Compute the mean and standard deviation from X_train 
transform → Apply the scaling to X_train'''


# Train a classification model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_scaled, y_train)
'''Creates 100 decision trees (n_estimators) using 
random subsets of the training data and features.'''

# Evaluate performance
y_pred = classifier.predict(X_test_scaled)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Predict a near real-time transaction
# Example: A 25-year-old user makes an international $1800 transaction at 2 AM
new_transaction = [[1800, 2, 25, 4, 1, 5]]
prediction = classifier.predict(sc.transform(new_transaction))

print("\nNew Transaction Prediction:", "FRAUD" if prediction[0] == 1 else "LEGITIMATE")
