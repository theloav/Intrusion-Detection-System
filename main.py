import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN optimization warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data, preprocess_data
from model import build_model

# Load the dataset
file_paths = [
    'data/Monday-WorkingHours.pcap_ISCX.csv',
    'data/Tuesday-WorkingHours.pcap_ISCX.csv',
    'data/Wednesday-workingHours.pcap_ISCX.csv',
    'data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'data/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'data/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
]

combined_df = load_data(file_paths)
combined_df, label_encoder = preprocess_data(combined_df)

# Check for null and infinite values
print("Null values count:")
print(combined_df.isnull().sum())
print("\nInfinite values count:")
print(np.isinf(combined_df).sum())

# Replace infinite values with NaN, then fill NaN with the column mean
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
combined_df.fillna(combined_df.mean(), inplace=True)

# Select features and labels
X = combined_df.drop('Label', axis=1)
y = combined_df['Label']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM [samples, time steps, features]
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_encoded, test_size=0.2, random_state=42)

# Build and train the model
model = build_model((X_train.shape[1], X_train.shape[2]), y_encoded.shape[1])
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot accuracy and loss
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Save the model
model.save('your_model_name.keras')
