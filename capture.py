import os
import pyshark
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import load_model
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Function to capture live network traffic
def capture_live_traffic(interface='Wi-Fi', capture_duration=10):
    """Capture live network traffic using pyshark."""
    print("Capturing network traffic...")
    capture = pyshark.LiveCapture(interface=interface)
    capture.sniff(timeout=capture_duration)
    
    # Extract important data from the packets
    packet_data = []
    for packet in capture:
        try:
            packet_data.append({
                'src_ip': packet.ip.src,       # Source IP
                'dst_ip': packet.ip.dst,       # Destination IP
                'protocol': packet.transport_layer,  # Protocol
                'length': packet.length        # Packet length
            })
        except AttributeError:
            # Skip packets without the required fields
            continue
    
    # Convert to a pandas DataFrame
    df = pd.DataFrame(packet_data)
    print(f"Captured {len(df)} packets")
    return df

# Function to preprocess the live captured data
def preprocess_live_data(df, scaler):
    """Preprocess live captured data: scale and reshape it."""
    # Replace missing values with 0 (if any)
    df.fillna(0, inplace=True)

    # Only select numerical features for scaling
    features = ['length']  # You can add more features if available

    # Scale the data using the same scaler used during training
    scaled_data = scaler.transform(df[features])

    # Reshape the data for LSTM [samples, time steps, features]
    reshaped_data = scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1])
    
    return reshaped_data

# Main function
def main():
    # Step 1: Load the trained model
    model = load_model('your_model_name.keras')  # Ensure this is the correct path to your trained model
    
    # Step 2: Load or initialize the scaler (use the one from training)
    # Assuming you had already saved or defined the same scaler during training
    scaler = StandardScaler()
    
    # Step 3: Capture live traffic
    live_traffic_df = capture_live_traffic(interface='Wi-Fi', capture_duration=10)  # Adjust the interface to your OS
    
    # Check if packets were captured
    if live_traffic_df.empty:
        print("No packets captured.")
        return
    
    # Step 4: Preprocess the captured data
    preprocessed_data = preprocess_live_data(live_traffic_df, scaler)
    
    # Step 5: Make predictions using the trained model
    predictions = model.predict(preprocessed_data)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Step 6: Print predicted results
    print("Predicted Classes (0 for Normal, 1 for Attack):", predicted_classes)
    
    # Step 7: Plot the predicted classes
    plt.plot(predicted_classes)
    plt.title('Predicted Classes for Live Captured Traffic')
    plt.ylabel('Class (0 = Normal, 1 = Attack)')
    plt.xlabel('Packet Index')
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
