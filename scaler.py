import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import pickle

# List of CSV files
csv_files = [
    'data/Monday-WorkingHours.pcap_ISCX.csv',
    'data/Tuesday-WorkingHours.pcap_ISCX.csv',
    'data/Wednesday-workingHours.pcap_ISCX.csv',
    'data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'data/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'data/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
]

# Step 1: Combine the CSV files into a single DataFrame
combined_data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Strip whitespace from column names and replace spaces with underscores
combined_data.columns = combined_data.columns.str.strip().str.replace(' ', '_')

# Print the columns and the first few rows for debugging
print("Columns in the combined DataFrame:", combined_data.columns)
print(combined_data.head())

# Step 2: Select the features you want to scale
features_to_scale = ['Destination_Port', 'Flow_Duration', 'Total_Fwd_Packets',
                     'Total_Backward_Packets', 'Total_Length_of_Fwd_Packets',
                     'Total_Length_of_Bwd_Packets', 'Fwd_Packet_Length_Max',
                     'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean',
                     'Fwd_Packet_Length_Std', 'Bwd_Packet_Length_Max',
                     'Bwd_Packet_Length_Min', 'Bwd_Packet_Length_Mean',
                     'Bwd_Packet_Length_Std', 'Flow_Bytes/s', 'Flow_Packets/s',
                     'Flow_IAT_Mean', 'Flow_IAT_Std', 'Flow_IAT_Max', 'Flow_IAT_Min',
                     'Fwd_IAT_Total', 'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Max',
                     'Fwd_IAT_Min', 'Bwd_IAT_Total', 'Bwd_IAT_Mean', 'Bwd_IAT_Std',
                     'Bwd_IAT_Max', 'Bwd_IAT_Min', 'Fwd_PSH_Flags', 'Bwd_PSH_Flags',
                     'Fwd_URG_Flags', 'Bwd_URG_Flags', 'Fwd_Header_Length',
                     'Bwd_Header_Length', 'Fwd_Packets/s', 'Bwd_Packets/s',
                     'Min_Packet_Length', 'Max_Packet_Length', 'Packet_Length_Mean',
                     'Packet_Length_Std', 'Packet_Length_Variance', 'FIN_Flag_Count',
                     'SYN_Flag_Count', 'RST_Flag_Count', 'PSH_Flag_Count', 'ACK_Flag_Count',
                     'URG_Flag_Count', 'CWE_Flag_Count', 'ECE_Flag_Count', 'Down/Up_Ratio',
                     'Average_Packet_Size', 'Avg_Fwd_Segment_Size', 'Avg_Bwd_Segment_Size',
                     'Fwd_Header_Length.1', 'Fwd_Avg_Bytes/Bulk', 'Fwd_Avg_Packets/Bulk',
                     'Fwd_Avg_Bulk_Rate', 'Bwd_Avg_Bytes/Bulk', 'Bwd_Avg_Packets/Bulk',
                     'Bwd_Avg_Bulk_Rate', 'Subflow_Fwd_Packets', 'Subflow_Fwd_Bytes',
                     'Subflow_Bwd_Packets', 'Subflow_Bwd_Bytes', 'Init_Win_bytes_forward',
                     'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
                     'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean',
                     'Idle_Std', 'Idle_Max', 'Idle_Min']

if all(feature in combined_data.columns for feature in features_to_scale):
    # Replace infinite values with NaN
    combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop any rows with missing values
    features = combined_data[features_to_scale].dropna()

    # Fit the scaler
    scaler = StandardScaler()
    scaler.fit(features)

    # Step 3: Save the scaler for later use
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved successfully.")
else:
    missing_features = [feature for feature in features_to_scale if feature not in combined_data.columns]
    print(f"Columns {missing_features} not found in the combined DataFrame. Please check the column names.")
