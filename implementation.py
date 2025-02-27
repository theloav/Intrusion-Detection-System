import pickle
from fastapi import FastAPI, HTTPException
import pandas as pd
from keras.models import load_model
from pydantic import BaseModel

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Check if the loaded object is of the correct type
print(type(scaler))  # Should be <class 'sklearn.preprocessing._data.StandardScaler'>


model = load_model("your_model_name.keras")

# Define the input schema with all necessary features
class InputData(BaseModel):
    Destination_Port: float
    Flow_Duration: float
    Total_Fwd_Packets: float
    Total_Backward_Packets: float
    Total_Length_of_Fwd_Packets: float
    Total_Length_of_Bwd_Packets: float
    Fwd_Packet_Length_Max: float
    Fwd_Packet_Length_Min: float
    Fwd_Packet_Length_Mean: float
    Fwd_Packet_Length_Std: float
    Bwd_Packet_Length_Max: float
    Bwd_Packet_Length_Min: float
    Bwd_Packet_Length_Mean: float
    Bwd_Packet_Length_Std: float
    Flow_Bytes_per_second: float
    Flow_Packets_per_second: float
    Flow_IAT_Mean: float
    Flow_IAT_Std: float
    Flow_IAT_Max: float
    Flow_IAT_Min: float
    Fwd_IAT_Total: float
    Fwd_IAT_Mean: float
    Fwd_IAT_Std: float
    Fwd_IAT_Max: float
    Fwd_IAT_Min: float
    Bwd_IAT_Total: float
    Bwd_IAT_Mean: float
    Bwd_IAT_Std: float
    Bwd_IAT_Max: float
    Bwd_IAT_Min: float
    Fwd_PSH_Flags: float
    Bwd_PSH_Flags: float
    Fwd_URG_Flags: float
    Bwd_URG_Flags: float
    Fwd_Header_Length: float
    Bwd_Header_Length: float
    Fwd_Packets_per_second: float
    Bwd_Packets_per_second: float
    Min_Packet_Length: float
    Max_Packet_Length: float
    Packet_Length_Mean: float
    Packet_Length_Std: float
    Packet_Length_Variance: float
    FIN_Flag_Count: float
    SYN_Flag_Count: float
    RST_Flag_Count: float
    PSH_Flag_Count: float
    ACK_Flag_Count: float
    URG_Flag_Count: float
    CWE_Flag_Count: float
    ECE_Flag_Count: float
    Down_Up_Ratio: float
    Average_Packet_Size: float
    Avg_Fwd_Segment_Size: float
    Avg_Bwd_Segment_Size: float
    Fwd_Header_Length_1: float
    Fwd_Avg_Bytes_per_Bulk: float
    Fwd_Avg_Packets_per_Bulk: float
    Fwd_Avg_Bulk_Rate: float
    Bwd_Avg_Bytes_per_Bulk: float
    Bwd_Avg_Packets_per_Bulk: float
    Bwd_Avg_Bulk_Rate: float
    Subflow_Fwd_Packets: float
    Subflow_Fwd_Bytes: float
    Subflow_Bwd_Packets: float
    Subflow_Bwd_Bytes: float
    Init_Win_bytes_forward: float
    Init_Win_bytes_backward: float
    act_data_pkt_fwd: float
    min_seg_size_forward: float
    Active_Mean: float
    Active_Std: float
    Active_Max: float
    Active_Min: float
    Idle_Mean: float
    Idle_Std: float
    Idle_Max: float
    Idle_Min: float

app = FastAPI()

@app.post("/predict")
async def predict(input_data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    # Scale input features
    scaled_features = scaler.transform(input_df)

    # Make predictions
    predictions = model.predict(scaled_features)

    # Convert predictions to binary classes
    predicted_class = (predictions > 0.5).astype(int).flatten().tolist()

    return {"predicted_class": predicted_class}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
