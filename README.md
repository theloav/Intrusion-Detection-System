# Intrusion Detection System using LSTM

📌 Overview
This Intrusion Detection System (IDS) leverages Long Short-Term Memory (LSTM) networks to capture temporal patterns in network traffic, detect anomalies, and handle long-term dependencies. The model achieves a training accuracy of 99.98%, ensuring high reliability in identifying potential security threats.

🚀 Features
✅ LSTM-based detection: Captures sequential dependencies for accurate anomaly detection.
✅ High accuracy: Achieves 99.98% training accuracy with optimized hyperparameters.
✅ Anomaly detection: Identifies potential cyber threats in network traffic.
✅ Scalability: Can be integrated into real-time monitoring systems.
✅ Preprocessing pipeline: Cleans and transforms raw network data before training.

🛠️ Tech Stack
Deep Learning Framework: TensorFlow / Keras
Language: Python
Libraries: NumPy, Pandas, Matplotlib, Scikit-learn
Model Architecture: LSTM Neural Network
🔧 Installation
1️⃣ Clone the repository:
git clone https://github.com/theloav/Intrusion-Detection-System.git
cd Intrusion-Detection-System
2️⃣ Install dependencies:
pip install -r requirements.txt
3️⃣ Run the IDS system:
python main.py

🧠 Model Training
Uses LSTM layers to capture time-based dependencies.
Dataset is normalized using MinMaxScaler.
Loss function: Binary Cross-Entropy
Optimizer: Adam

Training Performance
📌 Training Accuracy: 99.98%
📌 Loss: 0.002

📊 Results
Achieves high precision and recall for detecting anomalies.
Can detect both known and unknown threats in network traffic.
Efficient for real-time threat monitoring.

📝 Future Enhancements
🛠️ Real-time integration with security monitoring tools
📡 Support for multiple attack types
🎯 Fine-tuning with large-scale datasets
📜 License
This project is licensed under the MIT License.

🤝 Contributing
Pull requests are welcome! Open an issue for feature requests or bug reports.


