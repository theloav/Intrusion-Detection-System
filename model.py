import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def build_model(input_shape, num_classes):
    """Build and compile the LSTM model."""
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
