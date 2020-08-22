import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.python.keras.layers import Dense, LSTM

# Generator


coding_size = 30

class generator:
    def __init__(self):
        dense1 = Dense(100)
        dense2 = Dense(1000)
        lstm = LSTM(100, return_sequences=True)
        out_note = Dense(num_notes)
        out_speed = Dense(1)





note_input = InputLayer(input_shape=(None,))
velocity_input = InputLayer(input_shape=(None,))
