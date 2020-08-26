from model import RNNModel
from tools import TFRecordTools
import tensorflow as tf
import numpy as np

class Predictor:
    def __init__(self):
        self.stats_dict = TFRecordTools.read_stats_from_json('data/cache/examples/stats.json')
        self.model = self.load_trained_model()
        self.sequence_length = 100

    def predict(self, steps_to_predict):
        rand_input = {
            'instrument_type': np.ones((128, self.sequence_length)),
            'is_drum': np.ones((128, self.sequence_length)),
            'durations_in': np.ones((128, self.sequence_length)),
            'velocity_in': np.ones((128, self.sequence_length)),
            'tones_in': np.ones((128, self.sequence_length)),
            'octaves_in': np.ones((128, self.sequence_length)),
        }

        prediction = self.model.predict(rand_input)

        print(prediction)

    def load_trained_model(self):
        model = RNNModel.create(sequence_length=1)
        model.load_weights('data/model/rnn/')
        # tf.keras.models.load_model('data/model/rnn/')