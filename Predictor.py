from model import RNNModel
from tools import TFRecordTools
import tensorflow as tf
import numpy as np

from tools.MIDITools import MIDITools


class Predictor:
    def __init__(self):
        self.stats_dict = TFRecordTools.read_stats_from_json('data/cache/examples/stats.json')
        self.model = self.load_trained_model()
        self.sequence_length = 1

    def predict(self, steps_to_predict, prediction_seeds):
        self.model.reset_states()
        for i in range(len(prediction_seeds) - 1):
            prediction_seed = prediction_seeds[i]
            self.model.predict(prediction_seed)
        for i in range(steps_to_predict):
            prediction_seed = prediction_seeds[-1]
            prediction = self.model.predict(prediction_seed)
            prediction_seeds.append(self.sample_from_prediction(prediction, prediction_seed))
        final_prediction = [self.denormalize(x) for x in prediction_seeds]
        return final_prediction

    def load_trained_model(self):
        model = RNNModel.create(sequence_length=1, batch_size=1)
        model.load_weights('data/model/weights/rnn')
        tf.keras.utils.plot_model(model, 'rnn_seq_1')
        return model
        # tf.keras.models.load_model('data/model/rnn/')

    def sample_from_prediction(self, prediction, previous):
        print(prediction)
        return {
            'instrument_type': previous['instrument_type'],
            'is_drum': previous['is_drum'],
            'durations_in': prediction['durations_out'][0],
            'velocity_in': prediction['velocity_out'][0],
            'tones_in': self.sample(prediction['tones_out']),
            'octaves_in': self.sample(prediction['octaves_out']),
        }

    def sample(self, probability_distribution):
        x = np.random.choice(probability_distribution.shape[-1], p=probability_distribution[0, 0])
        x = np.array([[x]])
        return x

    def denormalize(self, x):
        x = {key: val[0][0] for key, val in x.items()}
        x['durations_in'] = x['durations_in'] * self.stats_dict['durations']['variance'] + self.stats_dict['durations']['mean']
        x['velocity_in'] = int(x['velocity_in'] * self.stats_dict['velocity']['variance'] + self.stats_dict['velocity']['mean'])
        x['instrument_type'] = int(x['instrument_type'])
        x['is_drum'] = x['is_drum'] == 1
        x['tones_in'] = int(x['tones_in'])
        x['octaves_in'] = int(x['octaves_in'])

        return x


if __name__ == '__main__':
    sequence = Predictor().predict(
        steps_to_predict=200,
        prediction_seeds=[{
            'instrument_type': np.ones((1, 1)),
            'is_drum': np.ones((1, 1)),
            'durations_in': np.ones((1, 1)),
            'velocity_in': np.ones((1, 1)),
            'tones_in': np.ones((1, 1)),
            'octaves_in': np.ones((1, 1)),
        }]
    )
    midi = MIDITools.save_sequence_as_midi(sequence, 'data/generated/example.midi')

