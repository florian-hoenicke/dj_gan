from collections import defaultdict

from tools import TFRecordTools
from runstats import Statistics


class StatisticGen:
    def __init__(self, tfrecord_path):
        self.tfrecord_path = tfrecord_path

    def __call__(self, *args, **kwargs):
        raw_data_generator = TFRecordTools.read_tf_record_as_generator(self.tfrecord_path)
        mean_var = self.get_stats(raw_data_generator)
        return TFRecordTools.write_stats_as_json(mean_var, f'stats')

    def get_stats(self, raw_data_generator):
        print('start transform')
        stats_dict = defaultdict(Statistics)
        attribute_to_mean_var = defaultdict(dict)
        for track in raw_data_generator:
            for attribute, value_list in track.items():
                for value in value_list:
                    stats_dict[attribute].push(value)
        for attribute, statistics in stats_dict.items():
            mean_var_dict = attribute_to_mean_var[attribute]
            mean_var_dict['mean'] = statistics.mean()
            mean_var_dict['variance'] = statistics.variance()
        return attribute_to_mean_var

    def make_training_example_from_subtrack(self, track, start, end):
        sub_track = {
            'instrument_type': track['instrument_type'] * (end - start - 1),
            'is_drum': track['is_drum'] * (end - start - 1),
            'durations_in': track['durations'][start: end - 1],
            'velocity_in': track['velocity'][start: end - 1],
            'tones_in': track['tones'][start: end - 1],
            'octaves_in': track['octaves'][start: end - 1],
            'durations_out': track['durations'][start + 1: end],
            'velocity_out': track['velocity'][start + 1: end],
            'tones_out': track['tones'][start + 1: end],
            'octaves_out': track['octaves'][start + 1: end],
        }
        return sub_track
