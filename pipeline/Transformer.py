from tools import TFRecordTools

class Transformer:
    def __init__(self, tfrecord_path_list, stats_path, sliding_window=100):
        self.stats = TFRecordTools.read_stats_from_json(stats_path)
        self.tfrecord_path_list = tfrecord_path_list
        self.sliding_window = sliding_window
    def __call__(self, *args, **kwargs):
        for i,tfrecord_path in enumerate(self.tfrecord_path_list):
            raw_data_generator = TFRecordTools.read_tf_record_as_generator(tfrecord_path)
            transformed_data_generator = self.transform(raw_data_generator)
            yield TFRecordTools.write_tfrecords_from_generator(transformed_data_generator, f'transformed{i}')

    def transform(self, raw_data_generator):
        print('start transform')
        for track in raw_data_generator:
            self.normalize_track(track)
            track_len = len(track['octaves'])
            for i in range(0, track_len - self.sliding_window, self.sliding_window + 1):
                training_example_dict = self.make_training_example_from_subtrack(track, start=i, end=i+self.sliding_window + 1)
                yield training_example_dict

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

    def normalize_track(self, track):
        for attribute in ['velocity', 'durations']:
            value_list = track[attribute]
            for i in range(len(value_list)):
                value_list[i] = (value_list[i] - self.stats[attribute]['mean']) / self.stats[attribute]['variance']
