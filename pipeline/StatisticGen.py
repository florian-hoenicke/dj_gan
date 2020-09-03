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
        print('start stats')
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
