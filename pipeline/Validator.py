from collections import defaultdict

from tools import TFRecordTools
from runstats import Statistics


class Validator:
    def __init__(self, tfrecord_path):
        self.tfrecord_path = tfrecord_path

    def __call__(self, *args, **kwargs):
        raw_data_generator = TFRecordTools.read_tf_record_as_generator(self.tfrecord_path)
        self.validate_data(raw_data_generator)

    def validate_data(self, raw_data_generator):
        print('start validation')
        for track in raw_data_generator:
            pass
