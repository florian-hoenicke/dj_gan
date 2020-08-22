import os

import pretty_midi as pretty_midi
from glob import glob
import tensorflow as tf

class ExampleGen():
    def __init__(self, use_cache=True):
        self.rootdir = 'data/midi/'
        self.cache_dir = 'data/cache/examples'
        self.use_cache = use_cache

    def __call__(self, *args, **kwargs):
        return self.create_train_and_test()

    def serialize_instrument(self, instrument):
        return tf.train.Example(
            features=tf.train.Features(
                feature= {
                    # categorical features
                    'instrument_type': tf.train.Feature(int64_list=tf.train.Int64List(value=[instrument.program])),
                    'is_drum': tf.train.Feature(int64_list=tf.train.Int64List(value=[1 if instrument.is_drum else 0])),
                    'octaves': tf.train.Feature(int64_list=tf.train.Int64List(value=[note.pitch // 12 for note in instrument.notes])),
                    'tones': tf.train.Feature(int64_list=tf.train.Int64List(value=[note.pitch % 12 for note in instrument.notes])),
                    'durations': tf.train.Feature(float_list=tf.train.FloatList(value=[note.duration for note in instrument.notes])),
                    'velocity': tf.train.Feature(float_list=tf.train.FloatList(value=[note.velocity for note in instrument.notes])),
                }
            )
        )

    def instrument_generator_factory(self, files):
        def instrument_generator():
            print('method called')
            for file in files:
                try:
                    midi_pretty_format = pretty_midi.PrettyMIDI(file)
                except:
                    print('not midi', file)
                    continue
                for i, instrument in enumerate(midi_pretty_format.instruments):
                    yield self.serialize_instrument(instrument)
        return instrument_generator

    def save_as_tfrecord(self, files, file_name):
        file_path = f"{self.cache_dir}{file_name}.tfrecord"
        if not self.use_cache or not os.path.exists(file_path):
            with tf.io.TFRecordWriter(file_path) as tfwriter:
                for example in self.instrument_generator_factory(files)():
                    tfwriter.write(example.SerializeToString())
        return file_path

    def create_train_and_test(self, validation_data_split=0.2):
        self.download_files()
        all_files = [f for f in glob(f"{self.rootdir}**/*.mid", recursive=True)]
        split_index = int(len(all_files) * validation_data_split)
        train = all_files[split_index:]
        test = all_files[:split_index]
        train_file = self.save_as_tfrecord(train, 'train')
        test_file = self.save_as_tfrecord(test, 'test')
        return train_file, test_file

    def download_files(self):
        tf.keras.utils.get_file('midi.tar.zip', 'https://drive.google.com/u/0/uc?export=download&confirm=Ce6C&id=0B4wY8oEgAUnjX3NzSUJCNVZHbmc')



if __name__ == '__main__':
    ExampleGen().download_files()
    # ExampleGen().create_train_and_test()

