import os

import pretty_midi as pretty_midi
from glob import glob
import tensorflow as tf

class ExampleGen():
    def __init__(self, use_cache=True, validation_data_split=0.2, max_examples=None):
        self.rootdir = 'data/midis/'
        self.cache_dir = 'data/cache/examples/'
        self.use_cache = use_cache
        self.validation_data_split = validation_data_split
        self.max_examples = max_examples
        self.current_example = 1

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
            for file in files:
                try:
                    midi_pretty_format = pretty_midi.PrettyMIDI(file)
                except:
                    print('can not read file', file)
                    continue
                for i, instrument in enumerate(midi_pretty_format.instruments):
                    yield self.serialize_instrument(instrument)
        return instrument_generator

    def save_as_tfrecord(self, files, file_name):
        file_path = f"{self.cache_dir}{file_name}.tfrecord"
        file_exists = os.path.exists(file_path)
        if not self.use_cache or not file_exists:
            if file_exists:
                os.remove(file_path)
            with tf.io.TFRecordWriter(file_path) as tfwriter:
                for example in self.instrument_generator_factory(files)():
                    tfwriter.write(example.SerializeToString())
                    if self.current_example > self.max_examples:
                        break
                    self.current_example += 1
        return file_path

    def create_train_and_test(self):
        all_files = [f for f in glob(f"{self.rootdir}**/*.mid", recursive=True)]
        split_index = int(len(all_files) * self.validation_data_split)
        train = all_files[split_index:]
        test = all_files[:split_index]
        train_file = self.save_as_tfrecord(train, 'train')
        test_file = self.save_as_tfrecord(test, 'test')
        return train_file, test_file


if __name__ == '__main__':
    ExampleGen().create_train_and_test()

