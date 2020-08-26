import tensorflow as tf
import numpy as np
import json

cache_dir = 'data/cache/examples/'


def type_to_tf_type(some_type):
    if some_type in [int, np.int64]:
        return tf.int64
    elif some_type in [float, np.float64]:
        return tf.float64
    else:
        raise Exception(f'At the moment, the type {some_type} is not supported.')


def get_output_types_from_dict(feature_dict):
    return {key: type_to_tf_type(type(val[0])) for key, val in feature_dict.items()}


def make_dataset_generator(tf_record_dataset):
    first_example_bytes = list(tf_record_dataset.take(1))[0].numpy()
    first_example = tf.train.Example.FromString(first_example_bytes)
    first_dict = example_to_dict(first_example)
    output_types = get_output_types_from_dict(first_dict)

    def dataset_generator():
        for x in tf_record_dataset:
            example = tf.train.Example.FromString(x.numpy())
            yield example_to_dict(example)

    return dataset_generator(), output_types


def generator_wrapper_factory(tf_record_path):
    def generator_wrapper():
        tf_record_dataset = tf.data.TFRecordDataset(tf_record_path)
        g, _ = make_dataset_generator(tf_record_dataset)
        return g

    return generator_wrapper


def read_tf_record_as_dataset(tf_record_path):
    tf_record_dataset = tf.data.TFRecordDataset(tf_record_path)
    _, output_types = make_dataset_generator(tf_record_dataset)
    dataset = tf.data.Dataset.from_generator(generator_wrapper_factory(tf_record_path), output_types)
    return dataset


def read_tf_record_as_generator(tf_record_path):
    tf_record_dataset = tf.data.TFRecordDataset(tf_record_path)
    dataset_generator, _ = make_dataset_generator(tf_record_dataset)
    return dataset_generator


def dict_to_example(feature_dict):
    tf_feature_dict = {}
    for key, transmitted in feature_dict.items():
        transmitted_type = type(transmitted)
        if not transmitted_type == list:
            transmitted = [transmitted]
        value_type = type(transmitted[0])
        if value_type in [int, np.int64]:
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=transmitted))
        elif value_type in [float, np.float64]:
            feature = tf.train.Feature(float_list=tf.train.FloatList(value=transmitted))
        else:
            raise Exception(f'At the moment is {value_type} not supported')
        tf_feature_dict[key] = feature
    return tf.train.Example(
        features=tf.train.Features(
            feature=tf_feature_dict
        )
    )


def example_to_dict(example):
    result_dict = {}
    feature_dict = example.features.feature
    for key, wrapped_val in feature_dict.items():
        # wrapped_val = example.features.feature[key]
        bytes_list = wrapped_val.bytes_list.value
        int_list = wrapped_val.int64_list.value
        float_list = wrapped_val.float_list.value
        if len(bytes_list) != 0:
            val = bytes_list
        elif len(int_list) != 0:
            val = int_list
        else:
            val = float_list
        result_dict[key] = list(val)
    return result_dict


def write_tfrecords_from_generator(generator, file_name):
    file_path = f"{cache_dir}{file_name}.tfrecord"
    with tf.io.TFRecordWriter(file_path) as tfwriter:
        for input_dict in generator:
            example = dict_to_example(input_dict)
            tfwriter.write(example.SerializeToString())
    return file_path


def write_stats_as_json(mean_var, file_name):
    file_path = f'{cache_dir}{file_name}.json'
    with open(file_path, mode='w') as f:
        json.dump(mean_var, f)
    return file_path


def read_stats_from_json(file_name):
    with open(file_name) as f:
        return json.load(f)


if __name__ == '__main__':
    example = dict_to_example({
        'a': 4,
        'b': 0.1,
        'c': [4, 3, 2]
    })
    dict = example_to_dict(example)
    print('break')
