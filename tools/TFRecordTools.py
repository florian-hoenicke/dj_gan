import tensorflow as tf

def load_dataset(tf_record_path):
    tftf.data.TFRecordDataset(tf_record_path)

def save_dataset(dataset, tf_record_path)