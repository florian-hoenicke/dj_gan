from pipeline.ExampleGen import ExampleGen
import tensorflow as tf

train_path, test_path = ExampleGen(use_cache=True, max_examples=100)()
train_dataset = tf.data.TFRecordDataset(train_path)
for x in train_dataset:
    print(tf.train.Example.FromString(x.numpy()))
# train_dataset = train_path.map(lambda x: tf.train.Example.FromString(x.numpy()))


# print(train_path, test_path)