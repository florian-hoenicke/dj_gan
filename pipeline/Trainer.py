from model import RNNModel
from tools import TFRecordTools


def split_in_and_out(x):
    in_dict = {}
    out_dict = {}
    print('x', x)
    for key, val in x.items():
        if "_out" in key:
            out_dict[key] = val
        else:
            in_dict[key] = val
    return (in_dict, out_dict)


class Trainer:
    def __init__(self, train_transformed_path, test_transformed_path, batch_size=128, steps_per_epoch=20,
                 steps_per_validation=5, epochs=1):
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_validation = steps_per_validation
        self.epochs = epochs
        self.training_dataset = TFRecordTools.read_tf_record_as_dataset(train_transformed_path) \
            .repeat() \
            .shuffle(buffer_size=1000) \
            .map(split_in_and_out) \
            .batch(batch_size)
        self.test_dataset = TFRecordTools.read_tf_record_as_dataset(test_transformed_path) \
            .repeat() \
            .map(split_in_and_out) \
            .batch(batch_size)
        self.model = RNNModel.create()

    def __call__(self):
        self.model.fit(
            self.training_dataset,
            validation_data=self.test_dataset,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.steps_per_validation,
            epochs=self.epochs
        )
        self.model.save_weights('data/model/weights/rnn')
