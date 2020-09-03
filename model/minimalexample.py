import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write
from tensorflow import expand_dims
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


def save_audio_files(audio_files, epoch):
    path = f'data/generated/{epoch}/'
    try:
        os.makedirs(path)
    except:
        pass
    for i, audio_file in enumerate(audio_files):
        np_audio = audio_file.numpy().astype(np.int16)
        # if np_audio.min() < -32768 or np_audio.max() > 32767:
        #     print('ignore audio - min:', audio_file.min(), 'max:', audio_file.max())
        write(f'{path}{i}.wav', 8000, np_audio)
        plt.figure()
        plt.plot(np_audio)
        plt.savefig(f'{path}{i}.png')
        plt.close()
        break


class WavGanModel:
    def __init__(self, epochs=1, z_size=128, batch_size=8, depth=16, num_filters=32, steps=None):
        self.batch_size = batch_size
        self.z_size = z_size
        self.epochs = epochs
        self.depth = depth
        self.num_filters = num_filters
        self.steps = steps
        self.discriminator = self.create_gan()

    def create_discriminator(self, depth=16):
        inputs = {
            f'output{i}': Input(shape=(int(math.pow(2, i)))) for i in range(
                # 3,
                0,
                depth
            )
        }

        expanded = [expand_dims(input, 2) for name, input in inputs.items()]
        x = [
                Conv1D(
                    self.num_filters,
                    9,
                    # strides=2
                )(
                    x
                )
                for x in expanded[-3:-1]
            ]
        x = [Flatten()(x_i) for x_i in x]
        x = [LeakyReLU()(x_i) for x_i in x]
        # x = [BatchNormalization()(x_i) for x_i in x]

        x = tf.keras.layers.Concatenate()(x)
        x = Flatten()(x)
        x = Dense(30)(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=output)
        plot_model(
            model,
            to_file='model/discriminator.png',
            show_shapes=True,
            show_layer_names=True,
        )
        return model

    def create_gan(self):
        discriminator = self.create_discriminator(self.depth)
        discriminator.compile(
            loss=BinaryCrossentropy(),
            optimizer='rmsprop',
            metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5)]
        )
        return discriminator

    def train(self):
        # self.discriminator.trainable = True
        with open('deleteme-real.pickle', mode='rb') as f:
            real_batches = pickle.load(f)
        with open('deleteme.pickle', mode='rb') as f:
            fake_batches = pickle.load(f)

        while True:
            fake_real_batch = {
                name: tf.concat(
                    [fake_batch, real_batches[name]], axis=0
                ) for name, fake_batch in fake_batches.items()
            }
            labels = tf.constant([[0]] * self.batch_size + [[1]] * self.batch_size)
            acc_fake = self.get_accuracy_from_descriminator_result(self.discriminator(fake_batches), 0)
            # acc_real = self.get_accuracy_from_descriminator_result(self.discriminator(real_batches), 1)
            print('discriminator acc fake:', acc_fake)  # , 'real:', acc_real)
            if acc_fake == 1:
                print('stop')
            history_discriminator_real = self.discriminator.train_on_batch(fake_real_batch, y=labels, return_dict=True)
            print('discriminator history:', history_discriminator_real)

    def get_accuracy_from_descriminator_result(self, output, label):
        label = True if label == 1 else False
        counts = dict(zip(*np.unique((output > 0.5).numpy(), return_counts=True)))
        acc = (counts[label] if label in counts else 0) / batch_size
        return acc


if __name__ == '__main__':
    depth = 16
    batch_size = 32
    gan = WavGanModel(
        depth=depth,
        epochs=1000,
        batch_size=batch_size,
        num_filters=32,
        # steps=1
    )
    gan.train()
