import math
import os
import pickle
import random
from collections import defaultdict
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from tensorflow import expand_dims
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Conv1DTranspose, \
    BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Multiply
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
        if i == 15:
            break


class WavGanModel:
    def __init__(self, epochs=1, z_size=128, batch_size=8, depth=16, num_filters=32, steps=None):
        self.batch_size = batch_size
        self.z_size = z_size
        self.epochs = epochs
        self.depth = depth
        self.num_filters = num_filters
        self.steps = steps
        self.gan, self.generator, self.discriminator = self.create_gan()
        self.load_weights()

    def create_generator(self, depth=16):
        z = Input(shape=(self.z_size,))
        d = Dense(self.num_filters)(z)
        d = LeakyReLU()(d)
        d = Dense(self.num_filters)(d)
        d = LeakyReLU()(d)
        d = Dense(self.num_filters)(d)
        d = LeakyReLU()(d)


        # start with 1Hz, 10 seconds
        filters = [self.num_filters] * (depth - 1)
        w = [Dense(self.num_filters)(d) for i in range(depth)]
        w = [LeakyReLU()(w_i) for w_i in w]

        kernels = [2] + [4] + [8] + [8] * (depth - 4)
        x = Dense(1, name=f'output{0}')(w[0])
        outputs = {'output0': x}

        for i, [filter, kernel, w_i] in enumerate(zip(filters, kernels, w[1:])):
            x = expand_dims(x, 2)
            x = Conv1DTranspose(
                filters=filter,
                kernel_size=kernel,
                strides=1,
                padding='same',
                # use_bias=False,
                name=f'deconv_a{i}'
            )(x)
            w_i = expand_dims(w_i, 1)
            x = Multiply()([x, w_i])
            # x = BatchNormalization()(x)
            x = LeakyReLU()(x)

            # x = Conv1DTranspose(
            #     filters=filter,
            #     kernel_size=kernel,
            #     strides=1,
            #     padding='same',
            #     # use_bias=False,
            #     name=f'deconv_b{i}'
            # )(x)
            # # # x = BatchNormalization()(x)
            # x = LeakyReLU()(x)
            #
            # x = Conv1DTranspose(
            #     filters=filter,
            #     kernel_size=kernel,
            #     strides=1,
            #     padding='same',
            #     # use_bias=False,
            #     name=f'deconv_c{i}'
            # )(x)
            # x = BatchNormalization()(x)
            # x = LeakyReLU()(x)

            x = Conv1DTranspose(
                filters=1,
                kernel_size=kernel,
                strides=2,
                padding='same',
                # use_bias=False,
                activation=tf.keras.activations.tanh,
                name=f'deconv_d{i}'
            )(x)
            out = Flatten(name=f'output{i + 1}')(x)
            outputs[f'output{i + 1}'] = out
            # x = BatchNormalization()(out)
            x = out
        model = Model(
            inputs=z,
            outputs=outputs
        )

        plot_model(
            model,
            show_shapes=True,
            show_layer_names=True,
            to_file='model/generator.png'
        )
        # model.summary()
        return model

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
                Conv1D(self.num_filters, 1)(expanded[0]),  # 1
                Conv1D(self.num_filters, 2)(expanded[1]),  # 2
                Conv1D(self.num_filters, 3)(expanded[2]),  # 4
                Conv1D(self.num_filters, 4, strides=2)(expanded[3]),  # 8
                Conv1D(
                    self.num_filters,
                    5,
                    strides=2
                )(
                    # Conv1D(
                    #     self.num_filters,
                    #     4
                    # )(expanded[4])
                    expanded[4]
                )
            ] + [
                Conv1D(
                    self.num_filters,
                    9,
                    strides=2
                )(
                    x
                    # Conv1D(
                    #     self.num_filters,
                    #     4,
                    #     strides=2
                    # )(
                    #     Conv1D(
                    #         self.num_filters,
                    #         4
                    #     )(x)
                    # )
                )
                for x in expanded[5:]
            ]
        x = [Flatten()(x_i) for x_i in x]
        x = [LeakyReLU()(x_i) for x_i in x]
        # x = [BatchNormalization()(x_i) for x_i in x]



        x = [Dense(30)(x_i) for x_i in x]
        x = [LeakyReLU()(x_i) for x_i in x]
        x = [Dense(1, activation='sigmoid')(x_i) for x_i in x]
        output = tf.keras.layers.Average()(x)

        # alternative:
        # x = tf.keras.layers.Concatenate()(x)
        # x = Flatten()(x)
        # x = Dense(30)(x)
        # output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=output)
        plot_model(
            model,
            to_file='model/discriminator.png',
            show_shapes=True,
            show_layer_names=True,
        )
        return model

    def create_gan(self):
        # noise_in = Input(shape=(self.z_size))
        generator = self.create_generator(self.depth)
        discriminator = self.create_discriminator(self.depth)
        # discriminator_out = discriminator(generator(noise_in))
        # gan = tf.keras.models.Sequential([
        #     generator,
        #     discriminator
        # ])
        gan = Model(
            inputs=generator.input,
            outputs=discriminator(generator.output)
        )
        # gan = Model(
        #     inputs=noise_in,
        #     outputs=discriminator_out
        # )
        plot_model(
            gan,
            'model/gan.png',
            show_layer_names=True,
            show_shapes=True
        )
        discriminator.compile(
            loss=BinaryCrossentropy(),
            optimizer='rmsprop',
            metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5)]
        )
        discriminator.trainable = False

        gan.compile(
            loss=BinaryCrossentropy(),
            optimizer='rmsprop',
            metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5)]
        )

        return gan, generator, discriminator

    def train(self, dataset, mean, scale):
        for epoch in range(self.epochs):
            print('############ start epoch:', epoch, '###########')
            fake_batches = None
            for i, real_batches in enumerate(dataset):
                print('epoch: ', epoch, ', step: ', i)
                if self.steps is not None and i > self.steps:
                    break

                # phase 1 - training discriminator

                # fake_real_batch = {
                #     name: tf.concat(
                #         [fake_batch, real_batches[name]], axis=0
                #     ) for name, fake_batch in fake_batches.items()
                # }
                # print('discriminator real before', real_batches['output15'].numpy().tolist()[0][0][:10])
                # print('discriminator real', fake_real_batch['output15'].numpy().tolist()[8:][0][0])
                # print('discriminator fake', fake_real_batch['output15'].numpy().tolist()[:8][0][0])
                # fake_real_label = tf.constant([0] * self.batch_size + [1] * self.batch_size)
                self.discriminator.trainable = True
                # self.generator.trainable = False
                # print('train discriminator')
                # print('discriminator before training', str(self.discriminator.layers[32].variables)[0:100])
                # print('generator before training', str(self.generator.layers[1].variables)[0:100])
                real_batches = {k: tf.constant(v) for k, v in real_batches.items()}
                history_discriminator_fake = None
                history_discriminator_real = None
                history_generator = None


                noise = tf.random.normal(shape=[self.batch_size, self.z_size])
                fake_batches = self.generator(noise)
                fake_real_batch = {
                    name: tf.concat(
                        [fake_batch, real_batches[name]], axis=0
                    ) for name, fake_batch in fake_batches.items()
                }
                labels = tf.constant([[0]] * self.batch_size + [[1]] * self.batch_size)
                acc_fake = self.get_accuracy_from_descriminator_result(self.discriminator(fake_batches), 0)
                acc_real = self.get_accuracy_from_descriminator_result(self.discriminator(real_batches), 1)
                # print('discriminator acc fake:', acc_fake, 'real:', acc_real)

                if acc_real == 1 and acc_fake == 1:
                    print('skip discriminator training since it is doing too well', acc_real, acc_fake)
                else:
                    history_discriminator_real = self.discriminator.train_on_batch(fake_real_batch, y=labels, return_dict=True)

                print('discriminator history:', history_discriminator_real, 'acc_fake', acc_fake, 'acc_real', acc_real)
                if acc_real < 0.3 or acc_fake < 0.3:
                    continue

                # while True:
                #     acc = self.get_accuracy_from_descriminator_result(self.discriminator(real_batches), 1)
                #     print('discriminator real loss, acc:', acc)
                #     if acc == 1:
                #         break
                #     history_discriminator_real = self.discriminator.train_on_batch(real_batches,
                #                                                             tf.constant([1] * self.batch_size), )
                #
                #
                #     if acc > 0:
                #         break
                #
                # while True:
                #     noise = tf.random.normal(shape=[self.batch_size, self.z_size])
                #     fake_batches = self.generator(noise)
                #
                #     acc = self.get_accuracy_from_descriminator_result(self.discriminator(fake_batches), 0)
                #     print('discriminator fake loss, acc:', acc)
                #     if acc == 1:
                #         break
                #     history_discriminator_fake = self.discriminator.train_on_batch(fake_batches,
                #                                                               tf.constant([0] * self.batch_size))
                #     if acc > 0 :
                #         break





                # print('discriminator after training', str(self.discriminator.layers[32].variables)[0:100])
                # print('generator after training', str(self.generator.layers[1].variables)[0:100])
                # phase 2 - training generator


                self.discriminator.trainable = False
                # self.generator.trainable = True
                # print('train generator')
                # print('discriminator before training', str(self.discriminator.layers[32].variables)[0:100])
                # print('generator before training', str(self.generator.layers[1].variables)[0:100])
                while True:
                    noise = tf.random.normal(shape=[self.batch_size, self.z_size])
                    label = tf.constant([[1]] * self.batch_size)

                    acc = self.get_accuracy_from_descriminator_result(self.gan(noise), 1)
                    # print('generator loss, acc:', acc)
                    if acc == 1:
                        break
                    history_generator = self.gan.train_on_batch(noise, label, return_dict=True)
                    print('gan history:', history_generator)
                    if acc > 0.3:
                        break

                # print('discriminator after training', str(self.discriminator.layers[32].variables)[0:100])
                # print('generator after training', str(self.generator.layers[1].variables)[0:100])

                # if i == 0:
                #     print('discriminator_loss_acc_fake:', history_discriminator_fake)
                #     print('discriminator_loss_acc_real:', history_discriminator_real)
                #     print('generator_loss:', history_generator)
                #     print('=> total_loss:', [history_discriminator_fake[0] + history_discriminator_real[0] + history_generator[0],
                #                              history_discriminator_fake[1] + history_discriminator_real[1] + history_generator[1]])

            audio_files = fake_batches[f'output{self.depth - 1}']
            audio_files = audio_files * scale + mean
            save_audio_files(
                audio_files,
                # int(math.pow(2, self.depth)),
                epoch
            )
            self.save_weights()

    def get_accuracy_from_descriminator_result(self, output, label):
        label = True if label == 1 else False
        counts = dict(zip(*np.unique((output > 0.5).numpy(), return_counts=True)))
        acc = (counts[label] if label in counts else 0) / batch_size
        return acc

    def save_weights(self):
        self.gan.save_weights('data/weights/latest')

    def load_weights(self):
        try:
            self.gan.load_weights('data/weights/latest')
            print('weights loaded')
        except:
            print('no weights to load')

def create_sample(scaled, step, i):
    res = scaled[i * step:i * step + step]
    return res


def scale_to_halfe(y):
    if y.shape[0] % 2 == 1:
        y = y[:-1]
    return y[0::2] # + y[1::2]) / 2
    # scaled = []
    # for i in range(0, total_sequence_length, step):
    #     if step == 1:
    #         scaled.append(y[i])
    #     else:
    #         scaled.append(np.mean(y[i:i + step]))
    # return np.array(scaled)


def create_training_data():
    # sound = AudioSegment.from_mp3('data/experiment/dubstep-part.mp3')
    # sound.export('data/experiment/dubstep-part.wav', format="wav")
    current_sample_rate, y = read('data/experiment/dubstep-part-8000.wav')
    assert current_sample_rate == 8000
    # target_sample_rate = 8000
    # sample_step_size = int(current_sample_rate / target_sample_rate)
    # y = y[::sample_step_size, 0]
    # write('data/experiment/dubstep-part-8000.wav', target_sample_rate,y)

    y = y.astype(np.float32)




    # filename = f'data/deleteme/raw.wav'
    # yy = np.array(y, dtype=np.int16)
    # write(filename, 8000, yy)



    # plt.figure()
    # plt.plot(y)
    # plt.savefig(f'source_dubstep_visualization.png')
    # plt.close()

    mean = np.mean(y)
    # scale = np.std(y)
    scale = (y.max() - y.min()) / 2
    y = (y - mean) / scale

    # plt.figure()
    # plt.plot(y[0:32000])
    # plt.savefig(f'source_dubstep_visualization_norm.png')
    # plt.close()

    # write('data/experiment/dubstep8000.wav',target_sample_rate, y)
    total_sequence_length = y.shape[0]
    sample_len = int(math.pow(2, depth - 1))
    number_of_samples = int(total_sequence_length / sample_len)
    samples = []
    down_scaled = []
    scaled = y
    down_scaled.append(scaled)

    # yy = scaled * scale + mean
    # filename = f'data/deleteme/last_downscaled.wav'
    # yy = np.array(yy, dtype=np.int16)
    # write(filename, 8000, yy)

    for i in range(0, depth - 1):
        scaled = scale_to_halfe(scaled)
        down_scaled.insert(0, scaled)
    for i in range(0, number_of_samples - 1):
        if i % 50 == 0 and i != 0:
            # break
            print('generating samples', i)
        # small_sequence = y[i * sample_len: (i+1)*sample_len]
        sample = {
            f'output{j}': create_sample(scaled, int(math.pow(2, j)), i)
            for j, scaled in enumerate(down_scaled)
        }
        samples.append(sample)

        # if i==1590:
        #     yy = samples[i]['output15'] * scale + mean
        #     filename = f'data/deleteme/created_sample_100.wav'
        #     yy = np.array(yy, dtype=np.int16)
        #     write(filename, 8000, yy)

    with open('data/experiment/wav_train_examples.pickle', mode='wb') as f:

        # yy = samples[0]['output15'] * scale + mean
        # filename = f'data/deleteme/before_picke.wav'
        # yy = np.array(yy, dtype=np.int16)
        # write(filename, 8000, yy)

        pickle.dump([samples, mean, scale], f)
    print('')
    return [samples, mean, scale]


def open_training_data():
    with open('data/experiment/wav_train_examples.pickle', mode='rb') as f:
        res = pickle.load(f)
        return res


def check_training_data():
    with open('data/experiment/wav_train_examples.pickle', mode='rb') as f:
        samples, mean, scale = pickle.load(f)
        index = randint(0, len(samples) - 1)
        for i in range(index, index + 2):
            sample = samples[i]
            for name, rate in zip(['output15','output14', 'output13', 'output12', 'output11', 'output10', 'output9', 'output8','output7','output6','output5','output4',], [int(8000 / math.pow(2,i)) for i in range(0,12)]):

                # yy = samples[0]['output15'] * scale + mean
                # filename = f'data/deleteme/after_picke.wav'
                # yy = np.array(yy, dtype=np.int16)
                # write(filename, 8000, yy)

                output = sample[name]
                output = output*scale+mean
                path = f'data/deleteme/{i}/'
                if not os.path.exists(path):
                    os.makedirs(path)

                filename = f'{path}{name}.wav'
                output = output.astype(np.int16)
                write(filename, rate, output)


try dropouts


def merge_dicts(list_of_dicts):
    merged = defaultdict(list)
    for d in list_of_dicts:
        for k, v in d.items():
            merged[k].append(v)
    return merged


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

    [training_data, mean, scale] = create_training_data()
    # [training_data, mean, scale] = open_training_data()
    # check_training_data()

    # exit()
    random.shuffle(training_data)
    batched_data = []
    for i in range(0, len(training_data) - batch_size, batch_size):
        batched_data.append(merge_dicts(training_data[i:i + batch_size]))
    random.shuffle(batched_data)
    gan.train(batched_data, mean, scale)
    # gan.train([{
    #     f'output{i}': tf.ones((gan.batch_size, int(math.pow(2,i)))) for i in range(depth)
    # }])
