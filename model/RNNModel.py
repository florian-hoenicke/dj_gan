import tensorflow as tf


def create(sequence_length=100, max_tones=12, max_octaves=20, rnn_units=256, hidden_dense_size=100, tones_rnn_dim=32,
           octaves_rnn_dim=32, batch_size=None):
    instrument_type_in = tf.keras.layers.Input(shape=(sequence_length,), name='instrument_type', batch_size=batch_size)
    is_drum_in = tf.keras.layers.Input(shape=(sequence_length,), name='is_drum', batch_size=batch_size)

    instrument_type_enbedding = tf.keras.layers.Embedding(150, output_dim=20, name='instrument_embedding')(
        instrument_type_in)
    # instrument_type_enbedding = tf.keras.backend.repeat_elements(instrument_type_enbedding, sequence_length, 1)

    is_drum_embedding = tf.keras.layers.Embedding(2, output_dim=20, name='is_drum_embedding')(is_drum_in)
    # is_drum_embedding = tf.keras.backend.repeat_elements(is_drum_embedding, sequence_length, 1)

    durations_in, velocities_in, tones_in, octaves_in = [tf.keras.layers.Input(shape=(sequence_length,), name=name, batch_size=batch_size) for
                                                         name in
                                                         ['durations_in', 'velocities_in', 'tones_in', 'octaves_in']]
    durations_reshaped, velocities_reshaped = [tf.keras.layers.Reshape(target_shape=(sequence_length, 1))(x) for x in
                                               [durations_in, velocities_in]]
    tones_embedding = tf.keras.layers.Embedding(max_tones, output_dim=tones_rnn_dim)(tones_in)
    octaves_embedding = tf.keras.layers.Embedding(max_octaves, output_dim=octaves_rnn_dim)(octaves_in)
    #
    gru_states = [
        tf.keras.layers.GRU(
            units=rnn_units,
            return_sequences=True
        )(x) for x in [durations_reshaped, velocities_reshaped, tones_embedding, octaves_embedding]
    ]
    concatenation = tf.keras.layers.concatenate(gru_states + [instrument_type_enbedding, is_drum_embedding])
    hidden1 = tf.keras.layers.Dense(
        units=hidden_dense_size,
        activation='relu'
    )(concatenation)

    durations_out = tf.keras.layers.Dense(units=1, name='durations_out')(hidden1)
    velocity_out = tf.keras.layers.Dense(units=1, name='velocity_out')(hidden1)
    tones_out = tf.keras.layers.Dense(units=max_tones, activation='softmax', name='tones_out')(hidden1)
    octaves_out = tf.keras.layers.Dense(units=max_octaves, activation='softmax', name='octaves_out')(hidden1)

    model = tf.keras.models.Model(
        inputs={
            'instrument_type': instrument_type_in,
            'is_drum': is_drum_in,
            'durations_in': durations_in,
            'velocity_in': velocities_in,
            'tones_in': tones_in,
            'octaves_in': octaves_in
        },
        outputs={
            'durations_out': durations_out,
            'velocity_out': velocity_out,
            'tones_out': tones_out,
            'octaves_out': octaves_out
        },
    )
    tf.keras.utils.plot_model(
        model,
        'data/model/rnn.png',
        show_layer_names=True,
        show_shapes=True,
    )
    model.compile(
        optimizer='rmsprop',
        loss={
            'durations_out': tf.keras.losses.MeanSquaredError(),
            'velocity_out': tf.keras.losses.MeanSquaredError(),
            'tones_out': tf.keras.losses.SparseCategoricalCrossentropy(),
            'octaves_out': tf.keras.losses.SparseCategoricalCrossentropy()
        }
    )
    return model
