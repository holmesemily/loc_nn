import tensorflow as tf
import numpy as np
import time, os

DATASET_FOLDER = '../dataset/SSLR'
TRAIN_FOLDER = 'lsp_train_106'
TEST_FOLDER = 'lsp_test_106'
NORM_FOLDER = 'norm'
LABEL_FOLDER = 'label'
ALT_GCC_FOLDER = 'gcc2'
GCCFB_FOLDER = 'gccfb'

AUDIO_FOLDER = 'audio'
GT_FOLDER = 'gt_frame/'

FEATURES_FOLDER = 'features'
PREDICT_FOLDER = 'predict'

ACTIVE_FOLDER = TEST_FOLDER

GENERATE_MODEL = 1
GENERATE_DATASET = 1
MODEL_FIT = 1
PREDICT = 1

def is_within_5deg(y_true, y_pred):
    delta_theta = tf.abs(tf.subtract(y_true[0], y_pred[0]))
    return tf.reduce_mean(tf.cast(tf.less_equal(delta_theta, 5.0/360), tf.float32))

if GENERATE_MODEL:
    model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(51,40,6)),

            tf.keras.layers.Conv2D(filters=12, kernel_size=(5, 5), strides=(2, 2), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv2D(filters=96, kernel_size=(5, 5), strides=(2, 2), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
    ])

    model.summary()
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.MeanSquaredError(), # MAE
                metrics=['accuracy', is_within_5deg])

if GENERATE_DATASET:
    def example_generator_val():
        for filecount, filename in enumerate(os.listdir(os.path.join(DATASET_FOLDER, FEATURES_FOLDER, GCCFB_FOLDER, TEST_FOLDER))):
            cur_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, GCCFB_FOLDER, TEST_FOLDER, filename)
            cur_label = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, LABEL_FOLDER, TEST_FOLDER, filename.split('.')[0] + '.w8192_o4096.csv')
            file = np.genfromtxt(cur_file, delimiter=',')
            label = np.genfromtxt(cur_label, delimiter=',')
            dtype = np.int16
            vmax = np.iinfo(dtype).max
            file = (file / vmax).astype(np.float32)

            for index in range(label.shape[0]):
                line = np.reshape(file, (file.shape[0], 6, 40, 51))
                line = np.transpose(line, axes=[0, 3, 2, 1])[index]
                yield line, label[index]

    def example_generator():
        for filecount, filename in enumerate(os.listdir(os.path.join(DATASET_FOLDER, FEATURES_FOLDER, GCCFB_FOLDER, TRAIN_FOLDER))):
            cur_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, GCCFB_FOLDER, TRAIN_FOLDER, filename)
            cur_label = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, LABEL_FOLDER, TRAIN_FOLDER, filename.split('.')[0] + '.w8192_o4096.csv')
            file = np.genfromtxt(cur_file, delimiter=',')
            label = np.genfromtxt(cur_label, delimiter=',')
            dtype = np.int16
            vmax = np.iinfo(dtype).max
            file = (file / vmax).astype(np.float32)
            for index in range(label.shape[0]):
                line = np.reshape(file, (file.shape[0], 6, 40, 51))
                line = np.transpose(line, axes=[0, 3, 2, 1])[index]
                yield line, label[index]

    # create a dataset from the generator function above
    ds = tf.data.Dataset.from_generator(
        example_generator,
        output_signature=(
        tf.TensorSpec(shape=(51, 40, 6), dtype=tf.float32), 
        tf.TensorSpec(shape=(), dtype=tf.float32))
        )

    ds_val = tf.data.Dataset.from_generator(
        example_generator_val,
        output_signature=(
        tf.TensorSpec(shape=(51, 40, 6), dtype=tf.float32), 
        tf.TensorSpec(shape=(), dtype=tf.float32))
        )
    
    ds = ds.batch(256)
    ds = ds.repeat(10)
    ds_val = ds_val.batch(32)
    # ds = ds.shuffle(3, reshuffle_each_iteration=True)

if MODEL_FIT: 
    epoches = 10
    model.fit(ds, epochs=epoches, verbose=1, steps_per_epoch=30) #, validation_data=ds_val)
    # steps per epoch = samples / batchsize
    # test = 102775
    # train = 183059


if PREDICT:
    print("Predicting...")
    sample = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, GCCFB_FOLDER, TRAIN_FOLDER, "ssl-data_2017-05-13-15-25-43_0.csv")
    x_pred = np.genfromtxt(sample, delimiter=',', skip_header=0, dtype=float)
    x_pred = np.reshape(x_pred, (x_pred.shape[0], 6, 40, 51))
    x_pred = np.transpose(x_pred, axes=[0, 3, 2, 1])
    pred = model.predict(x_pred, verbose=2, steps=2)
    dump_file = os.path.join(DATASET_FOLDER, PREDICT_FOLDER, "pred.csv")
    np.savetxt(dump_file, pred, delimiter = ",")  
