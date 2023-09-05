"""
fc_model.py

Written by Emily Holmes during a LAAS-CNRS internship, 2023
Simple feed-forward fully connected model for SSL estimations
"""

# Imports
import tensorflow as tf
import numpy as np
import os

# Parameters
DATASET_FOLDER = '../dataset/SSLR'
TRAIN_FOLDER = 'lsp_train_106'
TEST_FOLDER = 'lsp_test_106'
LABEL_FOLDER = 'label'
ALT_GCC_FOLDER = 'gcc'

AUDIO_FOLDER = 'audio'
GT_FOLDER = 'gt_frame/'

FEATURES_FOLDER = 'features'
PREDICT_FOLDER = 'predict'

ACTIVE_FOLDER = TEST_FOLDER

GENERATE_MODEL = 1
GENERATE_DATASET = 1
MODEL_FIT = 1
PREDICT = 1
PREDIT_DETEC = 0            # Detection has its own constant, as I got the model right pretty earlier on
                            # and didn't want to re-train it every time 

'''
Custom metric function 

Compares the L2 norm of the prediction and ground truth. 
Returns 100% if 100% of predicted values are within a 20deg margin
'''
def is_within_20deg(y_true, y_pred):
    delta_theta = tf.abs(tf.subtract(y_true[0], y_pred[0]))
    return tf.reduce_mean(tf.cast(tf.less_equal(delta_theta, 20.0/360), tf.float32))


'''
Custom learning rate 

Returns a slower learning rate as epoches go on, allows for better convergence of the results
'''
def lr_scheduler(epoch, lr):
    if epoch > 1 & (epoch % 3) == 0:
        lr = 0.01/(2*(epoch-1)) 
        return lr
    return lr


if GENERATE_MODEL:
    '''
    Model generation - Predicts the azimuth
    Model is sequential (one input, one output)

    Input: the 51 center delays ([-25,25]) for the generalised cross-correlation (GCC-PHAT)
    between 4 microphones (totalling 6 different combinations)

    Structure:
    Fully connected layers with tanh activation, ending with a sigmoid activation

    Returns: a prediction between 0 (0°) and 1 (360°)
    '''
    model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(51,6)),
            tf.keras.layers.Dense(300, activation = 'tanh'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(300, activation = 'tanh'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(51, activation = 'tanh'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation = 'sigmoid'),

    ])

    model.summary()
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                loss=tf.keras.losses.MeanAbsoluteError(), 
                metrics=['accuracy', is_within_20deg])

    if PREDIT_DETEC:
        '''
        Model generation - Detect audio activity
        Model is sequential (one input, one output)

        Input: the 51 center delays ([-25,25]) for the generalised cross-correlation (GCC-PHAT)
        between 4 microphones (totalling 6 different combinations)

        Structure:
        Fully connected layers with tanh activation, ending with a sigmoid activation

        Returns: a prediction between 0 (0% chance of an audio activity) to 1 (100% chance of an audio activity)
        '''
        model_detec = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(51,6)),
                tf.keras.layers.Dense(300, activation = 'tanh'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(1, activation = 'sigmoid'),
        ])
        
        model_detec.summary()
        model_detec.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01, epsilon = 0.1),
                    loss=tf.keras.losses.BinaryCrossentropy(), 
                    metrics=['accuracy'])
    
if GENERATE_DATASET:

    # yield an input frame (51,6) and the ground truth azimuth
    def generator_doa():
        for filecount, filename in enumerate(os.listdir(os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ALT_GCC_FOLDER, TRAIN_FOLDER))):
            cur_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ALT_GCC_FOLDER, TRAIN_FOLDER, filename)
            cur_label = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, LABEL_FOLDER, TRAIN_FOLDER, filename.split('.')[0] + '.w8192_o4096.csv')
            file = np.genfromtxt(cur_file, delimiter=',')
            label = np.genfromtxt(cur_label, delimiter=',')
            for index in range(label.shape[0]):
                line = np.reshape(file, (file.shape[0], 51, 6))[index]
                yield line, label[index][1]

    # create a dataset from the generator function above
    ds = tf.data.Dataset.from_generator(
        generator_doa,
        output_signature=(
        tf.TensorSpec(shape=(51, 6), dtype=tf.float32), 
        tf.TensorSpec(shape=(), dtype=tf.float32))
        )
    ds = ds.batch(256)
    ds = ds.repeat(10)
    ds = ds.shuffle(3, reshuffle_each_iteration=True)

    if PREDIT_DETEC:
        # yield an input frame (51,6) and the ground truth activation
        def generator_detec():
            for filecount, filename in enumerate(os.listdir(os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ALT_GCC_FOLDER, TRAIN_FOLDER))):
                cur_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ALT_GCC_FOLDER, TRAIN_FOLDER, filename)
                cur_label = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, LABEL_FOLDER, TRAIN_FOLDER, filename.split('.')[0] + '.w8192_o4096.csv')
                file = np.genfromtxt(cur_file, delimiter=',')
                label = np.genfromtxt(cur_label, delimiter=',')
                for index in range(label.shape[0]):
                    line = np.reshape(file, (file.shape[0], 51, 6))[index]
                    yield line, int(label[index][0])

        ds_detec = tf.data.Dataset.from_generator(
            generator_detec,
            output_signature=(
            tf.TensorSpec(shape=(51, 6), dtype=tf.float32), 
            tf.TensorSpec(shape=(), dtype=tf.int8))
            )
        ds_detec = ds_detec.batch(256)
        ds_detec = ds_detec.repeat(10)
        ds_detec = ds_detec.shuffle(3, reshuffle_each_iteration=True)

'''
Train the model

Frames in the training dataset = 183059
Frames in the testing dataset = 102775
Steps per epoch = samples / batch_size
'''
if MODEL_FIT: 
    epoches = 15
    callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)]

    model.fit(ds, epochs=epoches, batch_size=64, verbose=2, steps_per_epoch=450, callbacks = callbacks) 

    if PREDIT_DETEC:
        model.fit(ds, epochs=epoches, batch_size=64, verbose=2, steps_per_epoch=450, callbacks = callbacks) 


'''
Predict and output prediction for comparison

Prediction is done on only one file for quick computation and comparison
'''
if PREDICT:
    print("Predicting...")
    sample = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ALT_GCC_FOLDER, TRAIN_FOLDER, "ssl-data_2017-05-13-15-25-43_0.csv")
    x_pred = np.genfromtxt(sample, delimiter=',', skip_header=0, dtype=float)
    x_pred = np.reshape(x_pred, (x_pred.shape[0], 51,6))
    
    pred = model.predict(x_pred, verbose=2, steps=2)
    dump_file = os.path.join(DATASET_FOLDER, PREDICT_FOLDER, "pred_doa.csv")
    np.savetxt(dump_file, pred, delimiter = ",")  

    if PREDIT_DETEC:
        pred_detec = model_detec.predict(x_pred, verbose=2, steps=2)
        dump_file_detec = os.path.join(DATASET_FOLDER, PREDICT_FOLDER, "pred_detec.csv")
        np.savetxt(dump_file_detec, pred_detec, delimiter = ",")  
