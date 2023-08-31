import tensorflow as tf
import numpy as np
import time, os

DATASET_FOLDER = '../dataset/SSLR'
TRAIN_FOLDER = 'lsp_train_106'
TEST_FOLDER = 'lsp_test_106'
NORM_FOLDER = 'norm'
LABEL_FOLDER = 'label2'
ALT_GCC_FOLDER = 'gcc2'

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

def lr_scheduler(epoch, lr):
    if epoch > 1 & (epoch % 2) == 0:
        lr = lr/2
        return lr
    return lr

if GENERATE_MODEL:
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
                loss=tf.keras.losses.MeanAbsoluteError(), #MAE
                metrics=['accuracy'])

    model_detec = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(51,6)),
            tf.keras.layers.Dense(300, activation = 'tanh'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation = 'sigmoid'),
    ])
    
    model_detec.summary()
    model_detec.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1, epsilon = 0.1),
                loss=tf.keras.losses.BinaryCrossentropy(), #MAE
                metrics=['accuracy'])
    
if GENERATE_DATASET:

    def example_generator():
        for filecount, filename in enumerate(os.listdir(os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ALT_GCC_FOLDER, TRAIN_FOLDER))):
            cur_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ALT_GCC_FOLDER, TRAIN_FOLDER, filename)
            cur_label = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, LABEL_FOLDER, TRAIN_FOLDER, filename.split('.')[0] + '.w8192_o4096.csv')
            file = np.genfromtxt(cur_file, delimiter=',')
            label = np.genfromtxt(cur_label, delimiter=',')
            for index in range(label.shape[0]):
                line = np.reshape(file, (file.shape[0], 51, 6))[index]
                yield line, label[index][1]

    def example_generator_detec():
        for filecount, filename in enumerate(os.listdir(os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ALT_GCC_FOLDER, TRAIN_FOLDER))):
            cur_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ALT_GCC_FOLDER, TRAIN_FOLDER, filename)
            cur_label = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, LABEL_FOLDER, TRAIN_FOLDER, filename.split('.')[0] + '.w8192_o4096.csv')
            file = np.genfromtxt(cur_file, delimiter=',')
            label = np.genfromtxt(cur_label, delimiter=',')
            for index in range(label.shape[0]):
                line = np.reshape(file, (file.shape[0], 51, 6))[index]
                yield line, int(label[index][0])


    # create a dataset from the generator function above
    ds = tf.data.Dataset.from_generator(
        example_generator,
        output_signature=(
        tf.TensorSpec(shape=(51, 6), dtype=tf.float32), 
        tf.TensorSpec(shape=(), dtype=tf.float32))
        )

    ds_detec = tf.data.Dataset.from_generator(
        example_generator_detec,
        output_signature=(
        tf.TensorSpec(shape=(51, 6), dtype=tf.float32), 
        tf.TensorSpec(shape=(), dtype=tf.int8))
        )

    # ds_val = tf.data.Dataset.from_generator(
    #     example_generator_val,
    #     output_signature=(
    #     tf.TensorSpec(shape=(51, 6), dtype=tf.float32), 
    #     tf.TensorSpec(shape=(), dtype=tf.float32))
    #     )
    
    ds = ds.batch(256)
    ds = ds.repeat(10)
    
    ds_detec = ds_detec.batch(256)
    ds_detec = ds_detec.repeat(10)
    # ds_val = ds_val.batch(32)
    ds = ds.shuffle(3, reshuffle_each_iteration=True)

if MODEL_FIT: 
    epoches = 15
    callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)]

    model.fit(ds, epochs=epoches, batch_size=64, verbose=2, steps_per_epoch=100, callbacks = callbacks) #, validation_data=ds_val)
    # model_detec.fit(ds_detec, epochs=epoches, verbose=2, steps_per_epoch=30) #, validation_data=ds_val)
    # steps per epoch = samples / batchsize
    # test = 102775
    # train = 183059

# cnt_samples=0
# for filecount2, filename2 in enumerate(os.listdir(os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ALT_GCC_FOLDER, TRAIN_FOLDER))):
#     cur_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ALT_GCC_FOLDER, TRAIN_FOLDER, filename2)
#     file = np.genfromtxt(cur_file, delimiter=',')
#     cnt_samples += file.shape[0]

# print(cnt_samples)


if PREDICT:
    print("Predicting...")
    # model = tf.keras.models.load_model(MODEL_FOLDER + '.h5')
    sample = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ALT_GCC_FOLDER, TRAIN_FOLDER, "ssl-data_2017-05-13-15-25-43_0.csv")
    x_pred = np.genfromtxt(sample, delimiter=',', skip_header=0, dtype=float)
    x_pred = np.reshape(x_pred, (x_pred.shape[0], 51,6))
    pred = model.predict(x_pred, verbose=2, steps=2)
    dump_file = os.path.join(DATASET_FOLDER, PREDICT_FOLDER, "pred_doa.csv")

    # pred_detec = model_detec.predict(x_pred, verbose=2, steps=2)
    # dump_file_detec = os.path.join(DATASET_FOLDER, PREDICT_FOLDER, "pred_detec.csv")

    np.savetxt(dump_file, pred, delimiter = ",")  
    # np.savetxt(dump_file_detec, pred_detec, delimiter = ",")  
