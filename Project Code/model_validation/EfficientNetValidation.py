## Author: Chris Reekie CJR2198
## Script to validate results against official implementation


import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import *


if __name__ == '__main__':

    ## preprocess dataframe of training data ##

    train_df = pd.read_csv('data/1024x1024_train.csv')
    train_df['full_filename'] = train_df['filename']+'.jpg'

    ## split 80%/20%
    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=1337)

    ## Define augmentation Image generator ##
    ## 180 degree rotation, horizontal and vertical flip ##
    ## EfficientNet preprocessing (normalization and std) ##

    generator = preprocessing.image.ImageDataGenerator(
    rotation_range=180,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

    ## Data generator ##
    ## No resizing in generator ##
    trainds = generator.flow_from_dataframe(
        train_df, directory='data/1024x1024', x_col='full_filename', y_col='target',
        weight_col=None, target_size=(1024, 1024), color_mode='rgb',
        classes=None, class_mode='categorical', batch_size=200, shuffle=True, interpolation='nearest')

    valds = generator.flow_from_dataframe(
        valid_df, directory='data/1024x1024', x_col='full_filename', y_col='target',
        weight_col=None, target_size=(1024, 1024), color_mode='rgb',
        classes=None, class_mode='categorical', batch_size=200, shuffle=True, interpolation='nearest')


    ## Define model, random resize crop is done at the entry point to model
    input_layer = layers.Input(shape=(1024, 1024, 3))
    rand_resize_layer = preprocessing.RandomCrop(224, 224)(input_layer)
    output = EfficientNetB0(weights=None, include_top=True, classes=1)(rand_resize_layer)

    model = tf.keras.model(inputs=[input_layer], outputs=[output])

    ## RMSProp optimizer for parity between Pytorch implementation
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9, momentum=0.0)

    ## Binary crossentropy & ROC AUC metric
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

    ## Fit model for 60 epochs
    model_history = model.fit_generator(trainds, steps_per_epoch=len(train_df)//200, epochs=60, verbose=1,
    validation_data=valds, validation_steps=len(valds)//200,
    max_queue_size=10, workers=1, use_multiprocessing=False,
    shuffle=True)

    ## Save results
    historydf = pd.DataFrame(model_history.history)
    historydf.to_csv('OfficialEffNet.csv')