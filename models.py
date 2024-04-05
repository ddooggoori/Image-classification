import numpy as np
import pandas as pd
import tensorflow as tf
import os 
import time
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import *
from tqdm.keras import TqdmCallback 
# Setting CUDA environment
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def train_model():

    start = time.time()

    # Loading data
    X_train1 = np.load('BP1_array.npy', allow_pickle=True) / 255
    X_data = np.load('BP2_array.npy', allow_pickle=True) / 255

    X_train2 = X_data[:3000]
    X_test = X_data[3000:4000]
    X_val = X_data[4000:]

    X_train = np.concatenate((X_train1, X_train2))

    y_train1 = pd.read_csv(r'JSON/BP1_json_resized.csv')
    y_data = pd.read_csv(r'JSON/BP2_json_resized.csv')

    y = pd.concat([y_train1, y_data])

    y_train = y.iloc[:8000].drop('filename', axis=1).to_numpy()
    y_val = y.iloc[9000:].drop('filename', axis=1).to_numpy()
    y_test = y.iloc[8000:9000].drop('filename', axis=1).to_numpy()

    # Hyperparameters
    BATCH_SIZE = 128
    AUTOTUNE = tf.data.AUTOTUNE

    # Creating datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.batch(BATCH_SIZE, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)
    valid_dataset = valid_dataset.batch(BATCH_SIZE, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)

    # Building the model
    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(4, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compiling the model
    model.compile(optimizer='adam', loss='mse')

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    # Training the model
    history = model.fit(train_dataset, validation_data=valid_dataset, epochs=1000, callbacks=[early_stopping])

    # Evaluation
    mae = model.evaluate(test_dataset, verbose=0)
    rmse = np.sqrt(mae)
    print("MAE : {:.4f}".format(mae))
    print("RMSE : {:.4f}".format(rmse))

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    return model.predict(test_dataset)




def train_image_classifier(train_path, test_path, val_path):
    """
    Trains an image classifier using Keras and TensorFlow.

    Args:
    - train_path (str): Path to the directory containing training images.
    - test_path (str): Path to the directory containing test images.
    - val_path (str): Path to the directory containing validation images.

    Returns:
    - model: Trained TensorFlow model.
    """

    # Define a function to create image data generators
    def generator(path):
        # Create an ImageDataGenerator with rescaling
        datagen = ImageDataGenerator(rescale=1. / 255)
        # Generate batches of augmented/normalized data from the directory
        gen = datagen.flow_from_directory(path, target_size=(300, 300),
                                          batch_size=32, class_mode='categorical')
        return gen

    # Create generators for training, testing, and validation data
    train_gen = generator(train_path)
    test_gen = generator(test_path)
    val_gen = generator(val_path)

    # Define early stopping callback
    callbacks = [EarlyStopping(monitor='val_loss', patience=50)]

    # Define the CNN model architecture using the Sequential API
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=[CategoricalAccuracy(), AUC(), Recall(), Precision()])

    # Train the model
    model.fit(train_gen, epochs=1000, batch_size=32, validation_data=val_gen, callbacks=callbacks)

    # Evaluate the model on test data
    evaluate = model.evaluate(test_gen)

    return model
