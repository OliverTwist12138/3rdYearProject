import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy import ndimage
import random
def prepare(path):
    apnea_paths = [
        os.path.join(os.getcwd(), path+'apnea/' , x)
        for x in os.listdir(path+'apnea/')
    ]

    nonapnea_paths = [
        os.path.join(os.getcwd(), path+'nonapnea/' , x)
        for x in os.listdir(path+'nonapnea/')
    ]

    print("Apnea samples: " + str(len(apnea_paths)))
    print("Nonapnea Samples: " + str(len(nonapnea_paths)))

    apnea = np.array([np.load(path) for path in apnea_paths])
    nonapnea = np.array([np.load(path) for path in nonapnea_paths])

    apnea_labels = np.array([1 for _ in range(len(apnea))])
    nonapnea_labels = np.array([0 for _ in range(len(nonapnea))])

    x = np.concatenate((apnea, nonapnea), axis=0).astype(np.float32)
    y = np.concatenate((apnea_labels, nonapnea_labels), axis=0)
    print(x.dtype,y.dtype)
    return x, y

@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume

def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def dataLoader(x_train, y_train, x_val, y_val):
    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    batch_size = 2
    # Augment the on the fly during training.
    train_dataset = (
        train_loader.shuffle(len(x_train))
            .map(train_preprocessing)
            .batch(batch_size)
            .prefetch(2)
    )
    # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(len(x_val))
            .map(validation_preprocessing)
            .batch(batch_size)
            .prefetch(2)
    )
    return train_dataset, validation_dataset


def get_model(width=32, height=32, depth=20):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=32, kernel_size=4, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=(1,1,2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=(2,2,1))(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="relu")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def main():
    x_train, y_train = prepare('/home/zceeyan/project/3Ddataset/normalized/training/')
    x_val, y_val = prepare('/home/zceeyan/project/3Ddataset/normalized/validation/')
    print(x_train.dtype)
    print(
        "Number of samples in train and validation are %d and %d."
        % (x_train.shape[0], x_val.shape[0])
    )
    train_dataset, validation_dataset = dataLoader(x_train, y_train, x_val, y_val)

    # Build model.
    model = get_model(width=32, height=32, depth=20)
    model.summary()

    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    # Define callbacks.
    '''
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "./model/3d_image_classification.h5", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
    '''
    my_callbacks = [
        keras.callbacks.ModelCheckpoint("./model/3d_image_classification.h5", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
    ]
    # Train the model, doing validation at the end of each epoch
    epochs = 100
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=my_callbacks,
    )
if __name__ == '__main__':
    main()