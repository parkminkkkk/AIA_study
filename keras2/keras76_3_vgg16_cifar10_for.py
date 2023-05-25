import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train / 255.
x_test = x_test / 255.

# Define the configurations to compare
configurations = [
    {"name": "Configuration 1", "trainable": False, "layer": "Flatten"},
    {"name": "Configuration 2", "trainable": False, "layer": "GlobalAveragePooling2D"},
    {"name": "Configuration 3", "trainable": True, "layer": "Flatten"},
    {"name": "Configuration 4", "trainable": True, "layer": "GlobalAveragePooling2D"},
]

# Loop over configurations
for config in configurations:
    print(config["name"])

    # Build the model
    vgg16 = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(32, 32, 3),
    )
    vgg16.trainable = config["trainable"]

    model = Sequential()
    model.add(vgg16)

    if config["layer"] == "Flatten":
        model.add(Flatten())
    elif config["layer"] == "GlobalAveragePooling2D":
        model.add(GlobalAveragePooling2D())

    model.add(Dense(100))
    model.add(Dense(10, activation="softmax"))

    # Compile the model
    learning_rate = 0.1
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["acc"],
    )

    # Define callbacks
    es = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        verbose=1,
    )
    rlr = ReduceLROnPlateau(
        monitor="val_loss",
        patience=10,
        mode="auto",
        verbose=1,
        factor=0.5,
    )

    # Train the model
    model.fit(
        x_train,
        y_train,
        epochs=50,
        batch_size=512,
        verbose=1,
        validation_split=0.2,
        callbacks=[es, rlr],
    )

    # Evaluate the model
    results = model.evaluate(x_test, y_test)
    print("Test loss:", results[0])
    print("Test accuracy:", results[1])
    # print()

    '''
#1. {"name": "Configuration 1", "trainable": False, "layer": "Flatten"},
Epoch 50/50
79/79 [==============================] - 3s 41ms/step - loss: 1.0981 - acc: 0.6201 - val_loss: 1.2240 - val_acc: 0.5803 - lr: 0.0125
313/313 [==============================] - 4s 12ms/step - loss: 1.2327 - acc: 0.5772
Test loss: 1.2326778173446655
Test accuracy: 0.5771999955177307

#2. {"name": "Configuration 2", "trainable": False, "layer": "GlobalAveragePooling2D"},
Epoch 50/50
79/79 [==============================] - 3s 41ms/step - loss: 1.1023 - acc: 0.6193 - val_loss: 1.2142 - val_acc: 0.5835 - lr: 0.0125
313/313 [==============================] - 3s 9ms/step - loss: 1.2267 - acc: 0.5810
Test loss: 1.226719617843628
Test accuracy: 0.5809999704360962

#3. {"name": "Configuration 3", "trainable": True, "layer": "Flatten"},
Epoch 00020: early stopping
313/313 [==============================] - 3s 9ms/step - loss: nan - acc: 0.1000
Test loss: nan
Test accuracy: 0.10000000149011612

#4. {"name": "Configuration 4", "trainable": True, "layer": "GlobalAveragePooling2D"},
Epoch 00020: early stopping
313/313 [==============================] - 3s 9ms/step - loss: nan - acc: 0.1000
Test loss: nan
Test accuracy: 0.10000000149011612
    '''