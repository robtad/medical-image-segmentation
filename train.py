import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from data_loader import load_data
from model import build_unet

# Parameters
input_shape = (256, 256, 1)
batch_size = 16
epochs = 50
learning_rate = 1e-4
data_dir = 'Data'

# Load data
x_train, y_train = load_data(data_dir, mode='train')
x_test, y_test = load_data(data_dir, mode='test')

# Build model
model = build_unet(input_shape)
model.compile(optimizer=Adam(learning_rate), loss=binary_crossentropy, metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Save model
model.save('retina_blood_vessel_segmentation_model.h5')
