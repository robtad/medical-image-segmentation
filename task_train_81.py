import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def build_unet(input_shape):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    up3 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv2))
    merge3 = concatenate([conv1, up3], axis=3)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(merge3)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(conv3)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv3)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def load_data(data_dir, mode="train"):
    image_dir = os.path.join(data_dir, mode, "image")
    mask_dir = os.path.join(data_dir, mode, "mask")

    images = []
    masks = []

    for filename in sorted(os.listdir(image_dir)):
        img = load_img(os.path.join(image_dir, filename), color_mode="grayscale")
        mask = load_img(os.path.join(mask_dir, filename), color_mode="grayscale")

        img = img_to_array(img) / 255.0
        mask = img_to_array(mask) / 255.0

        img = resize(img, [256, 256])
        mask = resize(mask, [256, 256])

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)


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
model.save('retina_segmentation_model_81.h5')
