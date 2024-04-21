import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img

# Define paths
original_images_path = 'Data/train/image'
manual_segmented_images_path = 'Data/train/mask'

# Get list of image files
original_images_files = os.listdir(original_images_path)
manual_segmented_images_files = os.listdir(manual_segmented_images_path)

# Initialize lists to store images
original_images = []
manual_segmented_images = []

# Load original images
for image_file in original_images_files:
    image = load_img(os.path.join(original_images_path, image_file), color_mode='grayscale')
    image = img_to_array(image)
    original_images.append(image)

# Load manual segmented images
for image_file in manual_segmented_images_files:
    image = load_img(os.path.join(manual_segmented_images_path, image_file), color_mode='grayscale')
    image = img_to_array(image)
    manual_segmented_images.append(image)

# Convert lists to numpy arrays
original_images = np.array(original_images, dtype='float') / 255
manual_segmented_images = np.array(manual_segmented_images, dtype='float') / 255

# Define the model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(None, None, 1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(original_images, manual_segmented_images, epochs=50, batch_size=2, shuffle=True)

model.save('retina_segmentation_model_91.h5')
