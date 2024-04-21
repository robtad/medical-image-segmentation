import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

from keras.models import load_model
from keras.optimizers import Adam

# Load the trained model without compiling
model = load_model('retina_segmentation_model_91.h5', compile=False)

# Compile the model manually
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])



# Define paths for test data
test_images_path = 'Data/test/image'
test_masks_path = 'Data/test/mask'

# Get list of test image files
test_images_files = os.listdir(test_images_path)
test_masks_files = os.listdir(test_masks_path)

# Initialize lists to store test images and masks
test_images = []
test_masks = []

# Load test images
for image_file in test_images_files:
    image = load_img(os.path.join(test_images_path, image_file), color_mode='grayscale')
    image = img_to_array(image)
    test_images.append(image)

# Load test masks
for image_file in test_masks_files:
    image = load_img(os.path.join(test_masks_path, image_file), color_mode='grayscale')
    image = img_to_array(image)
    test_masks.append(image)

# Convert lists to numpy arrays and normalize
test_images = np.array(test_images, dtype='float') / 255
test_masks = np.array(test_masks, dtype='float') / 255

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_masks, batch_size=2)

print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')

# Output:
# Test loss: 0.1055150181055069    
# Test accuracy: 0.9185224771499634
