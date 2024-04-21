import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('retina_segmentation_model_91.h5')

# Load an image for segmentation
image_path = 'Data/test/image/0.png'
image = load_img(image_path, color_mode='grayscale')
image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0) / 255

# Perform segmentation
segmented_image = model.predict(image_array)[0]

# Post-process the segmented image as needed
# (e.g., thresholding, applying morphological operations)

# Display the original and segmented images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Segmented image
plt.subplot(1, 2, 2)
plt.imshow(segmented_image.squeeze(), cmap='gray')
plt.title('Segmented Image')
plt.axis('off')

plt.show()