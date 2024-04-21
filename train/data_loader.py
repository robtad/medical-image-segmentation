import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize



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
