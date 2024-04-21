# Retina Segmentation Model

This repository contains code for training and using a convolutional neural network (CNN) model for retinal image segmentation.

## Installation

To use this code, follow these steps:

1. Clone this repository to your local machine:

   ```
   https://github.com/robtad/medical-image-segmentation.git
   ```

2. Navigate to the project directory:

   ```
   cd medical-image-segmentation
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model, follow these steps:

1. Prepare your dataset by placing your original retinal images in the directory `Data/train/image` and their corresponding manually segmented masks in the directory `Data/train/mask`.

2. Run the training script:

   ```
   python task_train_91.py
   ```

   This script will load the images, preprocess them, define the model architecture, compile the model, and start training. You can adjust the number of epochs and batch size in the script as needed.

3. After training, the trained model will be saved as `retina_segmentation_model_91.h5` in the project directory.

### Using the Trained Model

To use the trained model for segmentation, you can run the following python file:

```
  python test_model.py
```
