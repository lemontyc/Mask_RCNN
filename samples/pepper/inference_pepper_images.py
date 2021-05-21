#%%
import os
import sys
import shutil
import random
import tensorflow as tf
import skimage.io
import json

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib

from samples.pepper import pepper

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

#%%
config = pepper.PepperConfig()
IMAGES_DIR = os.path.join(ROOT_DIR, "datasets/process")

# %%
# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
# config.display()

# %%
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
class_names = ['BG', 'pepper']

# %%
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# %%
# Or, load the last model you trained
weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# %%
try:
    while True:
        # Get file names in IMAGES_DIR
        file_names = next(os.walk(IMAGES_DIR + '/input'))[2]
        boxes = []

        if not file_names:
            print("Waiting for new image...")
        else:
            file_name = file_names[0]
            file  = file_name.split('.')[0]
            image = skimage.io.imread(os.path.join(IMAGES_DIR + '/input/', file_name))
            results = model.detect([image], verbose=1)
            r = results[0]
            boxes = r['rois']

            with open(IMAGES_DIR + '/boxes/' + file + '.json', 'w') as outfile:
                boxes = boxes.tolist()
                json.dump(boxes, outfile)

            shutil.move(IMAGES_DIR + '/input/' + file_name, IMAGES_DIR + '/processed/' + file_name,)
except KeyboardInterrupt:
    print("GPU is still being used. Please stop container")




# %%
IMAGES_DIR
# %%
