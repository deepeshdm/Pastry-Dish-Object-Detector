#
# # Path to Input Image
# IMAGE_PATH = 'C:/Users/dipesh/Desktop/fiverr-hasijayawardana/train/IMG_0008.JPG'
#
# # Path to pre-trained model
# SAVED_MODEL_PATH = r'C:/Users/dipesh/Desktop/fiverr-hasijayawardana/Pastry Detection/saved_model'
#
# # Path to 'label_map.pbtxt' file
# PATH_TO_LABELS = 'C:/Users/dipesh/Desktop/fiverr-hasijayawardana/Pastry Detection/label_map.pbtxt'
#
# # Object detection threshold
# THRESHOLD = 0.5
#
# # Path to save the output Image with name and extension
# SAVE_PATH = r"C:/Users/dipesh/Desktop/fiverr-hasijayawardana/outputs/output.jpeg"
#
# from API import detect_pastries
# detect_pastries(IMAGE_PATH, SAVED_MODEL_PATH, PATH_TO_LABELS,THRESHOLD,SAVE_PATH)

# ------------------------------------------------------------------------------------------

# Path to Input Image
IMAGE_PATH = 'C:/Users/dipesh/Desktop/fiverr-hasijayawardana/train/IMG_0008.JPG'

# Path to pre-trained model
SAVED_MODEL_PATH = r'C:/Users/dipesh/Desktop/fiverr-hasijayawardana/Pastry Detection/saved_model'

# Path to 'label_map.pbtxt' file
PATH_TO_LABELS = 'C:/Users/dipesh/Desktop/fiverr-hasijayawardana/Pastry Detection/label_map.pbtxt'

# Object detection threshold
THRESHOLD = 0.5

# Path to save the output Image with name and extension
SAVE_PATH = r"C:/Users/dipesh/Desktop/fiverr-hasijayawardana/outputs/output.jpeg"

import cv2
from API import detect_objects

# Numpy array of output Image and Dictionary of Top 10 detected objects with scores
Output_Image, Detected_Objects = detect_objects(IMAGE_PATH, SAVED_MODEL_PATH, PATH_TO_LABELS, THRESHOLD)

print(Detected_Objects)
cv2.imshow("OUTPUT", Output_Image)
cv2.waitKey()
