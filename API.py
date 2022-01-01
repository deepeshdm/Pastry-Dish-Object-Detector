import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np


def detect_pastries(image_path, saved_model_path, labelMap_path, min_threshold=0.5, image_save_path=None):
    """
    Parameters
    ----------
    image_path : Path to input image for the model
    saved_model_path : Path to pre-trained model
    labelMap_path : Path to 'label_map.pbtxt' file
    min_threshold : Minimum decision threshold for classification
    image_save_path : Path to save the output image

    Returns : Displays output Image with Bounding Boxes in a window
    -------

    """

    # -------------------------------------------------------------------------------

    # Object Classes
    OBJECT_LABELS = {1: 'Cutlet', 2: 'Egg Patis', 3: 'Fish Bun', 4: 'Fish Roti', 5: 'Kimbula Bun',
                     6: 'Puff Pastry', 7: 'Roll', 8: 'Sausage Bun', 9: 'Uludu Wade', 10: 'Vegetable Roti'}

    print('Loading model...this will take a minute')
    start_time = time.time()

    # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
    detect_fn = tf.saved_model.load(saved_model_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done ! Loading model took {} seconds'.format(round(elapsed_time, 3)))

    # -------------------------------------------------------------------------------

    # LOAD LABEL MAP DATA FOR PLOTTING
    category_index = label_map_util.create_category_index_from_labelmap(labelMap_path,
                                                                        use_display_name=True)

    # -------------------------------------------------------------------------------

    print('Running inference for {}... '.format(image_path), end='')

    image = cv2.imread(image_path)

    print("\n Resizing Image to 416x416...")
    image = cv2.resize(image,(416,416))


    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)

    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_with_detections = image.copy()

    # -------------------------------------------------------------------------------

    # print first 10 objects detected with scores
    print("\n----------TOP 10 DETECTED OBJECTS WITH SCORES----------")
    for i in range(11):
        obj_class = OBJECT_LABELS.get(detections['detection_classes'][i])
        obj_score = detections['detection_scores'][i]
        print(obj_class + " : " + str(round(obj_score, 3)))
    print("NOTE : Objects with score below threshold are not drawn.")
    print("-------------------------------------------------------")

    # -------------------------------------------------------------------------------

    # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=100,
        min_score_thresh=min_threshold,
        agnostic_mode=False)

    # save image if path given
    if image_save_path is not None:
        image = np.array(image_with_detections)
        image = image[:, :, ::-1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_save_path, image)
        print("Saving the output image...Done")

    # DISPLAYS OUTPUT IMAGE
    cv2.imshow("OUTPUT", image_with_detections)
    cv2.waitKey()


def detect_objects(image_path, saved_model_path, labelMap_path, min_threshold=0.5):
    """
    Parameters
    ----------
    image_path : Path to input image for the model
    saved_model_path : Path to pre-trained model
    labelMap_path : Path to 'label_map.pbtxt' file
    min_threshold : Minimum decision threshold for classification

    Returns : Numpy array of output Image & List of Detected Objects with scores
    -------
    """

    # -------------------------------------------------------------------------------

    # Object Classes
    OBJECT_LABELS = {1: 'Cutlet', 2: 'Egg Patis', 3: 'Fish Bun', 4: 'Fish Roti', 5: 'Kimbula Bun',
                     6: 'Puff Pastry', 7: 'Roll', 8: 'Sausage Bun', 9: 'Uludu Wade', 10: 'Vegetable Roti'}

    print('Loading model...this will take a minute')
    start_time = time.time()

    # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
    detect_fn = tf.saved_model.load(saved_model_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done ! Loading model took {} seconds'.format(round(elapsed_time, 3)))

    # -------------------------------------------------------------------------------

    # LOAD LABEL MAP DATA FOR PLOTTING
    category_index = label_map_util.create_category_index_from_labelmap(labelMap_path,
                                                                        use_display_name=True)

    # -------------------------------------------------------------------------------

    print('Running inference for {}... '.format(image_path), end='')

    image = cv2.imread(image_path)

    print("\n Resizing Image to 416x416...")
    image = cv2.resize(image,(416,416))

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)

    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_with_detections = image.copy()

    # -------------------------------------------------------------------------------

    detected_objects = {}

    # print first 10 objects detected with scores
    print("\n----------TOP 10 DETECTED OBJECTS WITH SCORES----------")
    for i in range(11):
        obj_class = OBJECT_LABELS.get(detections['detection_classes'][i])
        obj_score = detections['detection_scores'][i]
        print(obj_class + " : " + str(round(obj_score, 3)))

        # add data to dictionary
        detected_objects[i] = {obj_class: str(round(obj_score, 3))}

    print("NOTE : Objects with score below threshold are not drawn.")
    print("-------------------------------------------------------")

    # -------------------------------------------------------------------------------

    # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=100,
        min_score_thresh=min_threshold,
        agnostic_mode=False)

    image_with_detections = np.array(image_with_detections)

    return image_with_detections, detected_objects


# ------------------------------CODE RELATED TO WEBCAM OBJECT DETECTION------------------------------------#


def process_frame(image, detect_function, labelMap_path, min_threshold=0.5):
    """
    Parameters
    ----------
    image : Numpy array Image frame
    detect_function : Loaded model detection function
    labelMap_path : Path to 'label_map.pbtxt' file
    min_threshold : Minimum decision threshold for classification

    Returns : Numpy array of output Image
    -------
    """

    # -------------------------------------------------------------------------------

    # LOAD LABEL MAP DATA FOR PLOTTING
    category_index = label_map_util.create_category_index_from_labelmap(labelMap_path,
                                                                        use_display_name=True)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)

    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_function(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # -------------------------------------------------------------------------------

    # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=100,
        min_score_thresh=min_threshold,
        agnostic_mode=False)

    return image


def detect_webcam(detection_function, labelMap_path, min_threshold=0.5):

    print("Starting webcam...")

    # define a video capture object
    vid = cv2.VideoCapture(0)

    while (True):

        # Capture the video frame by frame
        ret, frame = vid.read()

        # ------------------------------------------------------------

        image = cv2.resize(frame, (416, 416))

        # # Numpy array of output Image and Dictionary of Top 10 detected objects with scores
        Output_Image = process_frame(image, detection_function, labelMap_path, min_threshold)

        # ------------------------------------------------------------

        # resizing the output image
        frame = cv2.resize(Output_Image, (620, 620))

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
