import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import time
import tensorflow as tf
import cv2
import numpy as np

from utils import label_map_util
from utils import visualization_utils

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
if cv2.__version__ < '4.5.1':
    print('Warning: This has only been tested for OpenCV 4.5.1.')

MODEL_NAME = 'ssd_mobilenet_v2_fpn_320'
PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(), 'model_data', MODEL_NAME, 'saved_model')
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(os.getcwd(), 'model_data', MODEL_NAME, 'label_map.pbtxt')
NUM_CLASSES = 1

if __name__ == '__main__':
    print("Loading saved model ...")
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    print("Model Loaded!")

    # Load label map and obtain class names and ids
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Open Video Capture (Camera)
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not video_capture.isOpened():
        raise SystemError('No video camera found')

    tic = time.time()
    while True:
      ret, frame = video_capture.read()
      if not ret:
          print('Error reading frame from camera. Exiting ...')
          break
          
      image_np = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      input_tensor = tf.convert_to_tensor(image_np)
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

      image_np_with_detections = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

      visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

      toc = time.time()
      fps = int(1/(toc - tic))
      tic = toc
      cv2.putText(image_np_with_detections, f"FPS: {fps}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
      cv2.imshow("img", image_np_with_detections)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print("Exiting ...")
    video_capture.release()
    cv2.destroyAllWindows()